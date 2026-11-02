"""验 answer_only_loss 与 HF 内置 out.loss 数学等价。

为什么必须先在小模型上验
------------------------
这个函数唯一的失效模式是**对齐错位** —— 比如正文忘了取 [:, :-1]，
或者标签忘了取 [:, 1:]。错位不会抛异常，loss 曲线看起来完全正常
（只是数值略有不同），训出来的模型却是错的。等发现时已经烧掉几天。

所以：先用一个真 HF 因果 LM（结构极小、CPU、秒级）把「同一个数」这件事钉死，
再上 35B。训练脚本里的 --verify-loss 是第二道闸，负责验 35B 上那条具体接线。

跑法： python scripts/_scratch/_test_answer_only_loss.py
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FAILS = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "route_c_trainer", ROOT / "scripts" / "50_routeC_e2e.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny_causal_lm(vocab=97, d=32, nl=2):
    """真 HF 因果 LM。优先 Qwen3（与 35B 同族），没有再退 GPT2。"""
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=vocab, hidden_size=d, num_hidden_layers=nl,
                          num_attention_heads=4, num_key_value_heads=2,
                          intermediate_size=d * 2)
        return Qwen3ForCausalLM(cfg), "Qwen3ForCausalLM"
    except Exception:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=vocab, n_embd=d, n_layer=nl, n_head=4)
        return GPT2LMHeadModel(cfg), "GPT2LMHeadModel"


def main() -> int:
    m = load_trainer()
    torch.manual_seed(0)

    net, name = tiny_causal_lm()
    net.eval()
    model = SimpleNamespace(llm=net)
    check(hasattr(m, "answer_only_loss"), "拿到 answer_only_loss")

    print(f"\n== 1. _lm_parts 认出结构（当前 {name}）==")
    body, head = m._lm_parts(model)
    check(head is net.lm_head, "lm_head 找对了")
    check(body is net.model, "文本主体找对了（非 VLM：就是 .model）")

    # VLM 形状的替身：body 上多挂一个 language_model
    class FakeVLM(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = inner.model
            self.lm_head = inner.lm_head
    v_body, v_head = m._lm_parts(SimpleNamespace(llm=FakeVLM(net)))
    check(v_body is net.model, "VLM 结构也认得（.model.language_model）")
    check(v_head is net.lm_head, "VLM 的 lm_head 找对了")

    print("\n== 2. 与 out.loss 逐位比对 ==")
    V, D = net.config.vocab_size, net.config.hidden_size
    B, T = 3, 11
    ok_all = True
    for trial, (n_lab, where) in enumerate([
            (4, "尾部连续"), (1, "只有最后一个位置"), (8, "中间一段"), (5, "尾部不连续")]):
        torch.manual_seed(100 + trial)
        emb = torch.randn(B, T, D, requires_grad=True)
        am = torch.ones(B, T, dtype=torch.long)
        lab = torch.full((B, T), -100, dtype=torch.long)
        for b in range(B):
            if where == "尾部连续":
                lab[b, T - n_lab:] = torch.randint(0, V, (n_lab,))
            elif where == "只有最后一个位置":
                lab[b, T - 1] = torch.randint(0, V, (1,))
            elif where == "中间一段":
                lab[b, 2:2 + n_lab] = torch.randint(0, V, (n_lab,))
            else:
                idx = torch.tensor([T - 1, T - 3, T - 5, 1, 4])[:n_lab]
                lab[b, idx] = torch.randint(0, V, (n_lab,))

        got = m.answer_only_loss(model, emb, am, lab)
        with torch.no_grad():
            ref = net(inputs_embeds=emb.detach(), attention_mask=am, labels=lab).loss
        d = abs(float(got.detach()) - float(ref))
        good = d < 1e-4
        ok_all &= good
        check(good, f"{where:<16} N={n_lab}  手工 {float(got.detach()):.6f}  "
                    f"HF {float(ref):.6f}  差 {d:.2e}")

    print("\n== 3. 梯度能回传，且**只**经过被选中的位置 ==")
    torch.manual_seed(7)
    emb = torch.randn(B, T, D, requires_grad=True)
    am = torch.ones(B, T, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    lab[:, T - 2:] = torch.randint(0, V, (B, 2))
    loss = m.answer_only_loss(model, emb, am, lab)
    loss.backward()
    check(emb.grad is not None and torch.isfinite(emb.grad).all(), "梯度有限")
    # 标签位置 t 用的是 hidden t-1，所以被选中的正文位置是 T-3 和 T-2。
    # ⚠️ 不能断言"未选中的位置梯度为 0"：注意力是**因果混合**的，
    #    位置 8 的 hidden 会汇聚 0..8 的信息，所以梯度必然回传到所有更早的位置。
    #    真正成立的不变量只有一个：**最后一个被选中位置之后**的 token
    #    影响不到任何被选中的输出，梯度必须恰好为 0。
    sel_pos = {T - 3, T - 2}
    hi = max(sel_pos)
    g = emb.grad.abs().sum(dim=(0, 2))
    tail_zero = all(float(g[t]) == 0.0 for t in range(hi + 1, T))
    check(tail_zero, f"最后被选中位置({hi})之后的位置梯度恰好为 0")
    check(any(float(g[t]) > 0.0 for t in range(0, hi + 1)),
          "被选中位置及其之前（经注意力）确实有梯度")

    print("\n== 4. 边界：一个标签都没有 ==")
    emb2 = torch.randn(B, T, D, requires_grad=True)
    lab_none = torch.full((B, T), -100, dtype=torch.long)
    l0 = m.answer_only_loss(model, emb2, am, lab_none)
    check(isinstance(l0, torch.Tensor) and l0.requires_grad,
          "返回的是可反传的张量（不是 Python float）")
    l0.backward()
    check(emb2.grad is not None and float(emb2.grad.abs().sum()) == 0.0,
          "全 -100 时梯度为 0 且不崩")

    print()
    if FAILS:
        print(f"===== {len(FAILS)} 项失败 =====")
        for f in FAILS:
            print("  - " + f)
        return 1
    print(f"===== 全部通过（在 {name} 上验的）=====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
