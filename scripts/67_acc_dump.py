"""三臂准确率的**定性版**：把输出全打出来，看模型到底在按什么答。

为什么需要它
------------
64_ 报出的数字有个说不通的地方：acc(A 真 RNA) 16.7% < acc(C 不给 RNA) 25.6%。
准确率日志里同一模板下三条**不同 RNA**给出了**完全相同**的答案
（first-6 -> 'GGGGAG'、G 计数 -> '273'、A -> '200'...）。

「同一个答案」这件事本身就是证据。但「不读 RNA」的话，不给 RNA 那一臂凭什么更高？
两种可能必须分开：
  (a) 码表向量是**负担**（把 RNA 位置换成 0 反而干净）；
  (b) 评测里有泄漏（长度 / 模板 / 解码）。

所以这一版做三件事：
  1. 三臂输出**全部落盘**，按臂分模板列表；
  2. 同一批 90 条上算「不看 RNA 只答众数」的地板（_mode_baseline.py 的口径）；
  3. 支持在**训练集**上跑同一套评测 —— 训练 ce 已经 0.2~0.9（几乎拟合了），
     那训练集准确率是多少？如果训练集高、测试集塌到地板，那是泛化问题，不是「没读 RNA」。

跑法:
    python scripts/67_acc_dump.py --artifacts runs/exp_14000 --splits "test:ABC,train:A"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
HID = ROOT / "data/processed/_fsq_full_pos.pt"
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(ch for ch in s.strip().upper() if ch.isalnum() or ch in ".%").rstrip(".")


def pick_roundrobin(pool: list[dict], n: int) -> list[dict]:
    """63_/64_ 的取法：按模板轮流取，保证每个模板都有代表。"""
    byq: dict[str, list] = {}
    for r in pool:
        byq.setdefault(r["question"], []).append(r)
    picked, i = [], 0
    while len(picked) < n and i < 10 ** 6:
        got = False
        for q, lst in byq.items():
            if i < len(lst) and len(picked) < n:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="runs/exp_14000")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--splits", default="test:ABC",
                    help="每段 'split:臂'，逗号分隔。例：test:ABC,train:A")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    art = ROOT / args.artifacts
    tag = args.tag or args.artifacts.replace("/", "_").replace("\\", "_")
    print(f"[art] {art}")
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB}  meta={meta}")

    from transformers import AutoTokenizer
    _tk0 = AutoTokenizer.from_pretrained(str(QWEN))
    NO_THINK = _tk0(" thinking\n\n<｜end▁of▁thinking｜>\n\n", add_special_tokens=False).input_ids

    def build(tk, r, clen):
        """**只到 assistant 头 + 空 think 块**。绝不能把答案段一起喂进去。

        64_ 的老写法是 ids = head + tk.enc(f" A: {r['answer']}{tk.eos}")，然后把这串
        整个喂给 generate(inputs_embeds=...)。transformers 在只给 inputs_embeds 时
        input_ids 初始化为 (B, 0)（generation/utils.py:694），返回的只有**新生成的 token**，
        所以那个评测实际问的是「金标准答案之后接着写什么」——模型会自己起一轮新问答再答一遍
        （实测：prompt 结尾 ' A: CUGGCC<|im_end|>' -> 续写 '<|im_start|>assistant\n A: GGGGAG'）。
        那不是第一次作答的准确率。见 _probe_eval_prompt.py 的逐字对照。
        """
        head = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
                + [tk.base_vocab] * clen
                + tk.enc(f"</RNA>{r['question']}")
                + tk.assistant_ids + NO_THINK)
        return head, [-100] * len(head)

    # ---- 数据 + 与训练集同口径的众数地板 ----
    # split 有三个值（train / val / test）。老版本写的是「非 train 即 test」，
    # 把 val 的 13,167 条混进了测试池 —— 取到的样本与 64_ 的口径不一致（它只用 test）。
    by_split: dict[str, list] = {}
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_split.setdefault(r.get("split", "train"), []).append(r)
    train = by_split.get("train", [])
    print("[data] 各 split：" + "  ".join(f"{k} {len(v):,}" for k, v in by_split.items()))
    cnt: dict[str, Counter] = {}
    for r in train:
        cnt.setdefault(r["question"], Counter())[norm(r["answer"])] += 1
    mode = {q: c.most_common(1)[0][0] for q, c in cnt.items()}

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {r["uid"]: int(d["offsets"][j]) for j, r in enumerate(allr)}
    len_of = {r["uid"]: int(lens[j]) for j, r in enumerate(allr)}

    # ---- 模型（adapter 必须用 set_peft_model_state_dict 灌进已有的 PeftModel）----
    tk = rc.Tok(_tk0)
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    _sd = load_file(str(art / "lora_adapter" / "adapter_model.safetensors"))
    _r = set_peft_model_state_dict(model.llm, _sd)
    _un = len(getattr(_r, "unexpected_keys", None) or [])
    print(f"[lora] 载入 {len(_sd)} 个张量   未匹配 {_un}")
    if _un > len(_sd) // 2:
        raise RuntimeError(f"adapter 大部分权重未匹配（{_un}/{len(_sd)}）—— 加载方式不对")
    with torch.no_grad():
        w = model.rna_emb.weight.float()
        print(f"[code_emb] 行范数 均值 {float(w.norm(dim=1).mean()):.3f}   "
              f"Qwen 基座 embedding 行范数 均值 "
              f"{float(model.base_emb.weight.float().norm(dim=1).mean()):.3f}")
    model.eval()

    P = 4
    g = torch.Generator().manual_seed(7)

    @torch.no_grad()
    def gen(r, i, arm, perm, picked):
        src = picked[perm[i]] if arm == "B" else r
        clen = len_of[src["uid"]] // P
        x = hid[off_of[src["uid"]]:off_of[src["uid"]] + clen * P].to(DEV, dtype=torch.float32)
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)
        if arm == "C":
            v = torch.zeros_like(v)
        ids, _ = build(tk, r, clen)
        ii = torch.tensor([ids], device=DEV)
        emb = model.embed(ii, v)
        am = torch.ones_like(ii)
        out = model.llm.generate(inputs_embeds=emb, attention_mask=am,
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = _tk0.decode(out[0], skip_special_tokens=True)
        return txt.split("A:")[-1].strip() if "A:" in txt else txt.strip()

    summary, all_rows = {}, []
    for item in args.splits.split(","):
        sp, _, arms = item.partition(":")
        arms = arms or "ABC"
        pool = by_split.get(sp)
        if not pool:
            print(f"[skip] 没有 {sp} 这个 split")
            continue
        picked = [r for r in pick_roundrobin(pool, args.n) if r["uid"] in off_of]
        floor = (sum(1 for r in picked if mode.get(r["question"]) == norm(r["answer"]))
                 / max(len(picked), 1))
        print(f"\n=== {sp} 集：{len(pool):,} 条池子，取 {len(picked)}，臂 {arms}，"
              f"众数地板 {100*floor:.1f}% ===")
        t0 = time.time()
        res, rows = {k: [] for k in arms}, []
        perm = torch.randperm(len(picked), generator=g).tolist()
        for i, r in enumerate(picked):
            for arm in arms:
                try:
                    raw = gen(r, i, arm, perm, picked)
                except Exception as e:  # noqa: BLE001
                    raw = f"<ERR {type(e).__name__}: {e}>"
                res[arm].append((norm(raw), norm(r["answer"]), r["question"], raw))
                rows.append(dict(split=sp, uid=r["uid"], arm=arm, question=r["question"],
                                 pred=norm(raw), raw=raw[:120], answer=norm(r["answer"]),
                                 clen=len_of[r["uid"]] // P))
            if (i + 1) % 10 == 0:
                a = {k: sum(1 for p, gg, _, _ in v if p == gg) / max(len(v), 1)
                     for k, v in res.items()}
                print(f"  {i+1}/{len(picked)}  " +
                      "  ".join(f"{k} {100*v:.1f}%" for k, v in sorted(a.items())) +
                      f"   {time.time()-t0:.0f}s", flush=True)
        acc = {k: sum(1 for p, gg, _, _ in v if p == gg) / max(len(v), 1)
               for k, v in res.items()}
        print(f"  -> " + "  ".join(f"acc({k}) {100*v:.1f}%" for k, v in acc.items())
              + f"   地板 {100*floor:.1f}%")
        per_tpl = {}
        print(f"\n{'模板':<50} " + "  ".join(f"{k:>7}" for k in arms) + "  不同预测数")
        for q in sorted({r["question"] for r in picked}):
            cells, nd = [], []
            for k in arms:
                sub = [(p, gg) for p, gg, qq, _ in res[k] if qq == q]
                cells.append(100 * sum(1 for p, gg in sub if p == gg) / max(len(sub), 1))
                nd.append(len({p for p, _ in sub}))
            per_tpl[q] = dict(acc=dict(zip(arms, cells)), distinct=dict(zip(arms, nd)))
            print(f"{q[:50]:<50} " + "  ".join(f"{c:>6.1f}%" for c in cells)
                  + f"   {dict(zip(arms, nd))}")
        print(f"  出现过的不重复预测数（共 {len(picked)} 条）："
              + str({k: len({p for p, _, _, _ in res[k]}) for k in arms}))
        for k, v in res.items():
            print(f"\n  {k} 臂前 6 条原始输出:")
            for p, gg, q, raw in v[:6]:
                print(f"    Q {q[:46]:<46} raw {raw[:40]!r:<42} 答案 {gg[:20]!r}")
        summary[sp] = dict(n=len(picked), acc=acc, floor=floor, arms=arms, per_template=per_tpl,
                           distinct={k: len({p for p, _, _, _ in res[k]}) for k in arms})
        all_rows += rows

    out_jsonl = ROOT / f"reports/acc_dump_{tag}.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows),
                         encoding="utf-8")
    (ROOT / f"reports/acc_dump_{tag}.json").write_text(
        json.dumps(dict(artifacts=str(art), splits=summary), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[save] {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
