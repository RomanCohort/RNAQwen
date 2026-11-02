"""三臂准确率 + Qwen3 关思考。

上一轮三臂全 0%，输出是 "THINK THE FIRST OF..." / "OKAY LET'S SEE..."。
Qwen3 是混合思考模型，我们的 build_batch 只写了 `<|im_start|>assistant\n`，
没有插空 think 块 —— 等于把"先思考"的先验留给了基座。

官方关思考的做法是在 assistant 头后面插：
    thinking

   <｜end▁of▁thinking｜>

这样模型直接输出答案，不需要先想。

同时把 max_new_tokens 提上去（12 个 token 连"OKAY LET'S SEE"都装不下）。

注意：训练时也没插这个块，所以评测的 prompt 与训练不完全一致。
但基座对这个空块的先验很强，先测了再说 —— 如果准了，训练那边也要补上。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
HID = ROOT / "data/processed/_fsq_full_pos.pt"
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(ch for ch in s.strip().upper()
                   if ch.isalnum() or ch in ".%").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="runs/A_v2_4096/artifacts")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--no-think", type=int, default=1, help="1=插空 think 块关思考")
    args = ap.parse_args()

    art = ROOT / args.artifacts
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB}  no_think={args.no_think}")

    # ---- 关思考的空块 ----
    from transformers import AutoTokenizer
    _tk0 = AutoTokenizer.from_pretrained(str(QWEN))
    NO_THINK = _tk0(" thinking\n\n<｜end▁of▁thinking｜>\n\n", add_special_tokens=False).input_ids
    print(f"[think] 空 think 块 token 数 {len(NO_THINK)}  {NO_THINK}")

    def build(tk, r, clen):
        a = f" A: {r['answer']}{tk.eos}"
        head = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
                + [tk.base_vocab] * clen
                + tk.enc(f"</RNA>{r['question']}")
                + tk.assistant_ids)
        if args.no_think:
            head = head + NO_THINK
        ids = head + tk.enc(a)
        n_head = len(head)
        lab = [-100] * n_head + ids[n_head:]
        return ids, lab

    # ---- 数据 ----
    recs = [json.loads(l) for l in open(ROOT / args.data, encoding="utf-8")
            if l.strip()]
    recs = [r for r in recs if r.get("split") == "test"]
    byq = {}
    for r in recs:
        byq.setdefault(r["question"], []).append(r)
    picked, i = [], 0
    while len(picked) < args.n:
        got = False
        for q, lst in byq.items():
            if i < len(lst) and len(picked) < args.n:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1
    print(f"[data] test {len(recs):,} 取 {len(picked)}")

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {r["uid"]: int(d["offsets"][j]) for j, r in enumerate(allr)}
    len_of = {r["uid"]: int(lens[j]) for j, r in enumerate(allr)}
    picked = [r for r in picked if r["uid"] in off_of]
    print(f"[data] 有隐藏态 {len(picked)}")

    # ---- 模型 ----
    tk = rc.Tok(_tk0)
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    print(f"[vocab] {tk.base_vocab}")
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    # model.llm 已经是 PeftModel（E2EModel.__init__ 里 get_peft_model 包的）。
    # 再套一层 PeftModel.from_pretrained 会形成嵌套，权重键全对不上、被当 UNEXPECTED 丢掉，
    # 评测就退化成「未微调的 Qwen + 训练好的码表」—— 准确率必然接近 0。
    # 正确做法：直接往已有的 PeftModel 里灌 adapter 的 state_dict。
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    _sd = load_file(str(art / "lora_adapter" / "adapter_model.safetensors"))
    _r = set_peft_model_state_dict(model.llm, _sd)
    _un = len(getattr(_r, "unexpected_keys", None) or [])
    _mi = len(getattr(_r, "missing_keys", None) or [])
    print(f"[lora] 载入 {len(_sd)} 个张量   未匹配 {_un}   缺失 {_mi}")
    if _un > len(_sd) // 2:
        raise RuntimeError(f"adapter 大部分权重未匹配（{_un}/{len(_sd)}）—— 加载方式不对")
    model.eval()
    print("[model] 就绪")

    P = 4
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(picked), generator=g).tolist()

    @torch.no_grad()
    def gen(idx, arm):
        r = picked[idx]
        src = picked[perm[idx]] if arm == "swap" else r
        clen = len_of[src["uid"]] // P
        x = hid[off_of[src["uid"]]:off_of[src["uid"]] + clen * P].to(DEV, dtype=torch.float32)
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)
        if arm == "nor":
            v = torch.zeros_like(v)
        ids, lab = build(tk, r, clen)
        ii = torch.tensor([ids], device=DEV)
        am = torch.ones_like(ii)
        emb = model.embed(ii, v)
        out = model.llm.generate(inputs_embeds=emb, attention_mask=am,
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = _tk0.decode(out[0], skip_special_tokens=True)
        return txt.split("A:")[-1].strip() if "A:" in txt else txt.strip()

    print("\n=== 生成 ===")
    res = {"A": [], "B": [], "C": []}
    t0 = time.time()
    for i, r in enumerate(picked):
        for arm, key in (("real", "A"), ("swap", "B"), ("nor", "C")):
            try:
                pred = gen(i, arm)
            except Exception as e:
                pred = f"<ERR {type(e).__name__}: {e}>"
            res[key].append((norm(pred), norm(r["answer"]), r["question"], pred))
        if (i + 1) % 20 == 0:
            a = {k: sum(1 for p, gg, _, _ in v if p == gg) / max(len(v), 1)
                 for k, v in res.items()}
            print(f"  {i+1}/{len(picked)}  " +
                  "  ".join(f"{k} {100*v:.1f}%" for k, v in a.items()) +
                  f"   {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 72)
    out = {}
    for k, v in res.items():
        acc = sum(1 for p, gg, _, _ in v if p == gg) / max(len(v), 1)
        out[k] = acc
        print(f"  acc({k}) = {100*acc:>5.1f}%   ({sum(1 for p,gg,_,_ in v if p==gg)}/{len(v)})")
    print()
    if out["A"] > out["B"] + 0.05 and out["A"] > out["C"] + 0.05:
        print(f"  ✅ RNA 起作用：A {100*out['A']:.1f}% > B {100*out['B']:.1f}% / C {100*out['C']:.1f}%")
    elif out["A"] > 0.05:
        print(f"  ⚠️ A {100*out['A']:.1f}% 但未显著高于对照")
    else:
        print("  ❌ 三臂都接近 0")

    bt = {}
    for p, gg, q, _ in res["A"]:
        bt.setdefault(q, []).append(p == gg)
    print("\n按模板（A 臂）:")
    for q, v in sorted(bt.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {100*sum(v)/len(v):>5.1f}%  ({len(v):>3})  {q[:52]}")
    print("\nA 臂前 10 条原始输出:")
    for p, gg, q, raw in res["A"][:10]:
        print(f"  Q {q[:42]}")
        print(f"    raw {raw[:70]!r}")
        print(f"    预测 {p[:24]!r}  答案 {gg[:24]!r}  {'✓' if p==gg else '✗'}")

    (ROOT / "reports/accuracy_3arm.json").write_text(json.dumps(
        dict(n=len(picked), acc=out, max_new=args.max_new, no_think=args.no_think),
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
