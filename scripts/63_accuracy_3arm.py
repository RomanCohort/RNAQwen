"""三臂准确率评测 —— 唯一能回答"模型会不会用 RNA"的指标。

消融测的是敏感度，不是对错。这批数据的答案是精确值（"AUGCAU"、"42"），
所以对错一目了然，不需要 ROUGE 那种模糊指标。

三臂：
  A  正常 RNA
  B  换成别的样本的 RNA
  C  不给 RNA（用零向量占位）
判据：acc(A) 必须显著高于 acc(B) 和 acc(C)。三个一样 = RNA 是装饰品。

生成是贪心解码，逐条。200 条 x 3 臂足够看出差别。
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
    """答案归一化：去空格、去句号、大写。'AUGCAU' vs 'augcau' 算对。"""
    return "".join(ch for ch in s.strip().upper()
                   if ch.isalnum() or ch in ".%").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="runs/e2e_A_v2_4096/artifacts")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=12)
    args = ap.parse_args()

    art = ROOT / args.artifacts
    print(f"[art] {art}")
    for f in ("fsq.pt", "code_emb.pt", "meta.json"):
        p = art / f
        print(f"  {'OK ' if p.exists() else '缺失'} {f}")
    has_adapter = (art / "lora_adapter").exists()
    print(f"  {'OK ' if has_adapter else '缺失'} lora_adapter/")

    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    print(f"[meta] {meta}")

    # 用检查点的 latent/levels，不用硬编码
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    VOC = LEV ** LAT
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, VOC
    print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={VOC}")

    # ---- 数据 ----
    recs = []
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "test":
                continue
            recs.append(r)
    # 按模板轮流取，保证每个模板都有代表
    byq = {}
    for r in recs:
        byq.setdefault(r["question"], []).append(r)
    picked, i = [], 0
    while len(picked) < args.n and i < 10 ** 6:
        got = False
        for q, lst in byq.items():
            if i < len(lst) and len(picked) < args.n:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1
    print(f"[data] test {len(recs):,} 条，取 {len(picked)} 条（按模板均匀）")

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {r["uid"]: int(d["offsets"][i]) for i, r in enumerate(allr)}
    len_of = {r["uid"]: int(lens[i]) for i, r in enumerate(allr)}
    ok = [r for r in picked if r["uid"] in off_of]
    print(f"[data] 其中有隐藏态的 {len(ok)}")
    if len(ok) < 10:
        print("太少，退出"); return 1

    # ---- 模型 ----
    from transformers import AutoTokenizer
    tk = rc.Tok(AutoTokenizer.from_pretrained(str(QWEN)))
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    if has_adapter:
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(art / "lora_adapter"))
    model.eval()
    print("[model] 就绪")

    P = 4
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(ok), generator=g).tolist()

    @torch.no_grad()
    def gen(idx, arm: str) -> str:
        r = ok[idx]
        P_ = rc.POOL
        src = ok[perm[idx]] if arm == "swap" else r
        clen = len_of[src["uid"]] // P_
        x = hid[off_of[src["uid"]]:off_of[src["uid"]] + clen * P_].to(DEV, dtype=torch.float32)
        v, _, _, _ = model.rna_vectors(x, tau=6.0)
        if arm == "nor":                      # 不给 RNA：零向量
            v = torch.zeros_like(v)
        ids, lab, am = rc.build_batch(tk, [(r, {"question": r["question"], "answer": ""}, clen, 0)])
        emb = model.embed(ids, v)
        out = model.llm.generate(inputs_embeds=emb, attention_mask=am,
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = tk._tk.decode(out[0], skip_special_tokens=True)
        return txt.split("A:")[-1] if "A:" in txt else txt

    print("\n=== 生成 ===")
    res = {"A": [], "B": [], "C": []}
    t0 = time.time()
    for i, r in enumerate(ok):
        for arm, key in (("real", "A"), ("swap", "B"), ("nor", "C")):
            try:
                pred = gen(i, arm)
            except Exception as e:
                pred = f"<ERR {type(e).__name__}>"
            res[key].append((norm(pred), norm(r["answer"]), r["question"]))
        if (i + 1) % 20 == 0:
            accs = {k: sum(1 for p, g_, _ in v if p == g_) / max(len(v), 1)
                    for k, v in res.items()}
            print(f"  {i+1}/{len(ok)}  " +
                  "  ".join(f"acc{k} {100*a:.1f}%" for k, a in accs.items()) +
                  f"   {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    out = {}
    for k, v in res.items():
        acc = sum(1 for p, g_, _ in v if p == g_) / max(len(v), 1)
        out[k] = acc
        print(f"  acc({k}) = {100*acc:.1f}%   ({sum(1 for p,g_,_ in v if p==g_)}/{len(v)})")
    print()
    if out["A"] > out["B"] + 0.05 and out["A"] > out["C"] + 0.05:
        print(f"  ✅ RNA 在起作用：A {100*out['A']:.1f}% 显著高于 B {100*out['B']:.1f}% / C {100*out['C']:.1f}%")
    elif out["A"] > 0.10 and out["B"] < 0.05:
        print(f"  ⚠️ A 明显高于 B 但绝对准确率低：{100*out['A']:.1f}%")
    else:
        print(f"  ❌ 三臂接近 —— RNA 是装饰品")
    print()
    print("按模板看 A 臂:")
    bt = {}
    for p, g_, q in res["A"]:
        bt.setdefault(q, []).append(p == g_)
    for q, v in sorted(bt.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        print(f"  {100*sum(v)/len(v):>5.1f}%  ({len(v):>3})  {q[:56]}")
    print()
    print("A 臂前 8 条实际输出:")
    for p, g_, q in res["A"][:8]:
        print(f"  Q {q[:44]}")
        print(f"    预测 {p[:30]!r}   答案 {g_[:30]!r}   {'✓' if p==g_ else '✗'}")

    (ROOT / "reports/accuracy_3arm.json").write_text(
        json.dumps(dict(n=len(ok), acc=out, max_new=args.max_new), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
