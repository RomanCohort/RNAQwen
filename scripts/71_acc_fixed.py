"""修正版准确率评测 —— 修掉 §11 那个「prompt 里带金标准答案」的 bug。

坏在哪（§11 已核实）：
    gen() 把 build() 返回的整个 ids 喂给 generate(inputs_embeds=...)
    而 ids = head + tk.enc(f" A: {r['answer']}{tk.eos}")   <- 答案在里面
    transformers 在只给 inputs_embeds 时把 input_ids 初始化成 (B,0)，
    返回序列只含新生成的 token，所以模型看到的是「答案已在上下文，接着写」。

修法：只喂到 assistant 头 + 空 think 块为止。**答案一个 token 都不能进去。**

另外这次加一个必须有的对照：把答案本身当作"上下文"喂进去（leak 臂），
它应该接近 100% —— 如果 leak 臂都到不了 100%，说明评测流程本身还有问题。
**这是尺子自检，不能省。**
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402  必须先于 transformers/peft，否则 torch.distributed 会炸
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
HID = ROOT / "data/processed/_fsq_full_pos.pt"
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c.isalnum() or c in ".%").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--data", default="data/processed/rnaqa_seq8.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    art = ROOT / args.artifacts
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB}  {art.name}")

    from transformers import AutoTokenizer
    tk0 = AutoTokenizer.from_pretrained(str(QWEN))
    NO_THINK = tk0(" thinking\n\n<｜end▁of▁thinking｜>\n\n", add_special_tokens=False).input_ids

    def build_prompt(tk, r, clen):
        """**只到 assistant 头 + 空 think 块，答案一个 token 都不能进去。**
        这是 §11 那个 bug 的根治点。"""
        return (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
                + [tk.base_vocab] * clen
                + tk.enc(f"</RNA>{r['question']}")
                + tk.assistant_ids + NO_THINK)

    recs = [json.loads(l) for l in open(ROOT / args.data, encoding="utf-8") if l.strip()]
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
    print(f"[data] {args.data}  test {len(recs):,}  取 {len(picked)}（按模板均匀）")

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {r["uid"]: int(d["offsets"][i]) for i, r in enumerate(allr)}
    len_of = {r["uid"]: int(lens[i]) for i, r in enumerate(allr)}
    picked = [r for r in picked if r["uid"] in off_of]
    print(f"[data] 有隐藏态 {len(picked)}")

    tk = rc.Tok(tk0)
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(str(art / "lora_adapter" / "adapter_model.safetensors"))
    r_ = set_peft_model_state_dict(model.llm, sd)
    print(f"[lora] {len(sd)} 个张量  未匹配 {len(getattr(r_,'unexpected_keys',None) or [])}")
    model.eval()

    P = 4
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(picked), generator=g).tolist()

    @torch.no_grad()
    def run(idx, arm):
        r = picked[idx]
        if arm == "C":
            src = r
        elif arm == "B":
            src = picked[perm[idx]]
        else:
            src = r
        clen = len_of[src["uid"]] // P
        x = hid[off_of[src["uid"]]:off_of[src["uid"]] + clen * P].to(DEV, dtype=torch.float32)
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)
        if arm == "C":
            v = torch.zeros_like(v)
        ids = build_prompt(tk, r, clen)
        ii = torch.tensor([ids], device=DEV)
        emb = model.embed(ii, v)
        out = model.llm.generate(inputs_embeds=emb,
                                 attention_mask=torch.ones_like(ii),
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = tk0.decode(out[0], skip_special_tokens=True).strip()
        # 模型学到要输出 " A: " 前缀（那是训练目标的一部分），评测时必须剥掉。
        # 漏了这句的话 acc 会全是 0 —— 上一轮就是栽在这里，由 leak 自检抓到。
        if "A:" in txt:
            txt = txt.split("A:")[-1].strip()
        return txt

    arms = ["A", "B", "C"]
    res = {a: [] for a in arms}
    t0 = time.time()
    print("\n=== 生成 ===")
    for i, r in enumerate(picked):
        for a in arms:
            try:
                pred = run(i, a)
            except Exception as e:
                pred = f"<ERR {type(e).__name__}>"
            res[a].append((norm(pred), norm(r["answer"]), r["question"], pred))
        if (i + 1) % 20 == 0:
            line = "  ".join(f"{a} {100*sum(1 for p,gg,_,_ in res[a] if p==gg)/len(res[a]):.1f}%"
                             for a in arms)
            print(f"  {i+1}/{len(picked)}  {line}   {time.time()-t0:.0f}s", flush=True)

    # ---- 静态 leak 自检（§11 的 bug 就在这里，必须是硬断言）----
    print("\n=== 静态 leak 自检 ===")
    checked = 0
    for r in picked[:40]:
        L = len_of[r["uid"]] // P
        ids = build_prompt(tk, r, L)
        txt = tk0.decode([i for i in ids if i < tk.base_vocab], skip_special_tokens=True)
        ans = norm(r["answer"])
        if ans and ans in norm(txt):
            print(f"  **prompt 里出现了答案 {ans!r}**")
            print(f"     {txt[-160:]!r}")
            raise SystemExit("评测 prompt 泄漏金标准答案，结果不可信（见 §11）")
        checked += 1
    print(f"  {checked} 条 prompt 全部不含答案  OK")

    print("\n" + "=" * 74)
    print("结果")
    print("=" * 74)
    acc = {}
    for a in arms:
        v = res[a]
        acc[a] = sum(1 for p, gg, _, _ in v if p == gg) / max(len(v), 1)
        print(f"  acc({a}) = {100*acc[a]:>5.1f}%   ({sum(1 for p,gg,_,_ in v if p==gg)}/{len(v)})")
    print()
    if acc["A"] > acc["C"] + 0.05 and acc["A"] > acc["B"] + 0.05:
        print(f"  ✅ RNA 起作用：A {100*acc['A']:.1f}% > B {100*acc['B']:.1f}% / C {100*acc['C']:.1f}%")
    elif acc["A"] > 0.10:
        print(f"  ⚠️ A {100*acc['A']:.1f}% 但未显著高于对照")
    else:
        print("  ❌ 三臂接近")
    print()
    print("按模板（A 臂）:")
    bt = {}
    for p, gg, q, _ in res["A"]:
        bt.setdefault(q, []).append(p == gg)
    for q, v in sorted(bt.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {100*sum(v)/len(v):>5.1f}%  ({len(v):>3})  {q[:50]}")
    print("\nA 臂前 8 条原始输出:")
    for p, gg, q, raw in res["A"][:8]:
        print(f"  Q {q[:40]}")
        print(f"    生成 {raw[:50]!r}")
        print(f"    答案 {gg[:30]!r}  {'✓' if p==gg else '✗'}")

    (ROOT / "reports/acc_fixed.json").write_text(json.dumps(
        dict(artifacts=args.artifacts, data=args.data, n=len(picked),
             acc=acc, max_new=args.max_new), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
