"""忠实评测：用 v3 自己的预计算码（_rnaqa_v3_codes.u16）+ v3 test split。

与 76_acc_by_source.py 的区别：76_ 现场从旧的隐藏态缓存重算码，可能与 v3
训练分布不符；这里直接用训练时喂给模型的同一份码（uid -> offset/len 切片），
口径最忠实。

三臂（同 76_）：
    A 真码           B 换一条同模板样本的码     C 零向量
指标：exact match（norm+cut 之后）+ char-F1。

用法：
    python scripts/_eval_v3_codes.py \
        --ckpt tmp/bench_ckpt/step_005550.pt \
        --data data/processed/rnaqa_v3_l234.jsonl \
        --codes data/processed/_rnaqa_v3_codes.u16 \
        --codes-idx data/processed/_rnaqa_v3_codes_index.pt \
        --per-template 5 --max-new 16 --out reports/acc_v3_faithful.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— 装 shim（本机 torch 缺分布式符号）

# 复用 50_ 的模型/分词/embed 逻辑，口径与训练一致
_spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

from rnaqwen.qwen36 import base_spec_from_ckpt_args  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c.isalnum() or c in ".%").rstrip(".")


def cut(s: str) -> str:
    return (s or "").split("\n", 1)[0]


def char_f1(pred: str, gold: str) -> float:
    a, b = Counter(norm(pred)), Counter(norm(gold))
    if not a or not b:
        return 0.0
    ov = sum((a & b).values())
    if ov == 0:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def spread(lst, n):
    if n <= 0 or len(lst) <= n:
        return lst
    st = len(lst) / n
    return [lst[min(int(i * st), len(lst) - 1)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--codes-idx", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-template", type=int, default=5)
    ap.add_argument("--templates", default="",
                    help="逗号分隔的模板 key 白名单；空=全部。l2 三模板才是码信道可答的题，"
                         "l3/l4（MFE、配对比例、整段描述）靠码不可能答对，混进来只会全 0%")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--long-max-new", type=int, default=48)
    ap.add_argument("--out", default="reports/acc_v3_faithful.json")
    args = ap.parse_args()

    ck_path = ROOT / args.ckpt
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    print(f"[ckpt] {ck_path.name}  step {ck.get('step')}  "
          f"{ck.get('t_elapsed', 0)/3600:.2f} h", flush=True)

    dc = torch.load(ROOT / args.codes_idx, map_location="cpu", weights_only=False)
    n_pos = int(dc["n_pos"])
    mm = np.memmap(ROOT / args.codes, dtype=np.uint16, mode="r", shape=(n_pos,))
    CODES = torch.from_numpy(mm)
    idx = {u: (int(dc["offsets"][i]), int(dc["lens"][i])) for i, u in enumerate(dc["uids"])}
    LAT, LEV = int(dc["latent"]), int(dc["levels"])
    print(f"[codes] n_pos={n_pos} vocab={dc['vocab']} uids={len(idx)} "
          f"latent={LAT} levels={LEV}", flush=True)

    base_dir, mk_kw = base_spec_from_ckpt_args(cargs)
    LORA_R = int(cargs.get("lora_r", 8))
    from transformers import AutoTokenizer
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tk = rc.Tok(tk0)

    print("[model] 载入 35B + adapter + 码表 …", flush=True)
    t0 = time.time()
    model = rc.E2EModel(LORA_R, cargs.get("rna_emb", "table"),
                        latent=LAT, levels=LEV, **mk_kw)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(ck["lora"].keys())
    if want - got:
        print(f"** {len(want - got)} 个可训参数不在 ckpt 里：{sorted(want-got)[:4]}")
        return 3
    model.eval()
    print(f"[model] 完成 {time.time()-t0:.0f}s  可训 {len(want)}  设备 {DEV}", flush=True)

    test = []
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != args.split or r.get("uid") not in idx:
                continue
            test.append(r)
    by_tpl = defaultdict(list)
    for r in test:
        by_tpl[r.get("template") or r["question"]].append(r)
    if args.templates:
        want = [t.strip() for t in args.templates.split(",") if t.strip()]
        keep = [k for k in by_tpl if any(w == k or w in k for w in want)]
        print(f"[tpl] 白名单 {want} -> 命中 {keep}")
        by_tpl = {k: by_tpl[k] for k in keep}
        if not by_tpl:
            print("** 白名单一个模板都没命中")
            return 5
    picks = []
    for tpl in sorted(by_tpl):
        picks += spread(by_tpl[tpl], args.per_template)
    print(f"[data] split={args.split} test={len(test)} templates={len(by_tpl)} "
          f"sampled={len(picks)}", flush=True)

    def vecs(uid):
        off, ln = idx[uid]
        code = CODES[off:off + ln].to(DEV, dtype=torch.long)
        v, _, _, _ = model.rna_vectors_from_codes(code)
        return v

    def gen(rec, alt_rec, arm):
        v = vecs(rec["uid"] if arm != "B" else alt_rec["uid"])
        if arm == "C":
            v = torch.zeros_like(v)
        L = v.shape[0]
        ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
               + [tk.base_vocab] * L
               + tk.enc(f"</RNA>{rec['question']}")
               + tk.assistant_ids + tk.no_think
               # 训练序列是 `...no_think + " A: " + answer`，标签从 " A: " 之前开始。
               # 评测 prompt 必须带上 " A:" 这个前缀，否则模型看到的是"话没说完"，
               # 会直接吐 unassigned / <|im_end|> 提前终止（实测 freetext 只生成 10 字符）。
               + tk.enc(" A:"))
        ii = torch.tensor([ids], device=DEV)
        emb = model.embed(ii, v)
        mn = args.long_max_new if rec.get("kind") == "freetext" else args.max_new
        out = model.llm.generate(inputs_embeds=emb,
                                 attention_mask=torch.ones_like(ii),
                                 max_new_tokens=mn, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = tk0.decode(out[0], skip_special_tokens=True).strip()
        if "A:" in txt:
            txt = txt.split("A:")[-1].strip()
        return txt

    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(picks), generator=g).tolist()
    res = {a: [] for a in "ABC"}
    t0 = time.time()
    for i, rec in enumerate(picks):
        for a in "ABC":
            try:
                pred = gen(rec, picks[perm[i]], a)
            except Exception:
                import traceback
                traceback.print_exc()
                return 4
            res[a].append([norm(cut(pred)), norm(rec["answer"]),
                           char_f1(cut(pred), rec["answer"]), pred,
                           rec["question"], rec.get("template"), rec.get("kind")])
        line = "  ".join(
            f"{a} {100*sum(1 for x in res[a] if x[0] == x[1])/len(res[a]):.1f}%"
            for a in "ABC")
        print(f"  {i+1}/{len(picks)}  {line}  {time.time()-t0:.0f}s", flush=True)

    raw_p = ROOT / (args.out + ".raw.json")
    raw_p.parent.mkdir(parents=True, exist_ok=True)
    raw_p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[raw] {raw_p.name}", flush=True)

    # ---- 汇总：按模板 ----
    groups = defaultdict(lambda: defaultdict(list))
    for a in "ABC":
        for row in res[a]:
            groups[row[5]][a].append(row)
    print("\n" + "=" * 92)
    print("按模板拆分（exact match / 平均 char-F1）")
    print("=" * 92)
    print(f"  {'template':<44}{'n':>4}{'A':>7}{'B':>7}{'C':>7}{'A-C(pp)':>9}"
          f"{'F1_A':>7}{'F1_C':>7}")
    summary = {}
    for tpl in sorted(groups):
        n = len(groups[tpl]["A"])
        if n == 0:
            continue
        acc = {a: sum(1 for r in groups[tpl][a] if r[0] == r[1]) / n for a in "ABC"}
        f1 = {a: sum(r[2] for r in groups[tpl][a]) / n for a in "ABC"}
        print(f"  {str(tpl)[:44]:<44}{n:>4}{100*acc['A']:>6.1f}%{100*acc['B']:>6.1f}%"
              f"{100*acc['C']:>6.1f}%{100*(acc['A']-acc['C']):>+8.1f}"
              f"{f1['A']:>7.3f}{f1['C']:>7.3f}")
        summary[str(tpl)] = dict(n=n, acc=acc, f1=f1)
    # 总
    n = len(picks)
    acc = {a: sum(1 for r in res[a] if r[0] == r[1]) / n for a in "ABC"}
    f1 = {a: sum(r[2] for r in res[a]) / n for a in "ABC"}
    print("-" * 92)
    print(f"  {'ALL':<44}{n:>4}{100*acc['A']:>6.1f}%{100*acc['B']:>6.1f}%"
          f"{100*acc['C']:>6.1f}%{100*(acc['A']-acc['C']):>+8.1f}"
          f"{f1['A']:>7.3f}{f1['C']:>7.3f}")
    summary["ALL"] = dict(n=n, acc=acc, f1=f1)
    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"ckpt": str(ck_path), "step": ck.get("step"),
                                 "groups": summary}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"\n[out] {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
