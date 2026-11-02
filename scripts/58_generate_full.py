"""把 30 个模板套到全语料上 —— 4.49M 条，纯 Python，不用 API。

模板选择必须与 56_ 完全一致：这里重新跑一遍同样的筛选（同样的 4,481 条数据、
同样的判据），保证生成的模板集一模一样。写死模板列表会和 56_ 漂移。

pretrain_rna.jsonl 只有 seq / rna_type / source，所以 30 个模板里
唯一依赖类型字段的 "What type of RNA is this?" 也能用。

先按序列精确去重 —— 全语料里同一序列会有多个来源。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/processed/pretrain_rna.jsonl"
ANNO = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
OUT = ROOT / "data/processed/rnaqa_seq30_full.jsonl"

spec = importlib.util.spec_from_file_location("g56", ROOT / "scripts/56_generate_seq30.py")
g56 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g56)
CAND = g56.CAND
screen = g56.screen
S = g56.S


def pick30():
    """在 4,481 条带注释数据上重跑一遍筛选，取出和 56_ 相同的 30 个模板。"""
    recs = []
    with open(ANNO, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("sequence"):
                recs.append(r)
    cut = int(len(recs) * 0.8)
    tr, te = recs[:cut], recs[cut:]
    rows = []
    for q, fn, ml in CAND:
        s = screen(recs, tr, te, ml, fn)
        if s is None:
            continue
        if s["hit"] <= 0.20 and s["norm"] > 0.5:
            rows.append((q, fn, ml, s))
    rows.sort(key=lambda x: x[3]["hit"])
    return rows[:30]


def main() -> int:
    keep = pick30()
    print(f"选出模板 {len(keep)} 个（与 56_ 同判据）")

    # ---- 读全语料 + 精确去重 ----
    seen = {}
    with open(FULL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = (r.get("seq") or "").upper()
            if not s:
                continue
            if s in seen:
                continue
            seen[s] = r.get("rna_type") or ""
    print(f"全语料去重后 {len(seen):,} 条序列")

    # ---- 排除已带注释的那 4,481 条（避免与 rnaqa_seq30_split 重复）----
    anno_seq = set()
    with open(ANNO, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("sequence"):
                anno_seq.add(r["sequence"].upper())
    todo = [(s, t) for s, t in seen.items() if s not in anno_seq]
    print(f"排除已带注释的 {len(seen)-len(todo):,} 条 -> 待生成 {len(todo):,} 条")

    n = 0
    per_tpl = Counter()
    with open(OUT, "w", encoding="utf-8") as f:
        for seq, rna_type in todo:
            uid = "seq:" + hashlib.md5(seq.encode()).hexdigest()[:16]
            fake = {"sequence": seq, "rna_type": rna_type}
            for q, fn, ml, _s in keep:
                if len(seq) < ml:
                    continue
                try:
                    a = str(fn(fake))
                except Exception:
                    continue
                if not a:
                    continue
                f.write(json.dumps({
                    "uid": uid, "sequence": seq, "question": q, "answer": a,
                    "template": q, "source": "seq_template_full", "kind": "closed",
                }, ensure_ascii=False) + "\n")
                n += 1
                per_tpl[q] += 1
            if n and (n // 500000) != ((n - len(keep)) // 500000):
                print(f"    ... {n:,} 条", flush=True)

    print(f"\n[save] {OUT}")
    print(f"       {n:,} 条   {OUT.stat().st_size/1024/1024:.1f} MB")
    print(f"       RNA {len(todo):,} x 模板 {len(keep)} = {len(todo)*len(keep):,}"
          f"（受最小长度限制，实际 {n:,}）")
    print(f"       监督 token 估计 {n*3/1e6:.1f}M（论文 5K/121K 子集约 9.0M）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
