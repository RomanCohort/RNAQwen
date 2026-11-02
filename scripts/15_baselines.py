"""P3 前置：**地板基线**。不上模型也能跑，用来给后面所有数字定标尺。

为什么必须先做这一步
--------------------
`docs/METRICS.md` 记了一条实测缺陷：句子级 max-cosine 的 semantic F1 **地板极高**
（完全无关的两段文本也有 ~0.81）。也就是说，一个"什么都没说"的模型也能拿到好看的分。

所以任何模型分数都必须跟这组基线一起报：

    empty          空输出（模型什么都没说）
    generic        固定套话（"This is a non-coding RNA that regulates gene expression."）
    definition     把 NCBI 的 DEFINITION 行原样当答案（**信息泄漏型基线**，只作参照）
    shuffled       拿**另一条**记录的 abstract 当答案（= 随机配对的地板）
    identical      完美答案（上界，用来标出动态范围）

跑法：
    python scripts/15_baselines.py --data data/raw/rna_literature_50.jsonl \
        --embedder models/bert-base-uncased --out reports/baselines_50.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.eval.rouge import rouge_scores  # noqa: E402
from rnaqwen.eval.semantic import build_embedder, semantic_scores  # noqa: E402

GENERIC = (
    "This molecule is a non-coding RNA. It regulates gene expression and interacts with "
    "protein partners. Its secondary structure contains conserved motifs that are required "
    "for its function."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--field", default="literature", choices=["literature", "annotation"],
                    help="参考文本取哪个字段")
    ap.add_argument("--embedder", default=str(ROOT / "models" / "bert-base-uncased"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "baselines.json")
    args = ap.parse_args()

    rows = [r.as_dict() for r in read_jsonl(args.data)]
    rows = [r for r in rows if (r.get(args.field) or "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("[fail] 没有可用记录")
        return 1

    refs = [r[args.field] for r in rows]
    rng = random.Random(args.seed)
    shuffled = refs[:]
    rng.shuffle(shuffled)
    # 确保没人拿到自己那条
    for i in range(len(shuffled)):
        if shuffled[i] is refs[i] and len(refs) > 1:
            j = (i + 1) % len(shuffled)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    baselines = {
        "empty": ["" for _ in rows],
        "generic": [GENERIC for _ in rows],
        "definition": [r.get("definition", "") for r in rows],
        "shuffled": shuffled,
        "identical": refs,
    }

    emb = build_embedder(args.embedder)
    print(f"[baseline] {len(rows)} 条记录；参考字段={args.field}；嵌入={args.embedder}\n")
    print(f"{'baseline':12s} {'R-1':>7s} {'R-2':>7s} {'R-L':>7s} {'semP':>7s} {'semR':>7s} {'semF1':>7s}")

    report: dict = {"n": len(rows), "reference_field": args.field, "embedder": args.embedder, "baselines": {}}
    for name, hyps in baselines.items():
        r = rouge_scores(hyps, refs)
        s = semantic_scores(hyps, refs, emb)
        entry = {"rouge1": r.rouge1, "rouge2": r.rouge2, "rougeL": r.rougeL,
                 "semantic_precision": s.precision, "semantic_recall": s.recall, "semantic_f1": s.f1}
        report["baselines"][name] = entry
        print(f"{name:12s} {r.rouge1:7.4f} {r.rouge2:7.4f} {r.rougeL:7.4f} "
              f"{s.precision:7.4f} {s.recall:7.4f} {s.f1:7.4f}")

    floor = report["baselines"]["shuffled"]
    ident = report["baselines"]["identical"]
    report["dynamic_range"] = {
        "rouge1": [floor["rouge1"], ident["rouge1"]],
        "semantic_f1": [floor["semantic_f1"], ident["semantic_f1"]],
        "note": "可动区间 = 随机配对地板 到 完美答案上界。区间越窄，这条指标越没有分辨力。",
    }
    print(f"\n[baseline] ROUGE-1 可动区间     : {floor['rouge1']:.4f} → {ident['rouge1']:.4f}")
    print(f"[baseline] semantic F1 可动区间: {floor['semantic_f1']:.4f} → {ident['semantic_f1']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[baseline] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
