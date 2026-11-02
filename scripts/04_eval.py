"""P3 评测：对一批预测跑完整指标（ROUGE-1/2/L + 三种嵌入的 Semantic P/R/F1）。

输入是 JSONL，每行至少要有 `prediction` 和 `reference`（或 `gold`）。
其它字段（uid / variant / …）原样透传到报告里。

跑法：
    python scripts/04_eval.py --in data/processed/preds_50.jsonl --embedders bert,pubmedbert \
        --report reports/eval_50.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.eval.rouge import rouge_scores  # noqa: E402
from rnaqwen.eval.semantic import build_embedder, semantic_scores  # noqa: E402

PRED_KEYS = ("prediction", "pred", "hypothesis", "output", "response")
REF_KEYS = ("reference", "ref", "gold", "target", "annotation", "answer")


def pick(d: dict, keys) -> str | None:
    for k in keys:
        if k in d and isinstance(d[k], str):
            return d[k]
    return None


def load_pairs(path: Path) -> tuple[list[dict], list[str], list[str]]:
    rows, hyps, refs = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        h, r = pick(d, PRED_KEYS), pick(d, REF_KEYS)
        if h is None or r is None:
            raise ValueError(f"缺 prediction/reference 字段：{sorted(d)[:10]}")
        rows.append(d)
        hyps.append(h)
        refs.append(r)
    return rows, hyps, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--embedders", default="bert", help="逗号分隔：bert,pubmedbert,gpt 或 HF repo id")
    ap.add_argument("--group-by", default="variant", help="按哪个字段分组报告（没有就整体报）")
    ap.add_argument("--beta", type=float, default=1.0, help="ROUGE-L 的 beta（1.0=rouge_score 口径，1.2=ROUGE-1.5.5）")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "eval.json")
    args = ap.parse_args()

    rows, hyps, refs = load_pairs(args.inp)
    print(f"[eval] {len(hyps)} 条样本 <- {args.inp}")

    report: dict = {
        "input": str(args.inp),
        "n": len(hyps),
        "rouge_beta": args.beta,
        "overall": {},
        "by_group": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    r = rouge_scores(hyps, refs, beta=args.beta)
    report["overall"]["rouge"] = r.as_dict()
    print(f"[rouge] R-1={r.rouge1:.4f}  R-2={r.rouge2:.4f}  R-L={r.rougeL:.4f}")

    specs = [s.strip() for s in args.embedders.split(",") if s.strip()]
    for spec in specs:
        t0 = time.time()
        emb = build_embedder(spec)
        s = semantic_scores(hyps, refs, emb)
        report["overall"][f"semantic_{spec}"] = s.as_dict()
        print(f"[sem  ] {spec:12s} P={s.precision:.4f} R={s.recall:.4f} F1={s.f1:.4f}  "
              f"({time.time()-t0:.1f}s, {s.n_sent_hyp} vs {s.n_sent_ref} 句)")

    # 分组（例如 AIS / D&C 两个变体要分开报）
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(rows):
        groups[str(d.get(args.group_by, "all"))].append(i)
    if len(groups) > 1:
        print()
        for gname, idxs in sorted(groups.items()):
            gr = rouge_scores([hyps[i] for i in idxs], [refs[i] for i in idxs], beta=args.beta)
            entry = {"n": len(idxs), "rouge": gr.as_dict()}
            print(f"[group] {gname:8s} n={len(idxs):4d}  R-1={gr.rouge1:.4f}  R-2={gr.rouge2:.4f}  R-L={gr.rougeL:.4f}")
            for spec in specs:
                emb = build_embedder(spec)
                s = semantic_scores([hyps[i] for i in idxs], [refs[i] for i in idxs], emb)
                entry[f"semantic_{spec}"] = s.as_dict()
                print(f"          {spec:12s} F1={s.f1:.4f}  (P={s.precision:.4f} R={s.recall:.4f})")
            report["by_group"][gname] = entry

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["disclaimer"] = (
        "评测集为重建，绝对分与原论文报数不可直接比较；"
        "原版代码/权重/评测集均未开源。报告数字时必须带这句。"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
