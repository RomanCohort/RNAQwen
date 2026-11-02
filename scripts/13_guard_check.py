"""P0 验收闸门：扫一遍 QA 数据，报告有没有序列泄漏 / 格式问题。

这是**独立于生成流程**的第二道检查 —— 生成时已经拦过一次，但
（a）拦漏了、（b）后续手工改过数据，都需要这一步兜底。

跑法：
    python scripts/13_guard_check.py --in data/processed/rnaqa_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.guard import find_sequence_leak  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402

_BASES_RE = re.compile(r"[ACGTUacgtu]{8,}")
# ⚠️ 必须带词边界。老写法 `as an ai` 会命中 "acts **as an AI**MP3 suppressor" ——
# 实测 29,370 组 QA 报了 1 条"拒答"，逐条查是假阳性。两道闸门都是我自己正则的锅。
_REFUSAL_RE = re.compile(r"\b(as an ai\b|i cannot\b|i'm sorry\b|i am unable\b)|无法回答|作为人工智能", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--min-match", type=int, default=12)
    ap.add_argument("--show", type=int, default=5, help="最多展示几个坏例子")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    records = list(read_jsonl(args.inp))
    stats = Counter()
    leaks: list[dict] = []
    problems: list[dict] = []
    q_lens: list[int] = []
    a_lens: list[int] = []
    variants = Counter()

    for rec in records:
        stats["records"] += 1
        if not rec.qa:
            stats["records_without_qa"] += 1
        for p in rec.qa:
            stats["qa_total"] += 1
            q, a = p.get("question", ""), p.get("answer", "")
            variants[p.get("variant", "?")] += 1
            q_lens.append(len(q))
            a_lens.append(len(a))

            exact, ngrams = find_sequence_leak(f"{q} {a}", rec.sequence, min_match=args.min_match,
                                               kmer=args.min_match)
            if exact or ngrams:
                stats["leak"] += 1
                if len(leaks) < 200:
                    leaks.append({"uid": rec.uid, "variant": p.get("variant"),
                                  "exact": exact[:2], "ngrams": ngrams[:2],
                                  "q": q[:120], "a": a[:160]})
            # 其它脏数据
            if _BASES_RE.search(q) or _BASES_RE.search(a):
                stats["suspicious_bases"] += 1
            if _REFUSAL_RE.search(a):
                stats["refusal"] += 1
                problems.append({"uid": rec.uid, "kind": "refusal", "a": a[:120]})
            if not q.strip() or not a.strip():
                stats["empty_field"] += 1

    def pct(x: float) -> str:
        return f"{100*x:.1f}%"

    n = max(stats["qa_total"], 1)
    print(f"文件: {args.inp}")
    print(f"记录: {stats['records']}（其中 {stats['records_without_qa']} 条没有 QA）")
    print(f"QA  : {stats['qa_total']}  变体分布 {dict(variants)}")
    if q_lens:
        q_lens.sort(); a_lens.sort()
        print(f"问题长度: min={q_lens[0]} 中位={q_lens[len(q_lens)//2]} max={q_lens[-1]}")
        print(f"答案长度: min={a_lens[0]} 中位={a_lens[len(a_lens)//2]} max={a_lens[-1]}")
    print()
    print(f"[闸门] 序列泄漏        : {stats['leak']} ({pct(stats['leak']/n)})")
    print(f"[闸门] 疑似碱基串      : {stats['suspicious_bases']} ({pct(stats['suspicious_bases']/n)})")
    print(f"[闸门] 拒答/免责声明   : {stats['refusal']}")
    print(f"[闸门] 空字段          : {stats['empty_field']}")

    for item in leaks[: args.show]:
        print(f"  leak  {item['uid']} [{item['variant']}] exact={item['exact']} ngram={item['ngrams']}")
        print(f"        Q {item['q']}")
        print(f"        A {item['a']}")

    verdict = "PASS" if stats["leak"] == 0 and stats["refusal"] == 0 and stats["empty_field"] == 0 else "FAIL"
    print(f"\n[verdict] {verdict}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"stats": dict(stats), "variants": dict(variants), "verdict": verdict,
                        "leaks": leaks, "problems": problems[:50]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[report] -> {args.report}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
