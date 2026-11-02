"""诊断 `13_guard_check.py` 报出来的 "序列泄漏" 是**真泄漏**还是**闸门自身的假阳性**。

怀疑点
------
`guard.normalize_sequence()` 的实现是 `[^ACGUacgu] -> ""`，也就是**把文本里所有
非 ACGU 的字符全部删掉**再去做 12-mer 匹配。问题是：英文散文里 a/c/g/t/u 这几个
字母到处都是（"c**a**t", "**t**h**a**t", "**ac**t", "**g**u**t"），删掉其它字母后
英文句子会被拼成一长串"看起来像碱基"的字符，于是**随机撞上** reference 的某个 12-mer。

判据（不靠猜）
-------------
对每个被标红的 QA：
  1. 把 question+answer 按 `normalize_sequence` 归一化，定位命中的 12-mer 的**下标**；
  2. 映射回**原文**的下标区间，看这段区间在原文里是不是**连续 12 个 ACGU 字母**；
  3. 统计该区间内被删掉了多少个非 ACGU 字符（含空格、标点、辅音字母）。

如果区间里塞满了空格/辅音，说明命中的"碱基串"横跨了普通英文单词 → 假阳性。
再加一条独立证据：`13_guard_check` 自己的 `[ACGTU]{8,}` 疑似碱基串计数。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.guard import find_sequence_leak, normalize_sequence  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402

_BASES_RE = re.compile(r"[ACGTUacgtu]{8,}")


def locate(raw: str, kmer_norm: str) -> dict:
    """在归一化串里找到 kmer 的位置，再映射回原文，报告被删掉了什么。"""
    norm_chars: list[str] = []
    idx_map: list[int] = []          # norm 下标 -> 原文下标
    for i, ch in enumerate(raw):
        u = ch.upper()
        if u in "ACGU":
            norm_chars.append(u)
            idx_map.append(i)
    norm = "".join(norm_chars)
    pos = norm.find(kmer_norm)
    if pos < 0:
        return {"found": False}
    a, b = idx_map[pos], idx_map[pos + len(kmer_norm) - 1] + 1
    span = raw[a:b]
    dropped = [c for c in span if c.upper() not in "ACGU"]
    return {
        "found": True,
        "raw_span": span,
        "raw_len": len(span),
        "dropped_chars": "".join(dropped)[:40],
        "n_dropped": len(dropped),
        # 真泄漏的必要条件：原文里就是一段连续的碱基串
        "is_contiguous_bases": len(span) == len(kmer_norm),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--min-match", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "guard_leak_forensics.json")
    args = ap.parse_args()

    recs = list(read_jsonl(args.inp))
    hits = []
    n_qa = 0
    n_contig_in_text = 0
    for r in recs:
        for p in r.qa:
            n_qa += 1
            q, a = p.get("question", ""), p.get("answer", "")
            raw = f"{q} {a}"
            exact, ngrams = find_sequence_leak(raw, r.sequence,
                                               min_match=args.min_match, kmer=args.min_match)
            if not exact and not ngrams:
                continue
            # 独立的"文本里真有连续碱基串"判据
            has_contig = bool(_BASES_RE.search(raw))
            n_contig_in_text += int(has_contig)
            detail = [locate(raw, km) for km in exact[:3]]
            hits.append({
                "uid": r.uid, "variant": p.get("variant"), "kind": p.get("kind"),
                "exact": exact[:6], "ngrams": ngrams[:3],
                "text_has_base_run_ge8": has_contig,
                "locations": detail,
                "q": q[:150], "a": a[:200],
            })

    n_true = sum(1 for h in hits if h["text_has_base_run_ge8"])
    summary = {
        "records": len(recs), "qa_total": n_qa,
        "guard_hits": len(hits),
        "hits_with_contiguous_base_run_ge8": n_true,
        "hits_that_are_prose_artifacts": len(hits) - n_true,
        "all_locations_contiguous": all(
            loc.get("is_contiguous_bases") for h in hits for loc in h["locations"]
        ) if hits else True,
    }
    print(json.dumps({k: v for k, v in summary.items()}, indent=2, ensure_ascii=False))
    print()
    for h in hits:
        print("=" * 96)
        print(f"{h['uid']} [{h['variant']}/{h['kind']}]  文本内有连续碱基串={h['text_has_base_run_ge8']}")
        print(f"  命中 12-mer: {h['exact']}")
        for loc in h["locations"]:
            if loc.get("found"):
                print(f"  原文区间(长{loc['raw_len']}) : {loc['raw_span']!r}")
                print(f"  被删掉的非碱基字符({loc['n_dropped']}): {loc['dropped_chars']!r}")
                print(f"  → 原文里是连续碱基串？ {loc['is_contiguous_bases']}")
        print(f"  A: {h['a'][:170]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "hits": hits},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[forensics] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
