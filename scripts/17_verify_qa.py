"""E 轨改进项 #3：**验真式数据筛选**（verifier-gated）。

思路
----
生成 QA 时模型可能写出"源文本其实不支持"的答案。原版没有这一层 —— 他们把 GPT-4o-mini
生成的东西直接拿去训。RNAcentral 自己的 `litsumm` 却有（`consistency_check_result` 列），
说明这一层在 2025–2026 已经算标配。

做法：对每条 QA，把「源文本 + 问题 + 答案」再喂回模型，让它判一句：
    SUPPORTED / UNSUPPORTED
不支持的丢掉。

为什么要单独写、不塞进 `12_generate_qa.py`
------------------------------------------
1. 它是 **E 轨**，必须能单独开关，不能污染复现轨的数据；
2. 它应该能跑在**已经生成好的**数据上（不用重跑一遍生成，省钱）；
3. 它要留下**统计证据**（支持的/不支持的/判定失败的），好写进报告。

跑法：
    python scripts/17_verify_qa.py --in data/processed/rnaqa_n6.jsonl \
        --out data/processed/rnaqa_n6_verified.jsonl --limit 200 --concurrency 8
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.llm import build_generator  # noqa: E402

VERIFY_PROMPT = """Decide whether the ANSWER is fully supported by the SOURCE.

SOURCE:
\"\"\"
{source}
\"\"\"

QUESTION: {question}
ANSWER: {answer}

Rules:
- Reply with exactly one word: SUPPORTED or UNSUPPORTED.
- SUPPORTED means every factual claim in the ANSWER can be traced to the SOURCE.
- If the ANSWER adds facts not present in the SOURCE, reply UNSUPPORTED.
- The SOURCE is literature excerpts about an RNA molecule; questions may be about the RNA
  itself or about findings reported in that literature.

Reply:"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek:deepseek-flash")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条记录（0=全部）")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-source-chars", type=int, default=6000,
                    help="源文本太长就截断，控制成本")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[verify] {len(rows)} 条记录 <- {args.inp}")

    gen = build_generator(args.model, concurrency=args.concurrency)

    # 每条 QA 一个判定任务
    jobs, meta = [], []
    for ri, r in enumerate(rows):
        src_ais = (r.get("literature") or "")[: args.max_source_chars]
        src_dc = (r.get("annotation") or "")[: args.max_source_chars]
        for qi, q in enumerate(r.get("qa") or []):
            src = src_ais if q.get("variant") == "ais" else src_dc
            jobs.append([{"role": "user", "content": VERIFY_PROMPT.format(
                source=src, question=q.get("question", ""), answer=q.get("answer", ""))}])
            meta.append((ri, qi))
    print(f"[verify] {len(jobs)} 个判定任务，并发 {args.concurrency}")

    t0 = time.time()
    outs = gen.chat(jobs, max_new_tokens=16, temperature=0.0)
    print(f"[verify] 判定完成，用时 {time.time()-t0:.1f}s")

    verdicts: dict[tuple[int, int], str] = {}
    stats = collections.Counter()
    for (ri, qi), raw in zip(meta, outs):
        up = raw.strip().upper()
        if up.startswith("SUPPORTED"):
            verdicts[(ri, qi)] = "supported"
            stats["supported"] += 1
        elif up.startswith("UNSUPPORTED"):
            verdicts[(ri, qi)] = "unsupported"
            stats["unsupported"] += 1
        else:
            verdicts[(ri, qi)] = "unparsed"
            stats["unparsed"] += 1
    by_variant = collections.Counter()
    by_variant_unsup = collections.Counter()
    for (ri, qi), v in verdicts.items():
        var = (rows[ri].get("qa") or [])[qi].get("variant", "?")
        by_variant[var] += 1
        if v == "unsupported":
            by_variant_unsup[var] += 1

    n = max(sum(stats.values()), 1)
    print(f"\n[verify] 支持 {stats['supported']} ({100*stats['supported']/n:.1f}%) · "
          f"不支持 {stats['unsupported']} ({100*stats['unsupported']/n:.1f}%) · "
          f"判定失败 {stats['unparsed']} ({100*stats['unparsed']/n:.1f}%)")
    for var in sorted(by_variant):
        print(f"[verify]   {var}: 不支持率 {100*by_variant_unsup[var]/max(by_variant[var],1):.1f}% "
              f"({by_variant_unsup[var]}/{by_variant[var]})")

    # 写出：默认保留全部，给每条 QA 打上 verdict 标记（这样下游可以自己决定怎么用）
    kept_total = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for ri, r in enumerate(rows):
            new_qa = []
            for qi, q in enumerate(r.get("qa") or []):
                v = verdicts.get((ri, qi), "missing")
                q = {**q, "verifier": v, "verified_by": gen.name}
                if v != "unsupported":
                    new_qa.append(q)
            d = {**r, "qa": new_qa}
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept_total += len(new_qa)
    print(f"\n[verify] 写出 {len(rows)} 条记录 / {kept_total} 组 QA（丢掉 unsupported）")
    print(f"[verify] -> {args.out}")

    rep = {"input": str(args.inp), "records": len(rows), "qa_total": sum(stats.values()),
           "supported": stats["supported"], "unsupported": stats["unsupported"],
           "unparsed": stats["unparsed"], "model": gen.name,
           "unsupported_rate_by_variant": {v: round(by_variant_unsup[v] / max(by_variant[v], 1), 4)
                                           for v in sorted(by_variant)},
           "kept_qa": kept_total}
    rp = args.report or args.out.with_suffix(".verify_report.json")
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[verify] 报告 -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
