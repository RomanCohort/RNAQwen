"""先量再花：DC 的 `summary` 到底能不能支撑 6 对 QA？还是只有 AIS 的 `context` 有空间？

背景
----
现在每变体 5.85 QA/RNA，贴着论文声明区间（5–14）的下沿。想往上走，但两个源文本长度差很多：

    summary (DC)   中位 1,145 字符 ≈ 276 token   ← 怀疑 3 对已经榨干
    context (AIS)  中位 2,093 字符 ≈ 682 token，p90 5,929（max 12,157）

这个脚本对**同一批记录、同一段源文本**一次生成 6 对，然后量"后 3 对相对于前 3 对有多少新东西"。
判据用**问题意图的去重率**：新问题如果只是在换措辞问同一件事，那就是冗余，不值得花钱。

跑法：
    python scripts/_density_probe.py --n 12 --model deepseek:deepseek-flash
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.prompts import QA_CLOSED, QA_FROM_TEXT, SYSTEM_RNA_EXPERT, fill  # noqa: E402
from rnaqwen.eval.rouge import rouge_l  # noqa: E402
from rnaqwen.llm import build_generator, extract_json  # noqa: E402

STOP = set("what which how does do is are the a an of in to for and or that this it its with by on "
           "was were can could would should may might be been being has have had role".split())


def content_words(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9\-]+", q.lower()) if w not in STOP and len(w) > 2}


def novelty(new_qs: list[str], old_qs: list[str]) -> tuple[float, float]:
    """返回 (平均 ROUGE-L 相似度, 平均新词占比)。相似度低 = 新问题确实在问别的事。"""
    rls, news = [], []
    for nq in new_qs:
        rls.append(max((rouge_l(nq, oq)[2] for oq in old_qs), default=0.0))
        cw, ow = content_words(nq), set().union(*[content_words(o) for o in old_qs]) if old_qs else set()
        news.append(len(cw - ow) / max(len(cw), 1))
    return (sum(rls) / len(rls) if rls else 0.0), (sum(news) / len(news) if news else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/raw/rnacentral_litsumm_v2.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--model", default="deepseek:deepseek-flash")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/density_probe.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 分层抽：一半长（lncRNA）、一半短（miRNA），因为两者源文本长度差很多
    longs = [r for r in rows if len(r["sequence"]) > 500][: args.n // 2]
    shorts = [r for r in rows if len(r["sequence"]) <= 30][: args.n // 2]
    picks = longs + shorts
    print(f"[probe] 取 {len(picks)} 条（长 {len(longs)} / 短 {len(shorts)}）；每条源生成 {args.pairs} 对")

    gen = build_generator(args.model, concurrency=args.concurrency)
    jobs, meta = [], []
    for r in picks:
        for variant, src in (("dc", r.get("annotation") or ""), ("ais", r.get("literature") or "")):
            src = src.strip()
            if not src:
                continue
            for kind, tpl in (("open", QA_FROM_TEXT), ("closed", QA_CLOSED)):
                jobs.append([{"role": "system", "content": SYSTEM_RNA_EXPERT},
                             {"role": "user", "content": fill(tpl, text=src, n=args.pairs)}])
                meta.append({"uid": r["uid"], "variant": variant, "kind": kind,
                             "src_chars": len(src), "seq_nt": len(r["sequence"])})

    t0 = time.time()
    outs = gen.chat(jobs, max_new_tokens=2000, temperature=0.6)
    print(f"[probe] {len(jobs)} 次调用完成，用时 {time.time()-t0:.1f}s")

    agg: dict[tuple[str, str], list] = {}
    for m, raw in zip(meta, outs):
        try:
            obj = extract_json(raw)
        except ValueError:
            continue
        items = obj if isinstance(obj, list) else (obj.get("qa") or obj.get("pairs") or [])
        qs = [str(x.get("question", "")).strip() for x in items if isinstance(x, dict) and x.get("question")]
        if len(qs) < 4:
            continue
        half = len(qs) // 2
        rl, nw = novelty(qs[half:], qs[:half])
        agg.setdefault((m["variant"], m["kind"]), []).append((rl, nw, len(qs), m["src_chars"]))

    print()
    print(f"{'变体/类型':14s} {'样本':>5s} {'平均产出对数':>12s} {'源文本字符':>10s} "
          f"{'后3对 vs 前3对 ROUGE-L':>22s} {'新词占比':>9s}")
    print("-" * 84)
    report = {"n_records": len(picks), "pairs_requested": args.pairs, "by_group": {}}
    for key in sorted(agg):
        vals = agg[key]
        rl = sum(v[0] for v in vals) / len(vals)
        nw = sum(v[1] for v in vals) / len(vals)
        nq = sum(v[2] for v in vals) / len(vals)
        sc = sum(v[3] for v in vals) / len(vals)
        report["by_group"][f"{key[0]}/{key[1]}"] = {
            "n": len(vals), "avg_pairs": round(nq, 1), "src_chars": round(sc),
            "rouge_l_new_vs_old": round(rl, 4), "new_content_words": round(nw, 4)}
        print(f"{key[0]+'/'+key[1]:14s} {len(vals):5d} {nq:12.1f} {sc:10.0f} {rl:22.4f} {nw:9.3f}")

    print()
    print("怎么读：")
    print("  ROUGE-L 低 + 新词占比高 → 后 3 对在问**别的事**，加密度有真信息")
    print("  ROUGE-L 高 + 新词占比低 → 后 3 对在换措辞问同一件事，加密度是花冤枉钱")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[probe] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
