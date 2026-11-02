"""P0：D&C 注释生成（本地 Qwen 当生成器）。

两趟批处理，而不是逐条跑 —— GPU 上批量吞吐和逐条的差距是数量级的：

    第 1 趟：所有记录的**所有块**的摘要 prompt 拼成一个大 batch → 生成
    第 2 趟：每条记录的合并 prompt 拼成一个大 batch → 生成 annotation

跑法：
    python scripts/11_generate_annotations.py \
        --in data/raw/rna_literature_50.jsonl \
        --out data/processed/annotations_50.jsonl \
        --model models/Qwen3-8B --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.annotate import divide  # noqa: E402
from rnaqwen.data.prompts import CHUNK_SUMMARY, COMBINE_ANNOTATION, SYSTEM_RNA_EXPERT, fill  # noqa: E402
from rnaqwen.data.records import read_jsonl, write_jsonl  # noqa: E402
from rnaqwen.llm import LocalHFGenerator, OpenAICompatGenerator  # noqa: E402


_SCHEMES = ("deepseek", "deepseek-think", "openai")


def make_generator(args):
    # 允许直接用 llm.build_generator 的 scheme（`deepseek:deepseek-flash` 之类）。
    # 好处：key 由 rnaqwen.secrets 统一解析，思考模式也统一关掉
    # （DeepSeek V4 默认开思考，会把输出预算烧光 —— 见 llm.py 里的注释）。
    head = args.model.split(":", 1)[0] if ":" in args.model else ""
    if head in _SCHEMES and not args.base_url:
        from rnaqwen.llm import build_generator

        gen = build_generator(args.model)
        print(f"[gen ] build_generator({args.model!r}) -> {gen.name}")
        return gen
    if args.base_url:
        return OpenAICompatGenerator(base_url=args.base_url, model=args.model, name=f"openai:{args.model}")
    return LocalHFGenerator(model_path=args.model, batch_size=args.batch_size, dtype=args.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="models/Qwen3-8B", help="本地权重路径，或 --base-url 下的模型名")
    ap.add_argument("--base-url", default=None, help="给了就走 OpenAI 兼容 HTTP，不本地加载")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--trace", action="store_true", help="连中间摘要一起存（调数据质量时用）")
    args = ap.parse_args()

    records = list(read_jsonl(args.inp))
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("[fail] 输入为空")
        return 1
    print(f"[in ] {len(records)} 条记录 <- {args.inp}")

    gen = make_generator(args)
    t0 = time.time()

    # ---- 第 1 趟：所有块摘要 ----
    jobs: list[tuple[int, int, str]] = []          # (record_idx, chunk_idx, chunk)
    chunks_per_record: list[list[str]] = []
    for i, rec in enumerate(records):
        chunks = divide(rec.literature, max_chars=args.max_chars, max_chunks=args.max_chunks)
        chunks_per_record.append(chunks)
        for j, c in enumerate(chunks):
            jobs.append((i, j, c))

    print(f"[d&c] 第 1 趟：{len(jobs)} 个块摘要（{len(records)} 条记录）")
    convs = [
        [{"role": "system", "content": SYSTEM_RNA_EXPERT},
         {"role": "user", "content": fill(CHUNK_SUMMARY, chunk=c)}]
        for _, _, c in jobs
    ]
    summaries = gen.chat(convs, max_new_tokens=240, temperature=args.temperature)
    print(f"[d&c] 第 1 趟完成，用时 {time.time()-t0:.1f}s")

    per_record: list[list[str]] = [[] for _ in records]
    for (i, _, _), s in zip(jobs, summaries):
        per_record[i].append(s.strip())

    # ---- 第 2 趟：合并成 annotation ----
    print(f"[d&c] 第 2 趟：{len(records)} 条合并")
    convs = []
    for rec, sums in zip(records, per_record):
        block = "\n".join(f"{k+1}. {s}" for k, s in enumerate(sums) if s)
        convs.append(
            [{"role": "system", "content": SYSTEM_RNA_EXPERT},
             {"role": "user", "content": fill(COMBINE_ANNOTATION, definition=rec.definition or "(unknown)",
                                              organism=rec.organism or "(unknown)", summaries=block)}]
        )
    annotations = gen.chat(convs, max_new_tokens=420, temperature=args.temperature)
    print(f"[d&c] 第 2 趟完成，用时 {time.time()-t0:.1f}s")

    out = []
    for rec, sums, ann in zip(records, per_record, annotations):
        rec.annotation = ann.strip()
        d = rec.as_dict()
        d["provenance"] = {**rec.provenance, "annotation_generator": gen.name,
                           "annotation_date": time.strftime("%Y-%m-%d")}
        if args.trace:
            d["annotation_trace"] = {"chunk_summaries": sums}
        out.append(d)

    n = write_jsonl(args.out, out)
    print(f"\n[ok] 写入 {n} 条 -> {args.out}  总用时 {(time.time()-t0)/60:.1f} min")
    if out:
        print("\n[sample annotation]")
        print("  uid :", out[0]["uid"])
        print("  def :", out[0]["definition"][:100])
        print("  ann :", out[0]["annotation"][:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
