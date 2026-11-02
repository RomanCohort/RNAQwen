"""P0：指令数据生成（AIS + D&C 两个变体）+ 序列泄漏闸门。

    AIS  —— 用 abstract 直接生成 QA
    D&C  —— 用 11 号脚本产出的 annotation 生成 QA

批次里所有 prompt 一次发下去，生成完逐条过 `guard.check_record_qa`：
QA 里出现碱基串就整条丢（PLAN §3 P0 完成判据 / §5 风险登记）。

跑法：
    python scripts/12_generate_qa.py \
        --in data/processed/annotations_50.jsonl \
        --out data/processed/rnaqa_50.jsonl \
        --model models/Qwen3-8B --variants ais,dc --n-pairs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.guard import check_record_qa  # noqa: E402
from rnaqwen.data.prompts import QA_CLOSED, QA_FROM_TEXT, SYSTEM_RNA_EXPERT, fill  # noqa: E402
from rnaqwen.data.qa import _normalize_pairs  # noqa: E402
from rnaqwen.data.records import read_jsonl, write_jsonl  # noqa: E402
from rnaqwen.llm import build_generator, extract_json, strip_thinking  # noqa: E402

# 原版（arXiv 2411.08900v1 §2.3）要的是**两种**：
#   "we prompt GPT-4o-mini to generate **both open-ended and close-ended QA pairs**
#    ... to decompose the abstract into **atom-level QA pairs**"
# 之前只造了 open-ended，等于漏了一半。
PROMPTS: dict[str, str] = {"open": QA_FROM_TEXT, "closed": QA_CLOSED}


def make_generator(args):
    """`--model` 三种写法：本地权重路径 / `deepseek:deepseek-chat` / OpenAI 兼容服务。"""
    if args.base_url:
        return build_generator(f"{args.base_url}:{args.model}")
    if args.model.startswith(("deepseek:", "openai:")) or "://" in args.model:
        return build_generator(args.model, concurrency=args.concurrency, timeout=args.timeout)
    return build_generator(args.model, batch_size=args.batch_size, dtype=args.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="models/Qwen3-8B",
                    help="本地路径，或 deepseek:deepseek-chat，或 http://host:port/v1:model")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--variants", default="ais,dc", help="逗号分隔：ais / dc")
    ap.add_argument("--kinds", default="open,closed",
                    help="问答类型：open=开放式，closed=封闭式（yes/no + 四选一）。原版两种都要")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = 按 n_pairs 自动算（400×n_pairs×1.5，下限 1100）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--concurrency", type=int, default=8, help="走网络 API 时的并发数")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--min-match", type=int, default=12, help="序列泄漏检测：连续多少 nt 算泄漏")
    ap.add_argument("--keep-dropped", action="store_true", help="把被闸门拦下的也存到 sidecar")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in PROMPTS:
            print(f"[fail] 未知 kind：{k}（可选 {list(PROMPTS)}）")
            return 1
    records = list(read_jsonl(args.inp))
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("[fail] 输入为空")
        return 1
    print(f"[in ] {len(records)} 条记录 <- {args.inp}；变体={variants}；类型={kinds}；每条 {args.n_pairs} 组 QA")

    gen = make_generator(args)
    t0 = time.time()

    # ---- 组装 batch ----
    jobs: list[tuple[int, str, str, str]] = []   # (record_idx, variant, kind, text)
    for i, rec in enumerate(records):
        for kind in kinds:
            if "ais" in variants and rec.literature:
                jobs.append((i, "ais", kind, rec.literature))
            if "dc" in variants and rec.annotation:
                jobs.append((i, "dc", kind, rec.annotation))
    print(f"[qa ] 共 {len(jobs)} 个生成任务（{len(records)} 条 × {len(variants)} 变体 × {len(kinds)} 类型）")

    convs = [
        [{"role": "system", "content": SYSTEM_RNA_EXPERT},
         {"role": "user", "content": fill(PROMPTS[kind], text=text, n=args.n_pairs)}]
        for _, _, kind, text in jobs
    ]
    # ⚠️ 输出预算必须随 `--n-pairs` 涨。写死 1100 的话，要 6 对时会被截断 ——
    # 实测 3 对时就已经偶发顶到上限（`finish_length` 计数）。
    # 每对按 ~400 token 估（含 JSON 外壳），留 1.5 倍余量。
    max_new = args.max_new_tokens or max(1100, int(400 * args.n_pairs * 1.5))
    print(f"[qa ] max_new_tokens={max_new}（{len(jobs)} 个任务 × {args.n_pairs} 对/类）")
    outs = gen.chat(convs, max_new_tokens=max_new, temperature=args.temperature)
    print(f"[qa ] 生成完成，用时 {time.time()-t0:.1f}s")

    # ---- 解析 + 闸门 ----
    per_record: dict[int, list[dict]] = {i: [] for i in range(len(records))}
    dropped: list[dict] = []
    errors = 0
    for (i, variant, kind, _text), raw in zip(jobs, outs):
        rec = records[i]
        try:
            obj = extract_json(raw)
        except ValueError as exc:
            errors += 1
            dropped.append({"uid": rec.uid, "variant": variant, "reason": "unparsable_json",
                            "raw": strip_thinking(raw)[:400], "error": str(exc)})
            continue
        pairs = []
        for p in _normalize_pairs(obj):
            # close-ended 的答案很短（<25 词），长度闸门要放宽，否则全被当"太短"扔掉
            min_ans = 8 if kind == "closed" else 20
            p.update({"source": "abstract" if variant == "ais" else "annotation",
                      "generator": gen.name, "variant": variant,
                      "kind": p.get("kind") or ("open_ended" if kind == "open" else "closed")})
            if len(p["answer"]) >= min_ans and len(p["question"]) >= 8:
                pairs.append(p)
            else:
                dropped.append({"uid": rec.uid, "variant": variant, "kind": kind,
                                "reason": "too_short", **p})

        rep = check_record_qa(rec.uid, pairs, rec.sequence, min_match=args.min_match, kmer=args.min_match)
        if not rep.ok:
            bad = set(rep.exact_hits) | set(rep.ngram_hits)
            kept = []
            for p in pairs:
                blob = (p["question"] + " " + p["answer"]).upper().replace("T", "U")
                hit = [h for h in bad if h in blob]
                if hit:
                    dropped.append({"uid": rec.uid, "variant": variant, "reason": "sequence_leak",
                                    "hits": hit[:3], **p})
                else:
                    kept.append(p)
            pairs = kept
        per_record[i].extend(pairs)

    out = []
    total_qa = 0
    for i, rec in enumerate(records):
        d = rec.as_dict()
        d["qa"] = per_record[i]
        d["provenance"] = {**rec.provenance, "qa_generator": gen.name,
                           "qa_date": time.strftime("%Y-%m-%d"), "variants": variants}
        total_qa += len(d["qa"])
        out.append(d)

    n = write_jsonl(args.out, out)
    print(f"\n[ok] 写入 {n} 条记录 / {total_qa} 组 QA -> {args.out}")
    print(f"[gate] 拦下 {len(dropped)} 组（unparsable_json={errors}）")

    if args.keep_dropped and dropped:
        side = args.out.with_suffix(".dropped.jsonl")
        with side.open("w", encoding="utf-8") as fh:
            for d in dropped:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[gate] 明细 -> {side}")

    if out and out[0]["qa"]:
        print("\n[sample qa]")
        for p in out[0]["qa"][:3]:
            print(f"  Q[{p['variant']}] {p['question'][:110]}")
            print(f"  A      {p['answer'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
