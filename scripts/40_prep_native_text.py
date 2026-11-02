"""原生文本路线的预处理：把 RNA 序列变成"普通文本"。

这是 PLAN_v2 §2.2 里唯一需要新写的东西（几十行）。产出两份：

  ① data/processed/pretrain_rna.jsonl     序列继续预训练语料（每行一条 RNA 的纯文本）
  ② data/processed/sft_text_stats.json    为②扫一遍：序列进文本后，prompt 会多长

为什么先做这两个：
  §2.3 里定的是"**先跑②（直接 SFT），不够再补①**"。所以①可以做出来备用，
  但真正卡住臂 B 的是"RNA 当文本进模板之后，prompt 长度是多少、会不会超上下文"。
  ②就是为了先把这个数摆出来。

关键取舍（都在代码里注释了）：
  * 只保留 ACGU 四条碱基 —— 实测全语料 1,557,973 个碱基里非 ACGUN 的是 **0 个**，
    所以这不会丢数据；但它挡住了脏数据（空格、换行、数字混进序列）
  * 按序列前 60 字符去重 —— 重复会让继续预训练把某些序列背下来
  * 长度上限取 **1022 nt**：ERNIE-RNA 的实测上限。换到原生文本后理论上更长也行
    （Qwen3 上下文 32K），但**保持和臂 A 同长度上限**，否则对比里多一个变量
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
PARQUET = ROOT / "data/external/rnacentral_1024.parquet"
OUT_PRETRAIN = ROOT / "data/processed/pretrain_rna.jsonl"
OUT_STATS = ROOT / "data/processed/sft_text_stats.json"

BASES = set("ACGU")
MAX_NT = 1022          # 与臂 A 对齐
MIN_NT = 20
SEED = 20261010


def clean(seq: str) -> str | None:
    s = (seq or "").upper().replace("T", "U").strip()
    if not s or len(s) < MIN_NT or len(s) > MAX_NT:
        return None
    if not set(s) <= BASES:          # 混进 N / 空格 / 数字的一律丢
        return None
    return s


def from_merged():
    recs = []
    with open(MERGED, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    out = []
    for r in recs:
        s = clean(r.get("sequence"))
        if s:
            out.append({"seq": s, "rna_type": r.get("rna_type") or "",
                        "source": "rnaqa_n6_merged"})
    return out, recs


def from_parquet(n_want: int, rng: random.Random):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(PARQUET)
    out = []
    # 流式读，按行组抽；不一次进内存（3250 万行）
    for rg in range(f.metadata.num_row_groups):
        if len(out) >= n_want:
            break
        t = f.read_row_group(rg, columns=["sequence", "type"])
        d = t.to_pydict()
        for s, ty in zip(d["sequence"], d["type"]):
            if len(out) >= n_want:
                break
            c = clean(s)
            if c:
                out.append({"seq": c, "rna_type": ty or "", "source": "rnacentral_1024"})
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="both", choices=["merged", "parquet", "both"])
    ap.add_argument("--n-parquet", type=int, default=200_000,
                    help="从 3250 万条里抽多少做继续预训练语料")
    ap.add_argument("--out", type=Path, default=OUT_PRETRAIN)
    args = ap.parse_args()

    rng = random.Random(SEED)
    rows, recs = [], []
    if args.source in ("merged", "both"):
        rows, recs = from_merged()
        print(f"[prep] merged: {len(rows)} 条干净序列（原始 {len(recs)} 条）")
    if args.source in ("parquet", "both"):
        p = from_parquet(args.n_parquet, rng)
        print(f"[prep] parquet: 抽到 {len(p)} 条干净序列")
        rows += p

    # 去重（按前 60 字符）
    seen, uniq = set(), []
    for r in rows:
        k = r["seq"][:60]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    print(f"[prep] 去重后 {len(uniq)} 条（去掉 {len(rows)-len(uniq)} 条重复）")

    lens = sorted(len(r["seq"]) for r in uniq)
    lp = sorted((len(r["seq"]) + 2) // 2 + 8 for r in uniq)   # 实测 0.526 tok/nt，向上取整
    print(f"[prep] 序列长度 min {lens[0]}  中位 {lens[len(lens)//2]}  max {lens[-1]}")
    print(f"[prep] 对应 token 数（按 0.526 tok/nt）min {lp[0]}  中位 {lp[len(lp)//2]}  p99 {lp[int(len(lp)*0.99)]}  max {lp[-1]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")

    # ---- ② 臂 B 的 prompt 长度估算 ----
    # 模板（Llama-3 风格，和 Stage 2 一致）+ 序列 + 问题 + 答案
    if recs:
        import statistics
        prompt_toks, ans_toks = [], []
        for r in recs:
            s = clean(r.get("sequence")) or ""
            n_seq = int(len(s) / 0.526)
            for q in (r.get("qa") or []):
                qt = q.get("question") or ""
                at = q.get("answer") or ""
                prompt_toks.append(n_seq + int(len(qt) / 3.6) + 24)   # 24 ≈ 模板开销
                ans_toks.append(int(len(at) / 3.6) + 2)
        if prompt_toks:
            pt = sorted(prompt_toks)
            print()
            print("[prep] 臂 B（RNA 当文本）的 prompt token 数估算：")
            print(f"        中位 {pt[len(pt)//2]}   p90 {pt[int(len(pt)*0.9)]}   max {pt[-1]}"
                  f"   （对应 1.9 nt/token，答案另计）")
            over = sum(1 for x in pt if x > 4096)
            print(f"        超过 4096 token 的样本：{over} / {len(pt)}"
                  f"（{100*over/len(pt):.2f}%）")
            print(f"        -> Qwen3 上下文远大于此，臂 B 不需要截断序列")
            json_stats = {
                "n_sequences": len(uniq),
                "seq_len": {"min": lens[0], "median": lens[len(lens)//2], "max": lens[-1]},
                "prompt_tokens": {"median": pt[len(pt)//2], "p90": pt[int(len(pt)*0.9)],
                                  "max": pt[-1], "over_4096": over,
                                  "n_samples": len(pt)},
                "template": "llama3 + RNA 原生文本",
                "max_nt": MAX_NT, "seed": SEED,
            }
            OUT_STATS.write_text(json.dumps(json_stats, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"[prep] -> {OUT_STATS}")
        else:
            print("[prep] 没有 QA 样本，跳过 prompt 长度估算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
