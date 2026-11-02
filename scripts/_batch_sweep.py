"""batch size 扫描：把我们这台 iGPU 的真实吞吐曲线量出来。

为什么要单独量
--------------
原版配方是 `batch=1`，我们现在也是 `batch=1`。但那是原版拿 2×A100 40GB 跑出来的选择 ——
我们这边实测峰值显存只用了 **29.23 GiB / 99.7 GiB**，有 3.4 倍余量。

关键问题不是"能不能开大"，是**开大到底换不换来吞吐**：
  * agent A 已经测过 `grad_accum=8` 耗时是 `accum=2` 的 8 倍 —— **梯度累积不省算力**，
    它只是把 N 次前反向攒成 1 个 optimizer step；
  * 而**真正的 batch**（一次前向里塞 B 条）能把每步的固定开销（kernel launch / 调度）
    摊薄，这才是可能提速的地方。

这个脚本不改任何训练代码，直接调 `Stage1Aligner.build_batch` 量：

    B ∈ {1,2,4,8,16,32}   s/step · s/样本 · 峰值显存 · 有效 token 吞吐

**注意 pad 的代价**：B>1 时 RNA-FM 和 tokenizer 都要 pad 到 batch 内最长，
所以"样本数/秒"的收益会小于 B 的倍数 —— 这正是要量出来的东西。

跑法（**必须独占 GPU**）：
    python scripts/_batch_sweep.py --data data/processed/rnaqa_rnacentral_all.jsonl \
        --batches 1,2,4,8,16 --steps 4 --out reports/batch_sweep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402

from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402
from rnaqwen.paths import default_encoder, describe_encoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/rnaqa_rnacentral_all.jsonl")
    ap.add_argument("--llm", default=str(ROOT / "models/Qwen3-8B"))
    ap.add_argument("--encoder", default=str(ROOT / default_encoder()),
                    help="不填就走 E 轨默认（ERNIE-RNA）")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--steps", type=int, default=4, help="每个 batch 测几步（取中位数）")
    ap.add_argument("--nt-band", default=None,
                    help="限定序列长度区间 lo,hi（**控住 pad 这个混杂变量**；不填则自动取中位数 ±15%%）")
    ap.add_argument("--warmup-steps", type=int, default=2, help="先跑几步不算")
    ap.add_argument("--grad-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/batch_sweep.json")
    args = ap.parse_args()

    max_b = max(int(x) for x in args.batches.split(","))
    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    all_samples = [{"sequence": r["sequence"], "target": r.get("annotation") or ""}
                   for r in rows if (r.get("annotation") or "").strip()]

    # ⚠️ **长度分带是必须的**。第一版扫描没做这件事，量出来的结论是错的：
    #     B=1 抽到一条 94 nt 的 miRNA、B=2 抽到 1022 nt 的 lncRNA，
    #     于是"batch=2 比 batch=1 慢 4 倍" —— 那是 **pad 的代价，不是 batch 的代价**。
    # 语料是双峰的（22 nt 的成熟 miRNA + 1022 nt 的 lncRNA），不控长度就没法比。
    if args.nt_band:
        lo, hi = (int(x) for x in args.nt_band.split(","))
        samples = [s for s in all_samples if lo <= len(s["sequence"]) <= hi]
    else:
        lens = sorted(len(s["sequence"]) for s in all_samples)
        med = lens[len(lens) // 2]
        lo, hi = int(med * 0.85), int(med * 1.15)
        samples = [s for s in all_samples if lo <= len(s["sequence"]) <= hi]
    if len(samples) < max_b * 4:
        # 带太窄就放宽
        lo2, hi2 = int(lo * 0.6), int(hi * 1.4)
        samples = [s for s in all_samples if lo2 <= len(s["sequence"]) <= hi2]
    print(f"[sweep] 长度分带 [{lo}, {hi}] → {len(samples)} 条可用样本（控住 pad 这个混杂变量）")
    print(f"[sweep] 全部样本 {len(all_samples)} 条；batch ∈ {args.batches}；每档 {args.steps} 步")

    device = torch.device("cuda")
    cfg = Stage1Config(llm_path=args.llm, encoder_path=args.encoder,
                       grad_checkpointing=args.grad_checkpointing)
    t0 = time.time()
    model = Stage1Aligner(cfg)
    # ⚠️ 必须**逐个子模块**搬，不能 `Stage1Aligner(cfg).to(device)`。
    # `nn.Module.to()` 走的是内部 `_apply()`，不调用子模块自己的 `.to()` 覆写；
    # 而且实测**父模块 `.to()` 这条路径会触发一次 C++ 层 abort**
    # （无 Python traceback，栈落在 SDPA 的 math fallback 上）。
    # `train_stage1` 用的就是逐个子模块的写法 —— 这里必须跟它一致，否则量出来的不是训练的真实路径。
    model.encoder.to(device)
    model.llm.to(device)
    model.projection.to(device)
    model.projection.train()
    print(f"[sweep] 模型就绪 {time.time()-t0:.1f}s；{model.projection.describe()}")
    print(f"[sweep] device 校验: encoder={next(model.encoder.parameters()).device} "
          f"llm={next(model.llm.parameters()).device} proj={next(model.projection.parameters()).device}")

    opt = torch.optim.AdamW(model.projection.parameters(), lr=1e-6, weight_decay=0.05)

    report: dict = {"batches": {}, "data": str(args.data), "n_samples": len(samples),
                    "device": torch.cuda.get_device_name(0),
                    "mem_total_GiB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                    "grad_checkpointing": args.grad_checkpointing}
    print(f"\n{'batch':>6s} {'s/step(中位)':>13s} {'s/样本':>9s} {'样本/s':>9s} "
          f"{'相对加速':>9s} {'峰值显存 GiB':>12s} {'批内最大 RNA token':>16s}")
    print("-" * 92)

    base_per_sample = None
    cursor = 0
    for B in [int(x) for x in args.batches.split(",")]:
        # 取一段连续样本；循环使用
        batch_samples = [samples[(cursor + i) % len(samples)] for i in range(B)]
        cursor += B
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # warmup
        for _ in range(args.warmup_steps):
            loss = model.forward_loss([s["sequence"] for s in batch_samples],
                                      [s["target"] for s in batch_samples], device)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        times = []
        for _ in range(args.steps):
            t1 = time.time()
            loss = model.forward_loss([s["sequence"] for s in batch_samples],
                                      [s["target"] for s in batch_samples], device)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t1)

        med = statistics.median(times)
        per_sample = med / B
        peak = torch.cuda.max_memory_allocated() / 1024**3
        nt = max(len(s["sequence"]) for s in batch_samples)
        if base_per_sample is None:
            base_per_sample = per_sample
        speedup = base_per_sample / per_sample

        report["batches"][str(B)] = {
            "sec_per_step_median": round(med, 4),
            "sec_per_step_mean": round(statistics.mean(times), 4),
            "sec_per_sample": round(per_sample, 4),
            "samples_per_sec": round(1.0 / per_sample, 3),
            "speedup_vs_batch1": round(speedup, 3),
            "peak_mem_GiB": round(peak, 2),
            "max_rna_nt_in_batch": nt,
            "steps": args.steps,
        }
        print(f"{B:6d} {med:13.3f} {per_sample:9.3f} {1.0/per_sample:9.2f} "
              f"{speedup:9.2f}x {peak:12.2f} {nt:16d}")

    print()
    print("怎么读：")
    print("  `相对加速` 就是 batch 换来的真实吞吐收益（已经扣掉 pad 的代价）")
    print("  `峰值显存` 离 99.7 GiB 还有多少余量，决定还能不能往上开")
    print("  ⚠️ 开大 batch = 训练步数变少 ⟹ **LR 与 warmup 必须跟着调**，否则等于换了个优化问题")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[sweep] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
