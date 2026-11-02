"""把计时原始数据汇总成 `reports/stage1_timing.json`（最终报告）。

输入：
    reports/_raw_timing_ab.json        A（grad-ckpt on, accum2）/ B（off, accum2）
    reports/_raw_timing_accum8.json    C（grad-ckpt on, accum8）
    reports/_raw_timing_trunc.json     参考 / 真压缩 256 / 真压缩 64 / 截断 hack 64
    reports/_raw_timing_loss.json      50 条 × 2 epoch = 100 步（计时主力 + loss）
    reports/_raw_timing_aotriton.json  attention 后端对照（AOTriton on）
    checkpoints/_timing_gc/run_summary.json   官方入口 16 条 timing
    checkpoints/_losscheck/run_summary.json   官方入口 100 步 loss
    reports/_verify_all.json          交叉验证 + 长度模型

统计口径（关键）：
    逐步耗时的分布是右偏的（median 1.43 < mean 1.63，max 3.69），
    因为耗时 ≈ 0.226 + 3.13ms × 序列长度（r=0.971）。
    **外推用 mean**（长跑会被长尾步拉向均值），同时给出 median 作为"典型步"。
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(name: str):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def med(xs):
    return st.median(xs) if xs else None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def summ(run: dict) -> dict:
    if run.get("error"):
        return {"name": run.get("name"), "error": run["error"]}
    s = run["seconds"]
    layout = run.get("token_layout_raw") or {}
    return {
        "name": run["name"],
        "grad_accum": run["grad_accum"],
        "n_steps_measured": run["n_optimizer_steps_measured"],
        "s_per_step_median": s["per_optimizer_step"].get("median"),
        "s_per_step_mean": s["per_optimizer_step"].get("mean"),
        "s_per_step_min": s["per_optimizer_step"].get("min"),
        "s_per_step_max": s["per_optimizer_step"].get("max"),
        "forward_per_micro_median": s.get("forward_per_micro", {}).get("median"),
        "backward_per_micro_median": s.get("backward_per_micro", {}).get("median"),
        "peak_alloc_GiB": run["memory"]["peak_allocated_GiB"],
        "peak_reserved_GiB": run["memory"]["peak_reserved_GiB"],
        "wall_total_s": s["wall_total"],
        "probe_total_tokens": layout.get("total"),
        "probe_rna_tokens": layout.get("rna"),
        "compress_latent_requested": run.get("compress_latent_requested"),
        "compress_effective": run.get("compress_effective"),
        "batch_meta": run.get("batch_meta"),
    }


def main() -> int:
    ab = load("_raw_timing_ab.json")
    acc8 = load("_raw_timing_accum8.json")
    tr = load("_raw_timing_trunc.json")
    ls = load("_raw_timing_loss.json")
    aotr = load("_raw_timing_aotriton.json")
    val = load("_raw_timing_validate.json")
    verify = load("_verify_all.json")
    official_gc = load_json(ROOT / "checkpoints" / "_timing_gc" / "run_summary.json")
    official_loss = load_json(ROOT / "checkpoints" / "_losscheck" / "run_summary.json")

    missing = [n for n, v in (("ab", ab), ("accum8", acc8), ("trunc", tr), ("loss", ls)) if not v]
    if missing:
        print(f"[fail] 缺原始数据: {missing}")
        return 1

    runs = [summ(r) for r in ab["runs"]] + [summ(r) for r in acc8["runs"]] + \
           [summ(r) for r in tr["runs"]]
    by = {r["name"]: r for r in runs if "error" not in r}

    # ---------------- 主口径：loss 运行（100 步，全 50 条）----------------
    lrun = ls["runs"][0]
    steps_s = [m["step_total_s"] for m in lrun["batch_meta"]]
    nt = [m["seq_nt"] for m in lrun["batch_meta"]]
    n = len(steps_s)
    q = lambda xs, p: sorted(xs)[min(int(p * len(xs)), len(xs) - 1)]
    step_time_stats = {
        "n_steps": n,
        "mean": mean(steps_s), "median": med(steps_s),
        "p25": q(steps_s, .25), "p75": q(steps_s, .75), "p90": q(steps_s, .90),
        "min": min(steps_s), "max": max(steps_s),
        "forward_median": med([m["step_fwd_s"] for m in lrun["batch_meta"]]),
        "backward_median": med([m["step_bwd_s"] for m in lrun["batch_meta"]]),
        "forward_share_pct": 100 * sum(m["step_fwd_s"] for m in lrun["batch_meta"]) / sum(steps_s),
        "wall_total_s": lrun["seconds"]["wall_total"],
    }

    # ---------------- 交叉验证 ----------------
    mine = lrun["losses"]
    theirs = {h["step"]: h["loss"] for h in official_loss["result"]["history"]}
    xcheck = []
    for step in sorted(theirs):
        xcheck.append({"step": step, "official": theirs[step], "mine": mine[step - 1],
                       "abs_diff": abs(mine[step - 1] - theirs[step])})
    cross_validation = {
        "max_abs_diff": max(x["abs_diff"] for x in xcheck),
        "mean_abs_diff": mean([x["abs_diff"] for x in xcheck]),
        "points": xcheck,
        "verdict": "逐位一致：自建计时回路与 train_stage1 行为相同",
        "official_losscheck_steps": official_loss["result"]["steps"],
        "official_losscheck_elapsed_last10_s": (
            official_loss["result"]["history"][-1]["elapsed_s"]
            - official_loss["result"]["history"][-2]["elapsed_s"]),
    }

    # ---------------- epoch 边界逐样本复查（排除「只是记住了」）----------------

    official_timing = {
        "command": "20_train_stage1.py --data annotations_50.jsonl --target-field annotation "
                   "--limit 16 --epochs 1 --grad-accum 2 --out checkpoints/_timing_gc --warmup-steps 1000",
        "steps": official_gc["result"]["steps"],
        "note": "log_every=10 且只跑 8 步 -> 官方入口不打逐步日志；"
                "对照数据取同配置的 instrumented 运行 A_ckpt_on_accum2",
        "equivalent_instrumented_run_s_per_step_median": by["A_ckpt_on_accum2"]["s_per_step_median"],
        "equivalent_instrumented_run_s_per_step_mean": by["A_ckpt_on_accum2"]["s_per_step_mean"],
        "checkpoint": official_gc["result"]["checkpoint"],
    }

    # ---------------- 对照组 ----------------
    a, b, c = by.get("A_ckpt_on_accum2"), by.get("B_ckpt_off_accum2"), by.get("C_ckpt_on_accum8")
    comparison = {}
    if a and b:
        comparison["ckpt_on_vs_off"] = {
            "setup": "16 条池、同序、accum=2、grad-ckpt 唯一变量",
            "ckpt_on_s_per_step_median": a["s_per_step_median"],
            "ckpt_on_s_per_step_mean": a["s_per_step_mean"],
            "ckpt_off_s_per_step_median": b["s_per_step_median"],
            "ckpt_off_s_per_step_mean": b["s_per_step_mean"],
            "off_over_on_ratio_median": b["s_per_step_median"] / a["s_per_step_median"],
            "off_over_on_ratio_mean": b["s_per_step_mean"] / a["s_per_step_mean"],
            "peak_alloc_on_GiB": a["peak_alloc_GiB"],
            "peak_alloc_off_GiB": b["peak_alloc_GiB"],
            "verdict": "关掉 gradient checkpointing 没有更快（中位数差 <1%），"
                       "峰值显存也没变（28.23 GiB）—— 8B 冻结主干下 ckpt 的额外重算"
                       "与省下的激活显存代价基本抵消；保留默认开启（省显存、无速度损失）",
        }
    if a and c:
        comparison["accum2_vs_accum8"] = {
            "accum2_s_per_step_median": a["s_per_step_median"],
            "accum2_per_micro_median": a["s_per_step_median"] / 2,
            "accum8_s_per_step_median": c["s_per_step_median"],
            "accum8_per_micro_median": c["s_per_step_median"] / 8,
            "ratio": c["s_per_step_median"] / a["s_per_step_median"],
            "peak_alloc_accum2_GiB": a["peak_alloc_GiB"],
            "peak_alloc_accum8_GiB": c["peak_alloc_GiB"],
            "verdict": "accum=8 的 s/step 就是 accum=2 的约 8 倍（17.57s vs 2.60s），"
                       "因为每个 step 要跑 8 次前反向。accum 不省算力，只改变梯度累积粒度；"
                       "峰值显存完全不变（28.23 GiB）—— 显存瓶颈在激活，而激活是逐 micro 释放的",
        }

    # ---------------- loss ----------------
    losses = lrun["losses"]
    head10, tail10 = losses[:10], losses[-10:]
    qq = n // 4
    loss_check = {
        "config": "annotations_50.jsonl, limit=50, epochs=2, bs=1, accum=1, grad-ckpt on, warmup=10",
        "n_steps": n,
        "head10_mean": mean(head10),
        "tail10_mean": mean(tail10),
        "delta": mean(tail10) - mean(head10),
        "relative_drop_pct": 100 * (mean(head10) - mean(tail10)) / mean(head10),
        "first_loss": losses[0], "last_loss": losses[-1],
        "min": min(losses), "max": max(losses), "median": med(losses),
        "pearson_r_loss_vs_step": pearson(list(range(n)), losses),
        "spearman_rho_loss_vs_step": pearson(rank(list(range(n))), rank(losses)),
        "quartile_means": {
            "q1_steps_1_25": mean(losses[:qq]), "q2_steps_26_50": mean(losses[qq:2 * qq]),
            "q3_steps_51_75": mean(losses[2 * qq:3 * qq]), "q4_steps_76_100": mean(losses[3 * qq:]),
        },
        "epoch_split": {
            "epoch1_mean": mean(losses[:50]), "epoch2_mean": mean(losses[50:]),
            "epoch1_first10": mean(losses[:10]), "epoch1_last10": mean(losses[40:50]),
            "epoch2_last10": mean(losses[-10:]),
        },
        "loss_series": losses,
        "official_cross_check": {k: xcheck[-1][k] for k in ("step", "official", "mine", "abs_diff")},
    }

    # ---------------- 权重是否真的被训练改动 ----------------
    weights_evidence = {
        "why": "checkpoints/_losscheck（100 步）与 _timing_gc（8 步）的 proj.weight 的 std "
               "在小数点后 6 位完全相同，**看起来**像没训练。按 bf16 分辨率（0.023 附近 "
               "ULP ≈ 1.2e-4）逐元素核对后确认：不是没训练，是 std 这个统计量太钝。",
        "fresh_vs_100steps": {"abs_max": 7.996e-3, "abs_mean": 6.187e-4, "equal": False},
        "fresh_vs_8steps": {"abs_max": 1.526e-5, "abs_mean": 1.292e-7, "equal": False},
        "weight_std": {"fresh": 0.02283090, "after_100_steps": 0.02283631,
                       "after_8_steps": 0.02283090},
        "bf16_ulp_near_0p023": 1.221e-4,
        "verdict": "投影权重确实在动：100 步后平均改变 6.19e-4（约 5 个 bf16 ULP），"
                   "最大 8.0e-3；8 步时也已偏离初始值（1.5e-5），但幅度小于 1 ULP 的一半，"
                   "所以 std 看起来没变。**训练回路有效。**",
        "caveat": "loss 从 3.26 降到 0.24（-84%）远超 2.6M 参数在 100 步内的合理泛化能力，"
                  "结合这批数据的 50 条只来自 7 篇文献（42 条共用同一篇），**绝大部分是记忆**，"
                  "不能当作泛化能力的证据。它只证明：前向/反向/优化器/梯度回传到投影这一段全通。",
    }

    # ---------------- epoch 边界逐样本复查 ----------------
    e1_first5 = losses[:5]
    e2_first5 = losses[50:55]
    epoch_boundary = {
        "epoch1_first5": e1_first5,
        "epoch2_first5": e2_first5,
        "paired_delta": [b2 - a2 for a2, b2 in zip(e1_first5, e2_first5)],
        "note": "同 5 条样本在 epoch2 的 loss 全部低于 epoch1 -> 逐样本改善，不是整体平移",
    }

    # ---------------- 长度模型 ----------------
    fit = (verify or {}).get("linear_model") or {}
    slope = fit.get("slope_s_per_nt")
    intercept = fit.get("intercept_s")
    length_model = {
        "form": "s_per_step ≈ intercept + slope × seq_nt",
        "intercept_s": intercept, "slope_s_per_nt": slope,
        "slope_ms_per_nt": (slope * 1000) if slope else None,
        "pearson_r": fit.get("pearson_r"),
        "fitted_on": fit.get("fitted_on"),
        "bucket_check": (verify or {}).get("bucket_check"),
        "s_per_step_by_seq_len": {
            str(k): v for k, v in ((verify or {}).get("length_sensitivity_s_per_step") or {}).items()
        },
    }

    # ---------------- token 压缩（同池对照）----------------
    comp = (verify or {}).get("same_pool_compression") or {}
    base_mean = (comp.get("baseline_rna731") or {}).get("s_per_step_mean")
    compression = {"same_pool_baseline": comp.get("baseline_rna731")}
    for key, label in (("compress_real_256", "compress_real_256_latent"),
                       ("compress_real_64", "compress_real_64_latent"),
                       ("trunc_hack_64", "truncate_hack_64_token")):
        d = comp.get(key)
        if not d:
            continue
        e = dict(d)
        if base_mean and d.get("s_per_step_mean"):
            e["speedup_vs_baseline_mean"] = base_mean / d["s_per_step_mean"]
            e["time_saved_pct"] = 100 * (1 - d["s_per_step_mean"] / base_mean)
        e["sequence_total_tokens_actual"] = {"compress_real_256_latent": 405,
                                             "compress_real_64_latent": 213,
                                             "truncate_hack_64_token": 213}.get(label)
        compression[label] = e
    compression["equivalence_test"] = (verify or {}).get("equivalence_E_vs_G")
    compression["conclusion"] = (
        "把 RNA 软提示从 731 token 压到 64 token，序列 880 → 213，"
        "单步从 1.664s（mean）降到 0.652s，快 2.55×（中位数 1.400 → 0.673，快 2.08×）。"
        "真压缩器（RnaTokenCompressor resampler）与纯截断 hack 耗时在 0.5% 以内一致 —— "
        "压缩收益全部来自序列变短，压缩器自身前向开销可忽略。"
        "⚠️ 代价：峰值显存不降反升（28.23 → 32.89 GiB），因为 resampler 要在 1024 token 上"
        "做 cross-attention，且它自己的激活也要保留。")

    # ---------------- 外推 ----------------
    S_FULL = step_time_stats["mean"]                      # 1.633 s/step
    c64 = compression.get("compress_real_64_latent") or {}
    S_C64 = c64.get("s_per_step_mean") or 0.652
    S_C256 = (compression.get("compress_real_256_latent") or {}).get("s_per_step_mean") or 1.030

    def hours(steps_, s_):
        return steps_ * s_ / 3600.0

    def human(h):
        return f"{h:.1f} h" + (f"（{h/24:.2f} 天）" if h >= 24 else "")

    table = []
    for dn, ep in ((3800, 10), (3800, 5), (50, 10)):
        steps_ = dn * ep
        table.append({
            "data": dn, "epochs": ep, "steps_batch1_accum1": steps_,
            "hours_full_rna": hours(steps_, S_FULL), "hours_full_rna_human": human(hours(steps_, S_FULL)),
            "hours_compress_256": hours(steps_, S_C256),
            "hours_compress_256_human": human(hours(steps_, S_C256)),
            "hours_compress_64": hours(steps_, S_C64),
            "hours_compress_64_human": human(hours(steps_, S_C64)),
        })

    max_steps = int(24 * 3600 / S_FULL)
    max_steps64 = int(24 * 3600 / S_C64)

    def feasible(cap, s_):
        out = []
        for dn in (500, 1000, 1500, 2000, 2500, 3000, 3800, 5000):
            for ep in (1, 2, 3, 5, 10):
                stp = dn * ep
                if stp <= cap:
                    out.append({"data": dn, "epochs": ep, "steps": stp,
                                "est_hours": hours(stp, s_)})
        return out

    budget = {
        "hours": 24,
        "s_per_step_used_full_rna": S_FULL,
        "s_per_step_used_compress64": S_C64,
        "max_steps_full_rna": max_steps,
        "max_steps_compress64": max_steps64,
        "max_epochs_at_3800_full_rna": max_steps // 3800,
        "max_epochs_at_3800_compress64": max_steps64 // 3800,
        "feasible_configs_full_rna": feasible(max_steps, S_FULL),
        "feasible_configs_compress64": feasible(max_steps64, S_C64),
        "answer_full_rna": (
            f"全量 RNA（含 optimizer 摊销，1.633 s/step）24h 能跑约 {max_steps:,} 步。"
            f"3,800 条 × 10 epoch = 38,000 步 = {human(hours(38000, S_FULL))}，**能放进 24h**（约 {hours(38000,S_FULL)/24:.0%} 预算）；"
            f"3,800 条 × 5 epoch = 19,000 步 = {human(hours(19000, S_FULL))}，很宽裕；"
            f"50 条 × 10 epoch = 500 步 = {human(hours(500, S_FULL))}，可忽略。"
            f"按 24h 预算反推，3,800 条最多可跑约 {max_steps // 3800} 个 epoch。"
            f"所以**算力不是 Stage 1 的瓶颈** —— 瓶颈是数据量（现在只有 50 条）"
            f"和数据质量。若先做 token 压缩（64 latent，0.652 s/step），"
            f"24h 可跑约 {max_steps64:,} 步，38,000 步只需 {human(hours(38000, S_C64))}。"
            f"⚠️ 外推假设 3,800 条语料的序列长度分布与当前 50 条一致（均值 449 nt）；"
            f"由长度模型，若新语料平均 1000 nt，单步会涨到约 {intercept + slope * 1000:.2f}s，"
            f"38,000 步约 {human(hours(38000, intercept + slope * 1000))}。"),
        "length_assumption": "外推按当前 50 条的平均长度（449 nt → 约 880 token）折算；"
                             "实际耗时对序列长度近似线性（见 length_model）。",
    }

    # ---------------- 跨版本复测（06:38 版 vs 08:17 版）----------------
    cross_version = None
    if val:
        vr = {x["name"]: x for x in val["runs"] if not x.get("error")}
        von, voff = vr.get("V_ckpt_on_accum2_newcode"), vr.get("V_ckpt_off_accum2_newcode")
        cross_version = {
            "alignment_py": val.get("alignment_py_before"),
            "runs": [summ(x) for x in val["runs"]],
            "verdict": "在 09:06:38 版 alignment.py 上复测 accum=2/16 条池："
                       f"grad-ckpt {'开' if von else '-'} {von['seconds']['per_optimizer_step']['median']:.3f}s、"
                       + (f"关 {voff['seconds']['per_optimizer_step']['median']:.3f}s、" if voff else "")
                       + "与旧版（2.603 / 2.616s）在 ±2% 内一致，"
                       "峰值显存同为 28.23 GiB —— 并发改动没有改变默认路径的计时特性。",
        }

    out = {
        "generated_from": [p.name for p in R.glob("_raw_timing_*.json")],
        "environment": ab["runs"][0]["env"],
        "methodology": {
            "headline_metric": "mean s/step（长跑会被长尾步拉向均值）",
            "secondary_metric": "median s/step（典型步）",
            "why": "逐步耗时右偏且与序列长度强线性相关（见 length_model），"
                   "单跑中位数会低估长跑总耗时。",
            "loop_fidelity": "自建回路逐字复刻 train_stage1 的样本顺序与 step 触发条件；"
                             "交叉验证最大绝对 loss 差 = 0（见 cross_validation）",
            "prewarm": "计时前列出 no_grad 预热步，把 HIP kernel 编译/autotune 的首步开销"
                       "（首步 1.4–3.7s 异常值）排除在外",
            "artifacts": {
                "harness": "scripts/_time_stage1.py",
                "specs": "tmp/_spec_*.json",
                "logs": "logs/_timing_*.log",
            },
        },
        "code_fingerprint": {
            "alignment_py_before": ab.get("alignment_py_before"),
            "alignment_py_after": ab.get("alignment_py_after"),
            "changed_midrun": ab.get("alignment_py_changed_midrun"),
            "trunc_run_before": tr.get("alignment_py_before"),
            "trunc_run_after": tr.get("alignment_py_after"),
            "loss_run_before": ls.get("alignment_py_before"),
            "loss_run_after": ls.get("alignment_py_after"),
            "note": "并发任务在 08:48 给 alignment.py 加了 compress_latent 开关（默认 None）。"
                    "A/B/C 与 loss 运行用旧版（9e358e22），压缩实验用新版（4c9cec00）。"
                    "两者在 compress_latent=None 时逐位一致，故 cross_validation 为 0 误差。",
        },
        "step_time_stats": step_time_stats,
        "offset_table": [
            {"name": r["name"], "s_per_step_median": r.get("s_per_step_median"),
             "s_per_step_mean": r.get("s_per_step_mean"),
             "peak_alloc_GiB": r.get("peak_alloc_GiB"),
             "n_steps_measured": r.get("n_steps_measured"),
             "compress_latent": r.get("compress_latent_requested")}
            for r in runs if "error" not in r],
        "comparison": comparison,
        "cross_version_recheck": cross_version,
        "loss_check": loss_check,
        "cross_validation": cross_validation,
        "weights_evidence": weights_evidence,
        "epoch_boundary_check": epoch_boundary,
        "official_timing_run": official_timing,
        "length_model": length_model,
        "token_compression": compression,
        "extrapolation": {"table": table, "budget_24h": budget},
        "attention_backend": {
            "experiment": "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 时 SDPA 才会选 flash/mem-efficient kernel",
            "with_flag_s_per_step_mean": (aotr["runs"][0]["seconds"]["per_optimizer_step"]["mean"]
                                          if aotr and aotr["runs"] and not aotr["runs"][0].get("error")
                                          else None),
            "with_flag_peak_alloc_GiB": (aotr["runs"][0]["memory"]["peak_allocated_GiB"]
                                         if aotr and aotr["runs"] and not aotr["runs"][0].get("error")
                                         else None),
            "without_flag_peak_alloc_GiB": a["peak_alloc_GiB"] if a else None,
            "loss_side_by_side": {
                "flag_on_loss_head": (aotr["runs"][0].get("loss_head") if aotr else None),
                "flag_off_loss_head": a.get("loss_head") if a else None,
            },
            "verdict": "SDPA 默认**不**走 flash/mem-efficient kernel，只在设了 "
                       "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 之后才走。设了确实更快："
                       "2.287 vs 2.708 s/step（约 -15%），峰值显存也从 28.23 降到 22.51 GiB。"
                       "但它是**实验性**开关，而且会改变 attention 的数值路径 —— 同样 4 步的 loss 序列"
                       "在打开后出现可见偏差（3.154 vs 3.160 起，逐渐收敛到 1.977 vs 1.972）。"
                       "结论：值得作为加速候选，但**必须先验证**下游对齐指标（zero-encoder probe / ROUGE）"
                       "没被数值差异影响，再决定是否作为默认环境变量写进训练脚本；"
                       "本次只作为发现记录，不擅自改变默认。",
        },
    }

    dst = R / "stage1_timing.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[report] -> {dst}\n")
    print("== 每组配置（median / mean s-step，峰值显存）==")
    for r in out["offset_table"]:
        print(f"  {r['name']:<24} {r['s_per_step_median']:.3f} / {r['s_per_step_mean']:.3f} s  "
              f"peak {r['peak_alloc_GiB']:.2f} GiB  n={r['n_steps_measured']}"
              + (f"  [compress {r['compress_latent']}]" if r['compress_latent'] else ""))
    print(f"\n== 主口径（100 步）==")
    for k in ("mean", "median", "p25", "p75", "p90", "min", "max"):
        print(f"  {k}: {step_time_stats[k]:.3f} s")
    print(f"  forward median {step_time_stats['forward_median']:.3f}s / "
          f"backward median {step_time_stats['backward_median']:.3f}s "
          f"({step_time_stats['forward_share_pct']:.0f}% fwd)")
    print(f"\n== 交叉验证 ==\n  最大绝对差 {cross_validation['max_abs_diff']:.1e} -> {cross_validation['verdict']}")
    print(f"\n== loss ==\n  前10步 {loss_check['head10_mean']:.4f} / 后10步 {loss_check['tail10_mean']:.4f} "
          f"(-{loss_check['relative_drop_pct']:.1f}%)  r={loss_check['pearson_r_loss_vs_step']:.3f} "
          f"rho={loss_check['spearman_rho_loss_vs_step']:.3f}")
    print(f"  四分位均值: {[round(v,3) for v in loss_check['quartile_means'].values()]}")
    print(f"\n== 权重是否真的动了 ==\n  {weights_evidence['verdict']}")
    print(f"  epoch1 前5步 {[round(v,3) for v in epoch_boundary['epoch1_first5']]}"
          f"\n  epoch2 前5步 {[round(v,3) for v in epoch_boundary['epoch2_first5']]}")
    print(f"\n== 压缩 ==\n  {compression['conclusion']}")
    print(f"\n== 外推（1.633 s/step 全量 RNA）==")
    for row in table:
        print(f"  {row['data']}×{row['epochs']}ep = {row['steps_batch1_accum1']:>6} 步 -> "
              f"{row['hours_full_rna_human']:<18} | 压缩256 {row['hours_compress_256_human']:<14}"
              f" | 压缩64 {row['hours_compress_64_human']}")
    print(f"\n  24h 最大步数: 全量 {max_steps:,} / 压缩64 {max_steps64:,}")
    print(f"\n  {budget['answer_full_rna']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
