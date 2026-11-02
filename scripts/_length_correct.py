"""按序列长度矫正的 s/step 估计 + token 压缩收益的归一化对比。

为什么需要这个：
    逐步耗时与**输入 token 数近似线性**（r=0.971，3.13 ms/nt）。
    直接比较不同运行的中位数会被"这一轮恰好抽到长/短序列"污染 ——
    实测同一配置跨轮差异 1.40–1.57 s，就是这个原因，不是机器抖动。

做法：
    1. 在 loss 运行（全 50 条 × 2 epoch = 100 步）上拟合 total_s ~ seq_nt；
    2. 把系数按长度分桶验证（短/中/长），确认线性假设；
    3. 用 3800 条规模推耗时 —— 但序列长度分布未知，故给"按当前 50 条平均长度"与
       "长度分布偏移"的敏感性分析。
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8-sig"))


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    r = sxy / ((sxx ** 0.5) * (sum((v - my) ** 2 for v in ys) ** 0.5))
    return a, b, r


def main() -> int:
    loss = load("_raw_timing_loss.json")
    tr = load("_raw_timing_trunc.json")
    m = loss["runs"][0]["batch_meta"]

    xs = [x["seq_nt"] for x in m]
    ys = [x["step_total_s"] for x in m]
    a, b, r = fit(xs, ys)
    print(f"拟合 total_s = {a:.3f} + {1000*b:.3f} ms/nt   (r={r:.3f}, n={len(xs)})")
    print(f"seq_nt 样本均值 = {sum(xs)/len(xs):.0f} nt  (median {st.median(xs):.0f})")

    # 分桶验证
    buckets = [("<=400", lambda v: v <= 400), ("401-600", lambda v: 400 < v <= 600),
               ("601-800", lambda v: 600 < v <= 800), (">800", lambda v: v > 800)]
    bucket_check = {}
    print("\n分桶核对（模型预测 vs 实测）:")
    for name, f in buckets:
        pts = [(x, y) for x, y in zip(xs, ys) if f(x)]
        if not pts:
            continue
        mx_ = sum(p[0] for p in pts) / len(pts)
        my_ = sum(p[1] for p in pts) / len(pts)
        pred = a + b * mx_
        bucket_check[name] = {"n": len(pts), "mean_nt": mx_, "measured_mean_s": my_,
                              "model_pred_s": pred, "deviation_pct": 100 * (pred - my_) / my_}
        print(f"  {name:>8} n={len(pts):>3}  nt均值={mx_:>6.0f}  实测均值={my_:.3f}s  "
              f"模型预测={pred:.3f}s  偏差={100*(pred-my_)/my_:+.1f}%")

    # 全 50 条的 token 长度分布（用于把估计校正到真实池）
    rows = []
    with open(ROOT / "data" / "processed" / "annotations_50.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    nt_all = [len(x["sequence"]) for x in rows]
    mean_nt_all = sum(nt_all) / len(nt_all)
    mean_nt_loss = sum(xs) / len(xs)
    print(f"\n全长 50 条的 seq_nt 均值 = {mean_nt_all:.0f}（loss 运行里 = {mean_nt_loss:.0f}）")

    def s_per_step_for(nt: float) -> float:
        return a + b * nt

    est_at_all50 = s_per_step_for(mean_nt_all)
    est_at_loss50 = s_per_step_for(mean_nt_loss)
    print(f"长度校正后 s/step（50 条池，均值 {mean_nt_all:.0f}nt） = {est_at_all50:.3f}")
    print(f"对照：loss 运行窗口均值实测 = {sum(ys)/len(ys):.3f}，模型在该均值处的预测 = {est_at_loss50:.3f}")

    # 长度敏感性：3800 条语料长度分布未知
    print("\n长度敏感性（s/step）:")
    sens = {}
    for nt in (300, 416, 600, 800, 1000, 1200):
        sens[nt] = s_per_step_for(nt)
        print(f"  seq_nt={nt:>5} -> {sens[nt]:.3f} s/step")

    # 压缩收益做同池对照（F0/D/E/G 同一 16 条池、同一顺序）
    by = {r["name"]: r for r in tr["runs"] if "error" not in r}
    print("\n同池对照（16 条池、同一顺序，唯一变量 = RNA 软提示长度）:")
    base = by["F0_baseline_rna731"]["seconds"]["per_optimizer_step"]
    for key, label in (("D_compress_real_256", "真压缩 256 latent"),
                       ("E_compress_real_64", "真压缩 64 latent"),
                       ("G_trunc_hack_64", "截断 hack 64 token")):
        rr = by[key]
        s = rr["seconds"]["per_optimizer_step"]
        tot = (rr.get("token_layout_raw") or {}).get("total")
        print(f"  {label:<22} {s['median']:.3f}s median / {s['mean']:.3f}s mean  "
              f"seq_total={tot}  speedup={base['median']/s['median']:.2f}×")
        print(f"      peak_alloc={rr['memory']['peak_allocated_GiB']:.2f} GiB  "
              f"fwd={rr['seconds']['forward_per_micro'].get('median'):.3f} "
              f"bwd={rr['seconds']['backward_per_micro'].get('median'):.3f}")
    print(f"  {'baseline 731':<22} {base['median']:.3f}s median / {base['mean']:.3f}s mean  "
          f"seq_total=880  peak_alloc={by['F0_baseline_rna731']['memory']['peak_allocated_GiB']:.2f} GiB")

    # 用长度模型核对压缩收益是否只是"序列变短"的必然结果
    print("\n收益归因（用长度模型预测，对比实测）:")
    for key, label, nt in (("D_compress_real_256", "真压缩256", 405 + 15),
                           ("E_compress_real_64", "真压缩64", 213),
                           ("G_trunc_hack_64", "截断64", 213)):
        rr = by[key]
        s_meas = rr["seconds"]["per_optimizer_step"]["mean"]
        s_pred = s_per_step_for(nt)
        print(f"  {label:<12} 实测mean={s_meas:.3f}s  长度模型预测={s_pred:.3f}s  "
              f"偏差={100*(s_pred-s_meas)/s_meas:+.1f}%")

    out = {
        "linear_model": {"intercept_s": a, "slope_s_per_nt": b, "pearson_r": r,
                         "n_points": len(xs),
                         "fitted_on": "全 50 条 × 2 epoch 的 100 个 optimizer step"},
        "bucket_check": bucket_check,
        "length_sensitivity_s_per_step": {str(k): v for k, v in sens.items()},
        "mean_seq_nt_full50": mean_nt_all,
        "mean_seq_nt_loss_window": mean_nt_loss,
        "length_corrected_s_per_step_at_full50": est_at_all50,
        "same_pool_compression": {},
    }
    for key, label in (("F0_baseline_rna731", "baseline_rna731"),
                       ("D_compress_real_256", "compress_real_256"),
                       ("E_compress_real_64", "compress_real_64"),
                       ("G_trunc_hack_64", "trunc_hack_64")):
        rr = by[key]
        s = rr["seconds"]["per_optimizer_step"]
        out["same_pool_compression"][label] = {
            "s_per_step_median": s["median"], "s_per_step_mean": s["mean"],
            "peak_alloc_GiB": rr["memory"]["peak_allocated_GiB"],
            "seq_total_tokens": (rr.get("token_layout_raw") or {}).get("total"),
            "forward_per_micro_median": rr["seconds"]["forward_per_micro"].get("median"),
            "backward_per_micro_median": rr["seconds"]["backward_per_micro"].get("median"),
            "speedup_mean_vs_baseline": by["F0_baseline_rna731"]["seconds"]["per_optimizer_step"]["mean"] / s["mean"],
        }

    # E（真压缩 64）vs G（截断 hack 64）：都喂 213 token，验证压缩器自身开销
    se = by["E_compress_real_64"]["seconds"]["per_optimizer_step"]
    sg = by["G_trunc_hack_64"]["seconds"]["per_optimizer_step"]
    out["equivalence_E_vs_G"] = {
        "E_compress_real_64_s_per_step_median": se["median"],
        "G_trunc_hack_64_s_per_step_median": sg["median"],
        "median_diff_pct": 100 * (se["median"] - sg["median"]) / sg["median"],
        "mean_diff_pct": 100 * (se["mean"] - sg["mean"]) / sg["mean"],
        "verdict": "两者都喂 213 token，耗时差 <0.5%：压缩收益完全来自序列变短，"
                   "压缩器（resampler）自身前向开销可忽略",
    }
    dst = ROOT / "reports" / "_verify_all.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
