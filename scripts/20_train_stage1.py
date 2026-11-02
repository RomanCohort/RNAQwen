"""P1 入口：Stage 1 模态对齐（只训投影，编码器与 LLM 全冻结）。

PLAN §1 的关键设计（照抄原版）：**Stage 1 的输入里没有文本 prompt** ——
模型只看到投影后的 RNA 表征，被要求产出该 RNA 的描述。
这从结构上堵死"绕过编码器去抄 prompt 文本"这条捷径。

超参照抄原版（PLAN §1）：
    10 epochs · batch=1 · wd=0.05 · warmup=2048 · AdamW β=(0.9, 0.98)
    LR 1e-4 → 8e-5 · warmup LR 1e-6 · linear warmup + cosine

跑法：
    python scripts/20_train_stage1.py \
        --data data/processed/annotations_50.jsonl \
        --llm models/Qwen3-8B --encoder models/rnafm \
        --epochs 10 --out checkpoints/stage1

完成判据（PLAN §3 P1）：**置零编码器后，对齐指标明显下降。**
训练完请跑 `scripts/22_zero_encoder_probe.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Config, train_stage1  # noqa: E402
from rnaqwen.paths import default_encoder, describe_encoder, track_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--llm", default="models/Qwen3-8B")
    ap.add_argument("--encoder", default=default_encoder(),
                    help="不填就走 E 轨默认（ERNIE-RNA）。R 轨复现请显式给 models/rnafm")
    ap.add_argument("--out", default="checkpoints/stage1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-final", type=float, default=8e-5)
    ap.add_argument("--warmup-steps", type=int, default=2048)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None, help="不填就自动选 cuda（没有则 cpu）")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--target-field", default="annotation", choices=["annotation", "literature"],
                    help="训练目标文本用哪个字段（D&C 用 annotation，AIS 用 literature）")
    # ---- E 轨（改进轨）开关。默认全关 = 复现轨 ----
    ap.add_argument("--compress-latent", type=int, default=None,
                    help="E 轨：把 RNA token 压到 N 个 latent（None = 复现轨原样）。"
                         "实测 64 能快 2.55×（序列 880→213）。**开了它就不能和复现轨分数并列汇报**")
    ap.add_argument("--compress-method", default="resampler",
                    choices=["resampler", "strided", "meanpool"])
    ap.add_argument("--tag", default="",
                    help="给输出目录加后缀，避免 E 轨覆盖复现轨的 checkpoint，例如 --tag e_compress")
    # ---- 收敛监控 ----
    ap.add_argument("--val-from", type=Path, default=None,
                    help="评测集目录（含 train.jsonl / val.jsonl）。给了就按 uid 拆出验证集，"
                         "每 --eval-every 步算 val loss + 置零探针。**强烈建议给**")
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="从这个 checkpoint 接着训（这个环境会杀进程，长跑必须能续）")
    # 定步数预算：做「压缩 vs 不压缩」这类配对对比时，两个 arm 必须停在同一步数。
    # 0 = 不限制。
    ap.add_argument("--max-steps", type=int, default=0,
                    help="训到这么多步就停（0=跑满 epochs）。配对对比用")
    # E 轨：去各向异性。给 34_compute_aniso_mean.py 产出的 .pt。
    # 实测 ERNIE-RNA 原始输出模长的 91.1% 落在跨序列公共方向上，减掉后特异部分保留 42.9%。
    ap.add_argument("--de-aniso", type=str, default=None,
                    help="E 轨：减掉编码器的公共方向（34_compute_aniso_mean.py 的产物）")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-samples", type=int, default=48)
    # 塌缩判据用**相对衰减**。实测：Δloss 从峰值 0.273(step1600) 衰减到 0.047(step13800)，
    # 绝对值判据一次都没触发，但那只是告警不是停机理由 —— 真正的停机理由是 val 平台期（过拟合）。
    ap.add_argument("--collapse-rel-drop", type=float, default=0.75,
                    help="Δloss 跌到滑动峰值的 (1-x) 以下 → 记一次塌缩嫌疑（默认 0.75，即跌到 25%%）")
    ap.add_argument("--collapse-min-peak", type=float, default=0.05,
                    help="Δloss 峰值低于此值就不算「曾经用过编码器」，不触发")
    ap.add_argument("--val-patience", type=int, default=12,
                    help="val_loss 连续多少次评估无改善 → 停（平台期 = 在过拟合）")
    ap.add_argument("--collapse-patience", type=int, default=3)
    ap.add_argument("--no-early-stop", action="store_true", help="塌缩了也不停（只告警）")
    args = ap.parse_args()

    rows = [r.as_dict() for r in read_jsonl(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    keep = []
    for r in rows:
        target = r.get("target_field") if False else (r.get(args.target_field) or "")
        if r.get("sequence") and target.strip():
            keep.append({"sequence": r["sequence"], "annotation": target, "uid": r.get("uid")})
    print(f"[stage1] {len(keep)}/{len(rows)} 条带可用目标文本（字段={args.target_field}）")

    # ---- 按评测集的 uid 划分拆 train/val（同一套哈希划分，和 P3 评测一致）----
    val_records = None
    if args.val_from:
        import json as _json

        def _uids(p):
            fp = args.val_from / p
            if not fp.exists():
                return set()
            return {_json.loads(l)["uid"] for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()}

        tr_uids, va_uids = _uids("train.jsonl"), _uids("val.jsonl")
        if tr_uids and va_uids:
            train_keep = [r for r in keep if r["uid"] in tr_uids]
            val_records = [r for r in keep if r["uid"] in va_uids]
            test_n = len(keep) - len(train_keep) - len(val_records)
            print(f"[stage1] 按 {args.val_from} 拆分：train {len(train_keep)} / "
                  f"val {len(val_records)} / 其余(未列入) {test_n}")
            keep = train_keep
        else:
            print(f"[stage1] ⚠️ {args.val_from} 里没读到 train/val uid，退化为不拆分")
    if not val_records:
        print("[stage1] ⚠️ 没给 --val-from：训练集不做划分，**无法判断是在学还是在背**")
    if not keep:
        print("[fail] 没有可用样本")
        return 1

    out_dir = args.out + (f"_{args.tag}" if args.tag else "")

    cfg = Stage1Config(
        llm_path=args.llm, encoder_path=args.encoder,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, lr_final=args.lr_final, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, out_dir=out_dir,
        grad_checkpointing=not args.no_grad_checkpointing,
        compress_latent=args.compress_latent, compress_method=args.compress_method,
        eval_every=args.eval_every, eval_samples=args.eval_samples,
        collapse_rel_drop=args.collapse_rel_drop, collapse_min_peak=args.collapse_min_peak,
        val_patience=args.val_patience, collapse_patience=args.collapse_patience,
        early_stop_on_collapse=not args.no_early_stop,
    
        max_steps=args.max_steps, de_aniso=args.de_aniso,)
    if args.compress_latent:
        print(f"[stage1] ⚠️ E 轨（改进轨）：RNA token 压缩到 {args.compress_latent} latent "
              f"（method={args.compress_method}）")
        print("[stage1] ⚠️ 这一版的数字**不能**和复现轨并列汇报（见 PLAN.md §7）")
    else:
        # ⚠️ 这里只能描述**压缩开关**，不能拿它推轨道标签。
        # 2026-10-09 踩过：只关压缩就打印「复现轨（R 轨）」，但 encoder 可能已经是
        # ERNIE-RNA —— 那仍然是 E 轨。轨道由 rnaqwen.paths.describe_encoder() 单独报。
        print("[stage1] RNA token 压缩：关闭（每个 token 都投影）")
    result = train_stage1(keep, cfg, device=args.device, val_records=val_records,
                          resume=str(args.resume_from) if args.resume_from else None)
    (Path(out_dir) / "run_summary.json").write_text(
        json.dumps({"config": cfg.as_dict(), "n_samples": len(keep),
                    "n_val": len(val_records or []), "result": result,
                    "track": track_of(cfg.encoder_path),
                    "compress_latent": cfg.compress_latent},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    # 评估曲线单独落一份，方便画图/对照
    evals = [h["eval"] for h in result.get("history", []) if isinstance(h, dict) and "eval" in h]
    if evals:
        (Path(out_dir) / "eval_curve.json").write_text(
            json.dumps(evals, indent=2, ensure_ascii=False), encoding="utf-8")
        last = evals[-1]
        print(f"[stage1] 末次评估：step={last['step']} val_loss={last['val_loss']} "
              f"Δloss(置零)={last['delta_loss_zeroed']} proj_std={last['proj_std']}")
        print(f"[stage1] 评估曲线 -> {Path(out_dir) / 'eval_curve.json'}")
    print(f"[stage1] 摘要 -> {Path(out_dir) / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
