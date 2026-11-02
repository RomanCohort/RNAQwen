"""P1 验收：**置零编码器探针**（PLAN §3 P1 的完成判据）。

判据原文：
    "置零编码器后，对齐指标明显下降"

这条是整个计划里最重要的自检 —— 它同时挡两个坑（PLAN §5 风险登记）：
  * **投影塌缩 / 编码器被绕过**：loss 降了，但置零毫无影响 → 模型压根没用上序列；
  * **QA 里复述了序列**：指标虚高 → 置零后分数不掉，就是这个症状。

做法：
    对同一批样本跑两次前向（teacher forcing），一次用真实 RNA 表征、一次把表征置零，
    比较 loss / token 准确率。真对齐的模型，置零后 loss 必须**明显**上升。

跑法：
    python scripts/22_zero_encoder_probe.py --ckpt checkpoints/stage1/projection_final.pt \
        --data data/processed/annotations_50.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402
from rnaqwen.paths import default_encoder, describe_encoder  # noqa: E402


@torch.no_grad()
def evaluate(model: Stage1Aligner, samples: list[dict], device: torch.device,
             zero_encoder: bool) -> dict:
    """返回平均 loss 与答案 token 的逐 token 准确率。"""
    total_loss, total_tok, correct = 0.0, 0, 0
    orig_forward = model.encoder.forward

    if zero_encoder:
        from rnaqwen.models.encoder import EncoderOutput

        def zero_forward(seqs):  # noqa: ANN001
            out = orig_forward(seqs)
            return EncoderOutput(hidden=torch.zeros_like(out.hidden), mask=out.mask, lengths=out.lengths)

        model.encoder.forward = zero_forward  # type: ignore[assignment]

    try:
        for s in samples:
            inputs, attn, labels = model.build_batch([s["sequence"]], [s["target"]], device)
            out = model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
            total_loss += float(out.loss)
            shift_logits = out.logits[:, :-1]
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100
            if mask.any():
                pred = shift_logits[mask].argmax(-1)
                correct += int((pred == shift_labels[mask]).sum())
                total_tok += int(mask.sum())
    finally:
        model.encoder.forward = orig_forward  # type: ignore[assignment]

    n = max(len(samples), 1)
    return {
        "loss": total_loss / n,
        "token_accuracy": correct / total_tok if total_tok else None,
        "n_samples": len(samples),
        "n_answer_tokens": total_tok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Stage1 产出的 projection_*.pt")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--llm", default="models/Qwen3-8B")
    ap.add_argument("--encoder", default=default_encoder(),
                    help="不填就走 E 轨默认（ERNIE-RNA）。R 轨复现请显式给 models/rnafm")
    ap.add_argument("--target-field", default="annotation", choices=["annotation", "literature"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "zero_encoder_probe.json")
    # E 轨 checkpoint 要对齐压缩配置（默认 None = 读 checkpoint 里存的）
    ap.add_argument("--compress-latent", type=int, default=None,
                    help="覆盖 checkpoint 里的 compress_latent（E 轨 checkpoint 一般不用填，会自动读到）")
    ap.add_argument("--compress-method", default=None,
                    help="覆盖 checkpoint 里的 compress_method")
    # 判据阈值。
    # ⚠️ 实测：E 轨 run1 的 Δloss 从峰值 0.273 衰减到 0.047，但 ROUGE-1 反而从 0.208 涨到 0.303。
    #    说明 Δloss 是**逐 token 平均**，答案变长/变准会把它摊薄，低 Δloss ≠ 编码器没用。
    #    所以这个探针的结论要和 31_stage1_rouge_curve.py 一起读，别单看这里的 PASS/FAIL。
    ap.add_argument("--pass-delta", type=float, default=0.5)
    ap.add_argument("--weak-delta", type=float, default=0.1)
    args = ap.parse_args()

    rows = [r.as_dict() for r in read_jsonl(args.data)]
    samples = []
    for r in rows:
        t = r.get(args.target_field) or ""
        if r.get("sequence") and t.strip():
            samples.append({"sequence": r["sequence"], "target": t})
        if args.limit and len(samples) >= args.limit:
            break
    if not samples:
        print("[fail] 没有样本")
        return 1

    blob = torch.load(args.ckpt, map_location="cpu")

    # checkpoint 里存的 config 是**训练时**的完整快照，字段集合和 Stage1Config 不完全一致
    # （多出 collapse_delta 之类的旧字段，且 llm_path/encoder_path 会和命令行参数撞名）。
    # 以前直接 **blob["config"] 会 TypeError；这里只挑 dataclass 认识的字段，
    # 且命令行显式给的路径优先。
    import dataclasses

    known = {f.name for f in dataclasses.fields(Stage1Config)}
    saved = {k: v for k, v in (blob.get("config") or {}).items() if k in known}
    dropped = sorted(set(blob.get("config") or {}) - known)
    if dropped:
        print(f"[zero-probe] 忽略 checkpoint 里的未知配置字段：{dropped}")
    saved["llm_path"] = args.llm
    saved["encoder_path"] = args.encoder
    # 允许命令行强制覆盖（E 轨 checkpoint 靠这个对齐压缩配置）
    if args.compress_latent is not None:
        saved["compress_latent"] = args.compress_latent
    if args.compress_method:
        saved["compress_method"] = args.compress_method
    cfg = Stage1Config(**saved)

    model = Stage1Aligner(cfg)
    model.projection.load_state_dict(blob["projection"])

    # E 轨 checkpoint 必须把压缩器一起装回去，否则 build_batch 的 token 数对不上。
    if cfg.compress_latent:
        if "compression" not in blob:
            print(f"[fail] 配置要求 compress_latent={cfg.compress_latent}，"
                  f"但 checkpoint 里没有 compression 权重")
            return 2
        model.compression.load_state_dict(blob["compression"])
        print(f"[zero-probe] 已装回压缩器 {blob.get('compression_spec')}")

    device = torch.device(args.device)
    model.encoder.to(device)
    model.llm.to(device)
    model.projection.to(device)
    if model.compression is not None:
        model.compression.to(device)
    model.eval()

    print(f"[zero-probe] {len(samples)} 条样本；投影 {model.projection.describe()}")
    real = evaluate(model, samples, device, zero_encoder=False)
    print(f"[zero-probe] 真实编码器 : loss={real['loss']:.4f}  tok_acc={real['token_accuracy']}")
    zero = evaluate(model, samples, device, zero_encoder=True)
    print(f"[zero-probe] 置零编码器 : loss={zero['loss']:.4f}  tok_acc={zero['token_accuracy']}")

    d_loss = zero["loss"] - real["loss"]
    acc_drop = ((real["token_accuracy"] or 0) - (zero["token_accuracy"] or 0))
    if d_loss > args.pass_delta:
        verdict = "PASS"
    elif d_loss > args.weak_delta:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"\n[zero-probe] Δloss = +{d_loss:.4f}   Δtoken_acc = -{acc_drop:.4f}   -> {verdict}")
    print("            ⚠️ 判据提醒：Δloss 是逐 token 均值，答案越准越长它越低。")
    print("               最终以 scripts/31_stage1_rouge_curve.py 的 ROUGE 为准。")
    if verdict == "FAIL":
        print("           置零几乎没影响 = 模型没在用序列（投影塌缩 / 抄文本），回去查。")

    report = {"checkpoint": str(args.ckpt), "data": str(args.data), "n": len(samples),
              "real": real, "zeroed": zero, "delta_loss": d_loss,
              "delta_token_accuracy": acc_drop, "verdict": verdict,
              "compress_latent": cfg.compress_latent, "compress_method": cfg.compress_method,
              "criterion": "PLAN §3 P1：置零编码器后对齐指标必须明显下降"
                           "（Δloss 仅作参考，最终判据见 ROUGE 曲线）"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[zero-probe] 报告 -> {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
