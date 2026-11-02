"""用**论文的口径**回答"Stage 1 到底有没有收敛"。

为什么必须做这个
----------------
训到一半的监控给了两条曲线：
    Δloss(置零编码器)  峰值 0.273 @ step 1600 → 衰减到 0.047 @ step 13800（只剩 17%）
    val_loss            最优 1.365 @ step 12800，但 6,200 步之后基本躺平

看起来是"编码器被抛弃 + 过拟合"。但**光看 loss 下不了结论** —— 而且原版给了可比数字：

    原版消融表的中间那一行（Modality Fusion = 只训投影 = Stage 1 的产物）
        AIS: ROUGE-1 = 0.2239
        D&C: ROUGE-1 = **0.0922**      ← 基本等于没学会

所以正确做法是**用同一把尺子量我们的 Stage-1-only 模型**：

    对每个 checkpoint，在 test 集上生成描述 → 算 ROUGE → 得到「ROUGE vs 训练步数」曲线

这条曲线一次回答三个问题：
  1. 早停点在哪（ROUGE 峰值对应的 step）
  2. 我们的 Modality Fusion 是多少，能不能对上原版的 0.0922
  3. Stage 1 到底在不在学（ROUGE 是否与 Δloss 曲线同形 → 两条独立证据互相印证）

**推理时的输入与 Stage 1 训练时逐字一致**（原版 §2.1，Q 留空）：
    [ "Q: <RNA>" | proj(rna) | "</RNA>\\nA: <Description>" ]  → 让模型接着写

跑法：
    python scripts/31_stage1_rouge_curve.py \
        --ckpt-dir checkpoints/stage1_run1 --eval data/evalset/test.jsonl \
        --compress-latent 64 --n-samples 64 --max-new-tokens 160 \
        --out reports/stage1_rouge_curve.json
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

import torch  # noqa: E402

from rnaqwen.eval.rouge import rouge_scores  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402
from rnaqwen.paths import default_encoder, describe_encoder  # noqa: E402


def step_of(path: Path) -> int:
    m = re.search(r"step(\d+)", path.name)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--eval", type=Path, default=ROOT / "data/evalset/test.jsonl")
    ap.add_argument("--llm", default=str(ROOT / "models/Qwen3-8B"))
    ap.add_argument("--encoder", default=str(ROOT / default_encoder()),
                    help="不填就走 E 轨默认（ERNIE-RNA）")
    ap.add_argument("--reference", default="reference_dc", choices=["reference_dc", "reference_ais"])
    ap.add_argument("--compress-latent", type=int, default=64)
    ap.add_argument("--compress-method", default="resampler")
    ap.add_argument("--n-samples", type=int, default=64, help="test 集里取多少条（0=全部）")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--every", type=int, default=1, help="每 N 个 checkpoint 测一个")
    ap.add_argument("--only", default=None, help="只测这些 step，逗号分隔")
    ap.add_argument("--resume", action="store_true",
                    help="读已有的 --out，跳过测过的 checkpoint（这个环境会杀进程）")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/stage1_rouge_curve.json")
    ap.add_argument("--pred-out", type=Path, default=None, help="把生成结果也存一份（调试用）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.eval.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n_samples:
        rows = rows[: args.n_samples]
    refs = [r[args.reference] for r in rows]
    print(f"[curve] test 集 {len(rows)} 条；参考字段 {args.reference}")

    ckpts = sorted(args.ckpt_dir.glob("projection_step*.pt"), key=step_of)
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        ckpts = [c for c in ckpts if step_of(c) in want]
    else:
        ckpts = ckpts[:: args.every]
    if not ckpts:
        print("[fail] 没找到 checkpoint")
        return 1

    # ---- 断点续跑：每测完一个 checkpoint 就落盘，重跑时跳过已完成的 ----
    # 这个环境会莫名其妙杀进程（实测 4 次），5 小时的活必须能接上。
    done: dict[int, dict] = {}
    if args.resume and args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            done = {c["step"]: c for c in prev.get("curve", [])}
            print(f"[curve] 续跑：已有 {len(done)} 个 checkpoint 的结果 -> {args.out}")
        except Exception as exc:  # noqa: BLE001
            print(f"[curve] ⚠️ 旧结果读不出来（{type(exc).__name__}），从头开始")
    todo = [c for c in ckpts if step_of(c) not in done]
    print(f"[curve] {len(ckpts)} 个 checkpoint：{[step_of(c) for c in ckpts]}")
    print(f"[curve] 待测 {len(todo)} 个，跳过 {len(ckpts)-len(todo)} 个")

    device = torch.device("cuda")
    cfg = Stage1Config(llm_path=args.llm, encoder_path=args.encoder,
                       grad_checkpointing=False,
                       compress_latent=args.compress_latent,
                       compress_method=args.compress_method)
    t0 = time.time()
    model = Stage1Aligner(cfg)
    model.encoder.to(device)
    model.llm.to(device)
    for mod in model.trainable_modules():
        mod.to(device)
        mod.eval()
    from rnaqwen.models.templates import RNA_PREFIX, STAGE1_SUFFIX
    prefix_ids = model.tokenizer(RNA_PREFIX, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    suffix_ids = model.tokenizer(STAGE1_SUFFIX, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    print(f"[curve] 模型就绪 {time.time()-t0:.1f}s；compression={args.compress_latent}")
    print(f"[curve] 模板：{RNA_PREFIX!r} + 软提示 + {STAGE1_SUFFIX!r}")

    curve = list(done.values())
    for ci, ck in enumerate(todo):
        blob = torch.load(ck, map_location="cpu")
        model.projection.load_state_dict(blob["projection"])
        if model.compression is not None and "compression" in blob:
            model.compression.load_state_dict(blob["compression"])
        elif model.compression is not None:
            print(f"[curve] ⚠️ {ck.name} 里没有 compression 键 —— 压缩器是随机的，结果不可信")

        preds = []
        t1 = time.time()
        with torch.no_grad():
            for r in rows:
                enc = model.encoder([r["sequence"]])
                hidden, mask = enc.hidden.to(device), enc.mask.to(device)
                if model.compression is not None:
                    hidden, mask = model.compression(hidden, mask)
                soft = model.projection(hidden)
                emb = model.llm.get_input_embeddings()
                inputs = torch.cat([emb(prefix_ids), soft, emb(suffix_ids)], dim=1)
                attn = torch.cat([
                    torch.ones((1, prefix_ids.shape[1]), dtype=mask.dtype, device=device),
                    mask,
                    torch.ones((1, suffix_ids.shape[1]), dtype=mask.dtype, device=device),
                ], dim=1)
                gen = model.llm.generate(inputs_embeds=inputs, attention_mask=attn,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False,
                                         pad_token_id=model.tokenizer.pad_token_id
                                         or model.tokenizer.eos_token_id)
                preds.append(model.tokenizer.decode(gen[0], skip_special_tokens=True).strip())

        sc = rouge_scores(preds, refs)
        dt = time.time() - t1
        rec = {"step": step_of(ck), "checkpoint": ck.name,
               "rouge1": round(sc.rouge1, 4), "rouge2": round(sc.rouge2, 4), "rougeL": round(sc.rougeL, 4),
               "n": len(preds), "seconds": round(dt, 1),
               "sample_pred": preds[0][:220] if preds else ""}
        curve.append(rec)
        curve.sort(key=lambda c: c["step"])
        print(f"[curve] {ci+1}/{len(todo)} step={rec['step']:>6d}  "
              f"R-1={rec['rouge1']:.4f} R-2={rec['rouge2']:.4f} R-L={rec['rougeL']:.4f}  ({dt:.0f}s)", flush=True)
        if args.pred_out and ci == 0:
            args.pred_out.write_text("\n".join(preds), encoding="utf-8")
        # 每测完一个就落盘 —— 被杀进程时最多丢一个 checkpoint
        args.out.parent.mkdir(parents=True, exist_ok=True)
        best_now = max(curve, key=lambda c: c["rouge1"])
        args.out.write_text(json.dumps({
            "eval": str(args.eval), "n_samples": len(rows), "reference": args.reference,
            "compress_latent": args.compress_latent, "max_new_tokens": args.max_new_tokens,
            "paper_reference_rouge1_dc": 0.0922,
            "curve": curve, "best": best_now, "complete": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    best = max(curve, key=lambda c: c["rouge1"])
    print(f"\n[curve] ROUGE-1 峰值 {best['rouge1']:.4f} @ step {best['step']}")
    print(f"[curve] 末点 step={curve[-1]['step']} R-1={curve[-1]['rouge1']:.4f}")
    print(f"[curve] 原版 Modality Fusion（D&C）参考值：ROUGE-1 = 0.0922")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "eval": str(args.eval), "n_samples": len(rows), "reference": args.reference,
        "compress_latent": args.compress_latent, "max_new_tokens": args.max_new_tokens,
        "paper_reference_rouge1_dc": 0.0922,
        "curve": curve, "best": best, "complete": len(curve) >= len(ckpts),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[curve] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
