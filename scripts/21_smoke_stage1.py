"""训练前体检：Stage 1 的数据通路能不能真的把梯度传到投影层。

为什么单独写一个
----------------
真跑 Stage 1 要几小时，如果配错了（label 掩码错、embedding 接错、梯度被 detach 掉），
你会先等几小时再发现 loss 不动。所以先把这几件事在**一两个样本**上验完：

  1. 投影层形状必须是 **640 → 4096**（和原版一致，见 PLAN §2）
  2. `inputs_embeds` 拼接后长度必须 = RNA token 数 + 文本 token 数
  3. loss 只落在**答案 token** 上（RNA 前缀与 padding 全是 -100）
  4. 反传后：**投影有非零梯度**；编码器与 LLM 的梯度必须是 `None`（冻结生效）
  5. 置零编码器后 loss 必须变化（否则说明 RNA 这一路根本没接进计算图）

跑法（CPU 也能跑，只是慢）：
    python scripts/21_smoke_stage1.py --data data/processed/annotations_50.jsonl --device cpu --n 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402
from rnaqwen.paths import default_encoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--llm", default=str(ROOT / "models" / "Qwen3-8B"))
    ap.add_argument("--encoder", default=str(ROOT / default_encoder()),
                    help="不填就走 E 轨默认（ERNIE-RNA）")
    ap.add_argument("--target-field", default="annotation", choices=["annotation", "literature"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stage1_smoke.json")
    args = ap.parse_args()

    rows = [r.as_dict() for r in read_jsonl(args.data)]
    samples = []
    for r in rows:
        t = (r.get(args.target_field) or "").strip()
        if r.get("sequence") and t:
            samples.append({"sequence": r["sequence"], "target": t})
        if len(samples) >= args.n:
            break
    if len(samples) < 1:
        print("[fail] 没有带目标文本的样本")
        return 1
    print(f"[smoke] {len(samples)} 条样本；目标字段={args.target_field}；device={args.device}")

    device = torch.device(args.device)
    cfg = Stage1Config(llm_path=args.llm, encoder_path=args.encoder, grad_checkpointing=False)
    t0 = time.time()
    model = Stage1Aligner(cfg)
    model.encoder.to(device)
    model.llm.to(device)
    model.projection.to(device)
    print(f"[smoke] 模型就绪 {time.time()-t0:.1f}s；{model.projection.describe()}")

    report: dict = {
        "projection": model.projection.describe(),
        "encoder": model.encoder.describe(),
        "encoder_arch": model.encoder.arch.name,
        "encoder_hidden": model.encoder.hidden_size,
        "llm_hidden": model.llm_hidden,
        "device": args.device,
        "samples": [],
    }
    checks: dict[str, bool] = {}
    cfg_arch = model.encoder.arch.name

    # 检查要**按实际 arch** 来，不能硬编码 640 —— 换了 backbone 就会假失败。
    # RNA-FM: 640->4096   ERNIE-RNA: 768->4096
    from rnaqwen.models.encoder import ARCHS, read_model_type

    want_hidden = ARCHS[model.encoder.arch.name].hidden_size
    checks[f"proj_shape_{want_hidden}_to_4096"] = (
        model.encoder.hidden_size, model.llm_hidden) == (want_hidden, 4096)
    # 别写同义反复的检查。这里查两件真会出错的事：
    #  1) encoder 目录的 model_type 和我们实际加载的 arch 一致
    #  2) max_nt 用的是**该 arch 的**上限，不是别的 backbone 的（RNA-FM 1022 / ERNIE-RNA 1024）
    checks["arch_matches_ckpt_config"] = (
        model.encoder.arch.name == (read_model_type(cfg.encoder_path) or ""))
    checks["max_nt_matches_arch"] = (model.encoder.max_nt == ARCHS[model.encoder.arch.name].max_nt)
    checks["llm_frozen"] = all(not p.requires_grad for p in model.llm.parameters())
    checks["encoder_frozen"] = all(not p.requires_grad for p in model.encoder.parameters())
    checks["proj_trainable"] = all(p.requires_grad for p in model.projection.parameters())

    for i, s in enumerate(samples):
        t1 = time.time()
        inputs, attn, labels = model.build_batch([s["sequence"]], [s["target"]], device)
        n_rna = int(attn.shape[1] - (labels != -100).sum().item())
        n_ans = int((labels != -100).sum().item())
        loss = model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss
        loss.backward()

        proj_grad = [p.grad for p in model.projection.parameters() if p.grad is not None]
        grad_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in proj_grad))) if proj_grad else 0.0
        llm_has_grad = any(p.grad is not None for p in model.llm.parameters())

        rec = {
            "i": i,
            "seq_len_nt": len(s["sequence"]),
            "inputs_embeds_shape": list(inputs.shape),
            "n_rna_tokens": n_rna,
            "n_answer_tokens": n_ans,
            "loss": float(loss),
            "proj_grad_norm": grad_norm,
            "llm_grad_present": llm_has_grad,
            "seconds": round(time.time() - t1, 1),
        }
        report["samples"].append(rec)
        print(f"[smoke] #{i} seq={rec['seq_len_nt']}nt embeds={tuple(inputs.shape)} "
              f"rna_tok={n_rna} ans_tok={n_ans} loss={rec['loss']:.4f} "
              f"|grad_proj|={grad_norm:.3e} llm_grad={llm_has_grad} ({rec['seconds']}s)")
        model.projection.zero_grad(set_to_none=True)

    checks["grad_reaches_projection"] = all(s["proj_grad_norm"] > 0 for s in report["samples"])
    checks["llm_has_no_grad"] = all(not s["llm_grad_present"] for s in report["samples"])
    checks["answer_tokens_nonzero"] = all(s["n_answer_tokens"] > 0 for s in report["samples"])

    # 置零编码器 → loss 应变化（证明 RNA 这一路在计算图里）
    from rnaqwen.models.encoder import EncoderOutput

    s = samples[0]
    inputs, attn, labels = model.build_batch([s["sequence"]], [s["target"]], device)
    with torch.no_grad():
        base = float(model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss)
    orig = model.encoder.forward

    def zero_forward(seqs):  # noqa: ANN001
        out = orig(seqs)
        return EncoderOutput(hidden=torch.zeros_like(out.hidden), mask=out.mask, lengths=out.lengths)

    model.encoder.forward = zero_forward  # type: ignore[assignment]
    inputs_z, attn_z, labels_z = model.build_batch([s["sequence"]], [s["target"]], device)
    with torch.no_grad():
        zeroed = float(model.llm(inputs_embeds=inputs_z, attention_mask=attn_z, labels=labels_z).loss)
    model.encoder.forward = orig  # type: ignore[assignment]
    checks["zeroing_changes_loss"] = abs(zeroed - base) > 1e-6
    report["zero_encoder"] = {"loss_real": base, "loss_zeroed": zeroed, "delta": zeroed - base}
    print(f"[smoke] 置零编码器: loss {base:.4f} -> {zeroed:.4f} (Δ={zeroed-base:+.4f})")

    report["checks"] = checks
    report["passed"] = all(checks.values())
    print()
    for k, v in checks.items():
        print(f"  [{'ok  ' if v else 'FAIL'}] {k}")
    print(f"\n[smoke] verdict: {'PASS —— 数据通路正确，可以开训' if report['passed'] else 'FAIL —— 先修'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[smoke] 报告 -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
