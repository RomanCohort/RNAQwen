"""P2 入口：Stage 2 指令微调（LoRA）。

原版（arXiv 2411.08900v1 §A）：10 epochs · warmup 200 · weight_decay 0.05 · **8B 全参** · 2×A100 40GB。
本机：单卡 8060S，全参 8B 不现实 → **LoRA**（PLAN §3 P2 已预判的偏离，见 reports/）。

模板（原版 §2.1）：Stage 1 的 prompt 改成 Llama-3 风格 —— 真问题进 prompt，答案算 loss：

    [投影后的 RNA 表征] + "Q: <RNA></RNA>\\n" + "###Human: <q>\\n###Assistant:" + <a>
     不计 loss（软提示）                           不计 loss（prompt）            计 loss

**不复用** `alignment.build_batch`（那是 Stage 1 模板、且问题 token 也会计 loss）；
组装逻辑在 `rnaqwen.models.finetune.encode_samples`。理由见该模块 docstring。

跑法（数据一到位就能直接跑）：

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"
    & C:\\ana\\envs\\comfyui\\python.exe scripts/23_train_stage2.py `
        --data data/processed/rnaqa_rnacentral.jsonl `
        --llm models/Qwen3-8B --encoder models/rnafm `
        --projection checkpoints/stage1/projection_final.pt `
        --preset attn_mlp --out checkpoints/stage2 `
        --epochs 10 --warmup-steps 200 --weight-decay 0.05

消融（四条 preset，只换 LoRA 容量，别的都不动）：

    --preset attn          q,k,v,o          r=16 alpha=32   ≈15.3 M (0.187%)
    --preset attn_mlp      q,k,v,o,+MLP     r=16 alpha=32   ≈43.6 M (0.533%)   ← 默认
    --preset attn_mlp_r8   q,k,v,o,+MLP     r= 8 alpha=16   ≈21.8 M (0.266%)
    --preset attn_mlp_r64  q,k,v,o,+MLP     r=64 alpha=128  ≈174.6 M (2.132%)

数据子集筛选（按 qa[].variant / qa[].kind）：

    --variant ais  --kind factoid          # AIS 变体、factoid 问题
    --variant dc   --kind closed           # D&C 变体、closed 问题
    --variant both --kind both             # 不过滤（默认）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— 必须最早：给这个残废 torch 补 torch.distributed

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.paths import default_encoder, describe_encoder  # noqa: E402
from rnaqwen.models.finetune import (  # noqa: E402
    DEFAULT_PRESET,
    LORA_PRESETS,
    TEMPLATE_LLAMA3,
    TEMPLATE_QWEN,
    TEMPLATE_STAGE1,
    Stage2Config,
    UnknownPresetError,
    frozen_backbone_params,
    preset_budget,
    resolve_preset,
    train_stage2,
)

BOTH = {"both", "", "all", "*"}


def split_multi(value: str) -> list[str]:
    """`ais,dc` -> ['ais','dc']；`both` -> []（表示不过滤）。"""
    v = (value or "").strip().lower()
    if v in BOTH:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def filter_records(rows: list[dict], variants: list[str], kinds: list[str]) -> tuple[list[dict], dict]:
    """按 qa[].variant / kind 筛 QA 对。返回 (records, stats)。"""
    stats = {"records_in": len(rows), "qa_in": 0, "qa_kept": 0,
             "variant_hist": {}, "kind_hist": {}, "kept_variant_hist": {},
             "kept_kind_hist": {}, "records_kept": 0}
    out: list[dict] = []
    for r in rows:
        seq = r.get("sequence") or ""
        if not seq:
            continue
        kept = []
        for qa in r.get("qa", []) or []:
            stats["qa_in"] += 1
            v = str(qa.get("variant", "")).lower()
            k = str(qa.get("kind", "")).lower()
            stats["variant_hist"][v] = stats["variant_hist"].get(v, 0) + 1
            stats["kind_hist"][k] = stats["kind_hist"].get(k, 0) + 1
            if variants and v not in variants:
                continue
            if kinds and k not in kinds:
                continue
            if not (qa.get("question") or "").strip() or not (qa.get("answer") or "").strip():
                continue
            kept.append(qa)
            stats["kept_variant_hist"][v] = stats["kept_variant_hist"].get(v, 0) + 1
            stats["kept_kind_hist"][k] = stats["kept_kind_hist"].get(k, 0) + 1
        if kept:
            rec = dict(r)
            rec["qa"] = kept
            out.append(rec)
            stats["qa_kept"] += len(kept)
    stats["records_kept"] = len(out)
    return out, stats


def default_data() -> Path:
    """数据到位就用 rnacentral 那份，否则退回 50 条的 dry-run 集。"""
    full = ROOT / "data" / "processed" / "rnaqa_rnacentral.jsonl"
    return full if full.exists() else ROOT / "data" / "processed" / "rnaqa_50.jsonl"


def preset_type(value: str) -> str:
    """CLI 上就校验 preset 名，报错时把可选项说清楚（未知 preset 必须报错退出）。"""
    try:
        resolve_preset(value)
    except UnknownPresetError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2 指令微调（LoRA）")
    ap.add_argument("--data", type=Path, default=None,
                    help="含 sequence + qa 的 jsonl（默认 rnaqa_rnacentral.jsonl，缺则 rnaqa_50.jsonl）")
    ap.add_argument("--llm", default="models/Qwen3-8B")
    ap.add_argument("--encoder", default=default_encoder(),
                    help="不填就走 E 轨默认（ERNIE-RNA）。R 轨复现请显式给 models/rnafm")
    ap.add_argument("--projection", default="", help="Stage 1 产出的 projection_*.pt")
    # E 轨：Stage 1 用了压缩就必须在这里给同样的值（压缩器权重从 checkpoint 里取）。
    # 留空而 checkpoint 带压缩器时会直接报错，防止 R/E 两轨混用。
    ap.add_argument("--compress-latent", type=int, default=None,
                    help="E 轨：Stage 1 用的 RNA token 压缩数（不填 = 复现轨）")
    ap.add_argument("--compress-method", default="resampler",
                    choices=["resampler", "strided", "meanpool"])
    ap.add_argument("--out", default="checkpoints/stage2")
    # ---- SOS-LoRA（E 轨变体）----
    # 论文 arXiv 2607.16252v1 §4 的原值：r_tot=8, K=4, gamma_max=2.5。
    # 但 r_tot 我们**沿用 --lora-r**（默认 16）—— 要和标准 LoRA 同预算对比，
    # 唯一变量才是「单一子空间 vs K 个正交专家」。照抄 8 会让预算也变，就没法归因。
    ap.add_argument("--adapter", default="lora", choices=["lora", "sos"],
                    help="lora=标准 LoRA（走 peft）；sos=SOS-LoRA（自实现，可合并）")
    ap.add_argument("--sos-k", type=int, default=4, help="专家数 K，r_tot 必须能被它整除")
    ap.add_argument("--sos-gamma-max", type=float, default=2.5, help="多尺度上界 γ_max")
    ap.add_argument("--sos-ortho-coef", type=float, default=1.0, help="跨专家正交正则权重")

    ap.add_argument("--preset", default=DEFAULT_PRESET, type=preset_type,
                    help="LoRA 预设；只覆盖 targets/r/alpha。可用：" + "/".join(sorted(LORA_PRESETS)))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=None, help="显式覆盖预设的 r")
    ap.add_argument("--lora-alpha", type=int, default=None, help="显式覆盖预设的 alpha")
    ap.add_argument("--lora-targets", default=None,
                    help="逗号分隔，显式覆盖预设的 target_modules")
    ap.add_argument("--use-rslora", action="store_true",
                    help="rsLoRA：缩放 α/√r 而非 α/r（r≥16 更稳）")

    ap.add_argument("--variant", default="both", help="ais | dc | both（逗号可多选）")
    # 注意：这里是 qa[].kind 的**原始值**（open_ended / closed），不是 open/closed。
    # 写 open 会筛出 0 条并静默地在后面才报错。
    ap.add_argument("--kind", default="both",
                    help="open_ended | closed | both（逗号可多选；也可直接用 qa[].kind 的自由标签）")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条记录（0=全部）")

    ap.add_argument("--template", default=TEMPLATE_LLAMA3,
                    choices=[TEMPLATE_LLAMA3, TEMPLATE_QWEN, TEMPLATE_STAGE1])
    ap.add_argument("--max-nt", type=int, default=1024)
    ap.add_argument("--max-prompt-tokens", type=int, default=256)
    ap.add_argument("--max-answer-tokens", type=int, default=512)
    ap.add_argument("--no-eos", action="store_true", help="答案尾部不补 EOS")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="不填就自动选 cuda（没有则 cpu）")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做数据筛选 + 配置/参数预算，不加载模型")
    args = ap.parse_args(argv)

    data_path = args.data or default_data()
    if not Path(data_path).exists():
        print(f"[fail] 数据不存在：{data_path}")
        return 1

    rows = [r.as_dict() for r in read_jsonl(data_path)]
    if args.limit:
        rows = rows[: args.limit]
    variants, kinds = split_multi(args.variant), split_multi(args.kind)
    keep, stats = filter_records(rows, variants, kinds)
    print(f"[stage2] 数据 {data_path}")
    print(f"[stage2] 记录 {stats['records_kept']}/{stats['records_in']}；"
          f"QA {stats['qa_kept']}/{stats['qa_in']}（variant={args.variant} kind={args.kind}）")
    print(f"[stage2] variant 分布 {stats['variant_hist']} -> 保留 {stats['kept_variant_hist']}")
    print(f"[stage2] kind    分布 {stats['kind_hist']} -> 保留 {stats['kept_kind_hist']}")
    if not keep:
        print("[fail] 筛选后没有样本")
        return 1

    lora_kwargs: dict = {"preset": args.preset, "lora_dropout": args.lora_dropout,
                         "use_rslora": args.use_rslora}
    if args.lora_r is not None:
        lora_kwargs["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        lora_kwargs["lora_alpha"] = args.lora_alpha
    if args.lora_targets is not None:
        lora_kwargs["lora_targets"] = tuple(
            t.strip() for t in args.lora_targets.split(",") if t.strip())

    try:
        cfg = Stage2Config(
            llm_path=args.llm, encoder_path=args.encoder,
            projection_path=args.projection, out_dir=args.out,
            adapter=args.adapter, sos_k=args.sos_k,
            sos_gamma_max=args.sos_gamma_max,
            sos_ortho_coef=args.sos_ortho_coef,
            compress_latent=args.compress_latent, compress_method=args.compress_method,
            epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
            lr=args.lr, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm, scheduler=args.scheduler,
            template=args.template, max_nt=args.max_nt,
            max_prompt_tokens=args.max_prompt_tokens,
            max_answer_tokens=args.max_answer_tokens,
            append_eos=not args.no_eos, seed=args.seed,
            log_every=args.log_every, save_every=args.save_every,
            grad_checkpointing=not args.no_grad_checkpointing,
            **lora_kwargs,
        )
    except UnknownPresetError as e:
        print(f"[fail] {e}")
        return 2

    frozen = frozen_backbone_params()
    n_lora = cfg.lora_param_estimate()
    budget = {"frozen_backbone_params": frozen, "lora_param_estimate": n_lora,
              "lora_ratio_pct": 100.0 * n_lora / frozen,
              "preset_table": {p: preset_budget(p) for p in LORA_PRESETS}}
    print(f"[stage2] {cfg.describe()}")
    print(f"[stage2] 参数预算：LoRA {n_lora/1e6:.1f} M / 冻结底座 {frozen/1e9:.2f} B "
          f"= {budget['lora_ratio_pct']:.3f}%")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": cfg.as_dict(), "data": str(data_path),
                     "filter": {"variant": args.variant, "kind": args.kind,
                                "limit": args.limit},
                     "data_stats": stats, "budget": budget,
                     "n_records": len(keep), "n_samples": stats["qa_kept"]}

    if args.dry_run:
        summary["dry_run"] = True
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stage2] dry-run：未加载模型；摘要 -> {out_dir / 'run_summary.json'}")
        return 0

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    result = train_stage2(keep, cfg, device=device)
    summary["result"] = result
    summary["trainable_params"] = {
        "lora_params_peft": result.get("lora_params_peft"),
        "projection_params": result.get("projection_params"),
        "trainable_params_total": result.get("trainable_params_total"),
        "n_lora_modules": result.get("n_lora_modules"),
        "frozen_params": result.get("frozen_params"),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage2] 摘要 -> {out_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
