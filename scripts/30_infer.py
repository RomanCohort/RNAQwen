"""推理：把消融阶梯上三档配置各跑一遍，产出可直接喂给 04_eval.py 的预测文件。

PLAN §4 的阶梯（**复现出这条"越接越强"的阶梯，才是复现成功的真正判据**）：

    rna_sequence   裸文本输入：把序列当字符串塞进 prompt，不走编码器
    modality_fusion 只训投影：RNA-FM → 投影 → LLM，但没做 Stage2 指令微调
    rna_gpt        完整：投影 + Stage-2 LoRA

跑法：
    python scripts/30_infer.py --eval data/evalset/test.jsonl --mode rna_sequence --out preds/seq.jsonl
    python scripts/30_infer.py --eval data/evalset/test.jsonl --mode modality_fusion \
        --projection checkpoints/stage1/projection_final.pt --out preds/fusion.jsonl
    python scripts/30_infer.py --eval data/evalset/test.jsonl --mode rna_gpt \
        --projection checkpoints/stage1/projection_final.pt --lora checkpoints/stage2/final/lora \
        --out preds/full.jsonl
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

from rnaqwen.paths import default_encoder, describe_encoder  # noqa: E402

import torch  # noqa: E402

TASK_PROMPT = (
    "Describe the following RNA molecule: its type, function, mechanism and biological role. "
    "Write 3-5 sentences."
)
SEQ_PROMPT = (
    "The RNA sequence is: {seq}\n\n"
    "Describe this RNA molecule: its type, function, mechanism and biological role. "
    "Write 3-5 sentences."
)


def load_eval(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=["rna_sequence", "modality_fusion", "rna_gpt"])
    ap.add_argument("--llm", default="models/Qwen3-8B")
    ap.add_argument("--encoder", default=default_encoder(),
                    help="不填就走 E 轨默认（ERNIE-RNA）。R 轨复现请显式给 models/rnafm")
    ap.add_argument("--projection", type=Path, default=None)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--reference", default="reference_dc", choices=["reference_dc", "reference_ais"],
                    help="用哪个字段当参考文本（D&C 或 AIS 口径）")
    ap.add_argument("--question", default=(
        "Describe this RNA molecule: its type, function, mechanism and biological role. "
        "Write 3-5 sentences."))
    ap.add_argument("--template", default="llama3", choices=["llama3", "qwen", "stage1"],
                    help="**必须与训练时一致**（Stage 2 默认 llama3，照抄原论文）；见 "
                         "src/rnaqwen/models/templates.py")
    ap.add_argument("--compress-latent", type=int, default=None,
                    help="E 轨：RNA token 压缩到多少（None=复现轨原样）。要与训练时一致")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_eval(args.eval, args.limit)
    if not rows:
        print("[fail] 评测集为空")
        return 1
    print(f"[infer] mode={args.mode}  {len(rows)} 条样本 <- {args.eval}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16

    t0 = time.time()
    if args.mode == "rna_sequence":
        model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=dtype).to(args.device).eval()
        encoder = projection = None
    else:
        from rnaqwen.models.alignment import Stage1Aligner, Stage1Config
        from rnaqwen.models.encoder import RnaFmEncoder
        from rnaqwen.models.projection import LinearProjection

        cfg = Stage1Config(llm_path=args.llm, encoder_path=args.encoder)
        encoder = RnaFmEncoder(args.encoder, dtype=dtype).to(args.device)
        model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=dtype).to(args.device).eval()
        llm_hidden = int(model.config.hidden_size)
        projection = LinearProjection(encoder.hidden_size, llm_hidden).to(args.device, dtype)
        blob = None
        if args.projection:
            blob = torch.load(args.projection, map_location="cpu")
            projection.load_state_dict(blob["projection"])
            print(f"[infer] 载入 Stage1 投影 <- {args.projection} (step {blob.get('step')})")
        projection.eval()

        # E 轨：token 压缩（默认不开 = 复现轨）。必须与训练时的配置一致。
        #
        # ⚠️ 这里以前只 new 了一个 RnaTokenCompressor 就完事，**权重从来没载过** ——
        #    等于拿一个随机初始化的压缩器去推理，输出看着像样但全是错的（静默错误）。
        #    现在：配置从 checkpoint 里读（命令行可覆盖），权重必须从 blob 里 load。
        spec = (blob or {}).get("compression_spec") or {}
        n_latent = args.compress_latent or spec.get("n_latent")
        method = args.compress_method or spec.get("method") or "resampler"

        if n_latent and blob is None:
            raise SystemExit("[fail] 指定了 --compress-latent 但没给 --projection，"
                             "压缩器权重无处可取（随机初始化 = 结果无效）。")
        if n_latent and "compression" not in (blob or {}):
            raise SystemExit(f"[fail] --compress-latent={n_latent}，但 {args.projection} "
                             f"里没有 compression 权重 —— 这是 R 轨（未压缩）的 checkpoint。")
        if not n_latent and blob is not None and "compression" in blob:
            raise SystemExit(f"[fail] {args.projection} 是 E 轨（带压缩器）的产物，"
                             f"请加上 --compress-latent {spec.get('n_latent')}。")

        compressor = None
        if n_latent:
            from rnaqwen.models.compression import RnaTokenCompressor

            compressor = RnaTokenCompressor(
                encoder.hidden_size, int(n_latent), method=method).to(args.device, dtype)
            compressor.load_state_dict(blob["compression"])
            compressor.eval()
            print(f"[infer] E 轨：RNA token 压缩到 {n_latent}（method={method}，权重已载入）"
                  f"（**不可与复现轨分数并列汇报**）")

        if args.lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora)
            model = model.to(args.device).eval()
            print(f"[infer] 载入 Stage2 LoRA <- {args.lora}")
    print(f"[infer] 模型就绪，用时 {time.time()-t0:.1f}s")

    # ⚠️ 模板必须与**训练时**逐字一致（见 src/rnaqwen/models/templates.py）。
    #    原先这里用 Qwen 的 ChatML（apply_chat_template），而 Stage 2 训练用的是
    #    原版论文的 Llama-3 风格 `###Human: …\n###Assistant:` ——
    #    不一致会让推理直接落到**分布外**（不是掉点，是直接失效）。
    from rnaqwen.models.templates import (
        RNA_PREFIX,
        RNA_SUFFIX,
        format_prompt,
        sequence_as_text_input,
    )

    prefix_ids = tok(RNA_PREFIX, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
    suffix_ids = tok(RNA_SUFFIX, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
    print(f"[infer] template={args.template}  prefix={prefix_ids.shape[1]}tok suffix={suffix_ids.shape[1]}tok")

    def text_prompts(batch: list[dict]) -> list[str]:
        """rna_sequence 档：把序列当**字面文本**塞进同一个模板结构。"""
        return [sequence_as_text_input(args.question, r["sequence"], args.template) for r in batch]

    preds = []
    t1 = time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        if args.mode == "rna_sequence":
            enc = tok(text_prompts(batch), return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=args.temperature > 0,
                                     temperature=max(args.temperature, 1e-4),
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        else:
            # 模块融合：inputs = [ "Q: <RNA>" | proj(rna) | "</RNA>\n" | "###Human: …\n###Assistant:" ]
            with torch.no_grad():
                enc_out = encoder([r["sequence"] for r in batch])
                hidden = enc_out.hidden
                mask = enc_out.mask
                if compressor is not None:
                    hidden, mask = compressor(hidden, mask)
                rna_emb = projection(hidden)
            prompt_ids = tok([format_prompt(args.question, args.template) for _ in batch],
                             return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(args.device)
            emb = model.get_input_embeddings()
            pre = emb(prefix_ids).expand(rna_emb.shape[0], -1, -1)
            suf = emb(suffix_ids).expand(rna_emb.shape[0], -1, -1)
            post = emb(prompt_ids)
            inputs = torch.cat([pre, rna_emb, suf, post], dim=1)
            attn = torch.cat([
                torch.ones((rna_emb.shape[0], pre.shape[1]), dtype=enc_out.mask.dtype, device=args.device),
                enc_out.mask,
                torch.ones((rna_emb.shape[0], suf.shape[1]), dtype=enc_out.mask.dtype, device=args.device),
                torch.ones((rna_emb.shape[0], post.shape[1]), dtype=enc_out.mask.dtype, device=args.device),
            ], dim=1)
            with torch.no_grad():
                gen = model.generate(inputs_embeds=inputs, attention_mask=attn,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=args.temperature > 0,
                                     temperature=max(args.temperature, 1e-4),
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen, skip_special_tokens=True)

        for r, text in zip(batch, texts):
            preds.append({
                "uid": r.get("uid"),
                "mode": args.mode,
                "variant": args.reference.replace("reference_", ""),
                "prediction": text.strip(),
                "reference": r[args.reference],
                "n_nt": r.get("n_nt"),
                "organism": r.get("organism", ""),
            })
        print(f"  [{min(i+args.batch_size, len(rows))}/{len(rows)}] 用时 {time.time()-t1:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_pred = sum(1 for p in preds if p["prediction"].strip())
    print(f"\n[infer] 写出 {len(preds)} 条（非空 {n_pred}）-> {args.out}")
    if preds:
        print("[sample]", preds[0]["prediction"][:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
