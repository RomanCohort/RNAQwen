"""PLAN §7.1 —— RNA-FM 探针。

目标（10 分钟那一步）：
  1. 拉 `multimolecule/rnafm`，喂一条序列
  2. 确认 **输出维度**（应为 (B, L, 640)）
  3. 确认 **上下文长度上限**（config: max_position_embeddings=1026 → 1024 nt + BOS/EOS）
  4. 落盘一份 JSON 记录，供后续 stage-1 投影层对齐使用

跑法（在本机 rnaqwen 环境）：
    python scripts/01_probe_rnafm.py --seq data/probe/example.fasta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen.env.shim  # noqa: E402,F401  —— 必须早于 multimolecule

import torch  # noqa: E402

MODEL_ID = "multimolecule/rnafm"

DEMO_SEQUENCES = {
    # 一段真实的 RNAcentral 风格序列（tRNA 类），以及长度压力测试
    "demo_small": "GGGCUAUCCCGGAGAUGGCCGAGUGGUUAAGGCGUUUGACUGUAGAUUGCUGCUUGAAAAGCAGUAGGUCCCUGUUCGGUC",
    "demo_yeast_trna_phe": (
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    ),
}


def load_sequences(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEMO_SEQUENCES)
    seqs: dict[str, str] = {}
    name = None
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name = line[1:].strip() or f"seq{len(seqs)}"
            buf = []
        else:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=None, help="FASTA 文件；不给就用内置 demo 序列")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "rnafm_probe.json")
    ap.add_argument("--push-length", type=int, default=1022, help="上下文压力测试长度（实测上限就是 1022 nt）")
    args = ap.parse_args()

    from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    report: dict = {"model_id": args.model, "device": device, "torch": torch.__version__}

    t0 = time.time()
    tok = RnaTokenizer.from_pretrained(args.model)
    model = RnaFmModel.from_pretrained(args.model).to(device).eval()
    report["load_seconds"] = round(time.time() - t0, 1)

    cfg = model.config
    report["config"] = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "model_type": cfg.model_type,
    }
    report["n_params"] = sum(p.numel() for p in model.parameters())
    report["tokenizer_class"] = type(tok).__name__
    report["vocab_actual_size"] = getattr(tok, "vocab_size", None)

    # ---- 单条序列 forward ----
    seqs = load_sequences(args.seq)
    report["sequences"] = []
    for name, seq in seqs.items():
        enc = tok(seq, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        last = out.last_hidden_state
        rec = {
            "name": name,
            "input_len_nt": len(seq),
            "input_ids_shape": list(enc["input_ids"].shape),
            "output_shape": list(last.shape),
            "output_dim": last.shape[-1],
            "dtype": str(last.dtype),
            "mean_norm": round(float(last.norm(dim=-1).mean()), 4),
        }
        report["sequences"].append(rec)
        print(f"[ok] {name:22s} nt={len(seq):5d} ids={tuple(enc['input_ids'].shape)} out={tuple(last.shape)}")
        if name == next(iter(seqs)):
            report["special_tokens"] = {
                "cls/bos": int(enc["input_ids"][0, 0]),
                "sep/eos": int(enc["input_ids"][0, -1]),
            }
            report["first_tokens"] = enc["input_ids"][0, :6].tolist()

    # ---- 上下文长度压力测试 ----
    limit = int(report["config"]["max_position_embeddings"])
    probe_len = args.push_length
    long_seq = ("ACGU" * ((probe_len // 4) + 1))[:probe_len]
    enc = tok(long_seq, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    report["context_probe"] = {
        "requested_nt": probe_len,
        "input_ids_shape": list(enc["input_ids"].shape),
        "output_shape": list(out.last_hidden_state.shape),
        "max_position_embeddings": limit,
        "usable_nt": limit - 2,
        "measured_usable_nt": 1022,
        "note": "config 写 max_position_embeddings=1026，但真正生效的 token 上限是 1024"
                "（1024 token 能过、1025 就 IndexError）—— 所以可用 nt 是 1022 而不是 1024。"
                "见 scripts/01b_probe_rnafm_limit.py。",
    }
    print(f"[ctx] requested {probe_len} nt -> ids {tuple(enc['input_ids'].shape)} "
          f"(config max_position_embeddings={limit}；**实测**可用 nt=1022 = 1024 token)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
