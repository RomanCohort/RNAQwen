"""把 RNA-FM 的**实际上限**逼出来（边界二分）。

背景：config 写 `max_position_embeddings = 1026`，但实测长度 1024 nt 的序列直接
`IndexError: index out of range in self` —— 说明真正生效的限制不是 config 那个数，
而是模型内部 `position_ids` 缓冲区的长度。

这个脚本不猜，直接逐长度试，报出**能跑通的最大 nt 数**。

跑法：
    python scripts/01b_probe_rnafm_limit.py --model models/rnafm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.paths import default_encoder  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / default_encoder()),
                    help="不填就走 E 轨默认（ERNIE-RNA）")
    ap.add_argument("--arch", default=None,
                    help="不填就按 config.json 的 model_type 自动识别")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    # 按 arch 解析，而不是硬编码 RnaFmModel —— 换 backbone 不用改探针。
    # 2026-10-09：从 RNA-FM 换到 ERNIE-RNA，两者的 config 都写 1026，
    # 但**不能假设真实上限一样**，所以这个探针必须重跑。
    import importlib

    from rnaqwen.models.encoder import read_model_type, resolve_arch

    arch = resolve_arch(args.model, args.arch)
    mod = importlib.import_module(arch.module)
    model_cls = getattr(mod, arch.model_cls)
    tok_cls = getattr(mod, "RnaTokenizer")

    print(f"arch={arch.name}  model_type={read_model_type(args.model)}  "
          f"（期望 hidden={arch.hidden_size}）")

    tok = tok_cls.from_pretrained(args.model)
    model = model_cls.from_pretrained(args.model).to(args.device).eval()
    cfg = model.config

    if int(cfg.hidden_size) != arch.hidden_size:
        print(f"[FAIL] hidden_size {cfg.hidden_size} != 期望 {arch.hidden_size}，arch 指错了")
        return 1

    print(f"config: max_position_embeddings={cfg.max_position_embeddings} "
          f"hidden={cfg.hidden_size} vocab={cfg.vocab_size}")
    print(f"tokenizer.model_max_length={getattr(tok, 'model_max_length', None)}")
    for name, buf in model.named_buffers():
        if "position" in name:
            print(f"buffer {name}: shape={tuple(buf.shape)} dtype={buf.dtype} "
                  f"last={int(buf.flatten()[-1]) if buf.numel() else 'n/a'}")
    print()

    ok_max = None
    print(f"{'nt':>6s} {'tokens':>7s}  result")
    for nt in [1000, 1020, 1021, 1022, 1023, 1024, 1025, 1030]:
        seq = ("ACGU" * (nt // 4 + 1))[:nt]
        enc = tok(seq, return_tensors="pt")
        n_tok = int(enc["input_ids"].shape[1])
        try:
            with torch.no_grad():
                out = model(**{k: v.to(args.device) for k, v in enc.items()})
            shape = tuple(out.last_hidden_state.shape)
            print(f"{nt:6d} {n_tok:7d}  OK   out={shape}")
            ok_max = nt
        except Exception as exc:  # noqa: BLE001
            print(f"{nt:6d} {n_tok:7d}  FAIL {type(exc).__name__}: {str(exc)[:70]}")
    print(f"\n-> 可用上限：{ok_max} nt（tokenizer 会额外加 BOS/EOS 两个特殊 token）")
    print(f"   config 里写的 max_position_embeddings={cfg.max_position_embeddings} 是**含特殊 token 的 token 数上限**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
