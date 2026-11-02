"""CPU equivalence check for the patched decode-path depthwise causal conv.

Why: MIOpen on gfx1151 raises miopenStatusInternalError for F.conv1d, and the
decode path (Qwen3_5MoeGatedDeltaNet -> self.causal_conv1d_update ->
torch_causal_conv1d_update) calls F.conv1d directly -- patching self.conv1d
alone only covers prefill. Training is prefill-only, so it never showed up.

    python scripts/_scratch/_test_conv1d_update.py

Expect max|d_out| ~1e-6 (fp32 rounding) and max|d_state| == 0.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402  must precede transformers/peft
import torch  # noqa: E402
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M  # noqa: E402
from rnaqwen.qwen36 import patch_conv1d  # noqa: E402


def run(fn, x, st, w, b):
    st = st.clone()
    return fn(x, st, w, b), st


def main() -> int:
    orig = M.torch_causal_conv1d_update
    patch_conv1d(M)
    mine = M.torch_causal_conv1d_update
    print("patched:", mine is not orig)
    if mine is orig:
        print("** patch_conv1d did not replace the decode path")
        return 1

    torch.manual_seed(0)
    worst = 0.0
    for C, K, S, L in [(6, 4, 3, 1), (6, 4, 3, 5), (8, 4, 3, 1),
                       (8, 4, 3, 9), (16, 4, 3, 64)]:
        x = torch.randn(2, C, L)
        st = torch.randn(2, C, S)
        w = torch.randn(C, K)
        b = torch.randn(C)
        a1, s1 = run(orig, x, st, w, b)
        a2, s2 = run(mine, x, st, w, b)
        d = (a1 - a2).abs().max().item()
        ds = (s1 - s2).abs().max().item()
        worst = max(worst, d, ds)
        print(f"  C={C:>3} K={K} state={S} L={L:>3}  "
              f"max|d_out| {d:.3e}  max|d_state| {ds:.3e}")
    ok = worst < 1e-5
    print("WORST", f"{worst:.3e}", "OK" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
