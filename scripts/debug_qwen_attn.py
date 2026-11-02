"""诊断：Qwen3-8B 在本机 ROCm 上生成乱码，先定位是哪一层的问题。

现象（第一次 smoke test）：
    输入 "Name the four RNA nucleotides, one per word."
    输出 "matplotlib React React React React sys ..."   <- 完全不成句

可疑点排序：
  1. attention 后端（ROCm gfx1151 上 flash / mem-efficient 都标着 experimental）
  2. bfloat16 数值
  3. 权重本身（但 399 个张量都 load 成功，先怀疑前两个）

这个脚本按 (attn_impl, dtype) 做矩阵测试，同一段 prompt、贪心解码，
输出连不连贯一眼就能看出来。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = str(ROOT / "models" / "Qwen3-8B")
PROMPT = "Name the four RNA nucleotides, separated by spaces."

CASES = [
    ("sdpa", "bfloat16"),
    ("eager", "bfloat16"),
    ("eager", "float32"),
]


def run(attn: str, dtype_name: str) -> None:
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=dtype, attn_implementation=attn
        ).to("cuda").eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[{attn:6s}/{dtype_name:9s}] LOAD FAIL {type(exc).__name__}: {str(exc)[:120]}")
        return
    load_s = time.time() - t0

    msgs = [{"role": "user", "content": PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to("cuda")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{attn:6s}/{dtype_name:9s}] load={load_s:5.1f}s gen={time.time()-t1:5.1f}s")
    print(f"    -> {gen.strip()[:220]!r}")
    del model
    torch.cuda.empty_cache()


def main() -> int:
    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    for attn, dt in CASES:
        run(attn, dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
