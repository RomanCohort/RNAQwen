"""全量 40 层装载 + 真实前向 / 反向 —— 35B 启动前的最后一道闸。

验四件事：
  1. 40 层全部装上，没有参数留在 meta
  2. input_ids 路径前向
  3. inputs_embeds 路径前向（route C 的注入方式）
  4. 反向：梯度能回到注入的码向量（码表和 FSQ 全靠这条回传）

内存要盯住 RSS 和系统空闲 —— 装载过程中读 mmap 页会常驻工作集，
必须靠 trim_working_set() 周期修剪，否则 40 层 × 1.6 GiB 会把机器压死。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import torch  # noqa: E402
from rnaqwen.qwen36 import load_qwen36_text, rss_gb, free_gb  # noqa: E402

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main() -> int:
    try:
        model, text, lm_head, stats = load_qwen36_text(mode="int4", smoke=-1, log=log)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"** 装载失败 {type(e).__name__}: {e}")
        return 1
    log(f"✅ 全量装载成功 {stats}")
    log(f"   装载后 RSS {rss_gb():.2f} GiB  空闲 {free_gb():.1f} GiB  "
        f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    dev = "cuda"
    model.eval()
    bad = 0

    # (a) input_ids 路径
    ids = torch.tensor([[248045, 8678, 198, 248046, 248045, 846, 198]], device=dev)
    with torch.no_grad():
        t = time.time()
        l1 = model(input_ids=ids).logits.float()
        log(f"(a) input_ids        logits {tuple(l1.shape)}  "
            f"值域 [{l1.min():.2f},{l1.max():.2f}]  有限 "
            f"{bool(torch.isfinite(l1).all())}  用时 {time.time()-t:.2f}s")
    if not bool(torch.isfinite(l1).all()):
        bad += 1

    # (b) inputs_embeds 路径 —— route C 就是这么注入的
    emb = model.model.language_model.embed_tokens(ids).detach()
    v = torch.randn(3, emb.shape[-1], device=dev, dtype=emb.dtype, requires_grad=True)
    inp = emb.clone()
    inp[0, 2:5] = v.to(inp.dtype)
    inp.requires_grad_(True)
    t = time.time()
    l2 = model(inputs_embeds=inp).logits.float()
    log(f"(b) inputs_embeds   logits {tuple(l2.shape)}  有限 "
        f"{bool(torch.isfinite(l2).all())}  用时 {time.time()-t:.2f}s")
    if not bool(torch.isfinite(l2).all()):
        bad += 1

    # (c) 反向
    t = time.time()
    l2.pow(2).mean().backward()
    g = v.grad
    ok_g = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
    log(f"(c) 反向            g_v 有限 {bool(torch.isfinite(g).all()) if g is not None else False}  "
        f"|g_v|max {float(g.abs().max()):.3e}  非零 {ok_g}  用时 {time.time()-t:.2f}s")
    if not ok_g:
        bad += 1

    log(f"   峰值 RSS {rss_gb():.2f} GiB   空闲 {free_gb():.1f} GiB   "
        f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    print()
    print("=" * 78)
    print("  ✅ 35B 全量装载 + 前向 + 反向全部通过" if bad == 0
          else f"  ❌ {bad} 项失败")
    print("=" * 78)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
