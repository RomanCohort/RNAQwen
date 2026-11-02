"""配置一致性检查 —— 改了 VOCAB 之后，磁盘上的产物还是旧的吗？

背景：LATENT 从 12 改成 8，VOCAB 从 4096 变成 256。但 checkpoints/fsq_full/ 下的
产物是 VOCAB=4096 时生成的。两条路会踩坑：

  1) `--init-code-emb` 载入内容初始化码表 -> 形状 (4096,4096) vs 现在的 (256,4096)
     -> assert 直接抛（好在有 assert，没有静默）
  2) `FSQ_CKPT` 载入 FSQ.enc -> (12,768) vs 现在的 (8,768)
     -> 形状过滤后**一个张量都载不进来**，enc 其实是随机初始化

第 2 条是静默的：日志会打「载入 0 个张量」，但那行很容易被当成正常。
这个脚本把它变成一条硬检查。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

t0 = time.time()
PASS, FAIL = [], []


def check(name, cond, detail="", fatal=True):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else '**FAIL**'}  {name}" + (f"   {detail}" if detail else ""))


import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("routeC", ROOT / "scripts/50_routeC_e2e.py")
routeC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routeC)

LATENT, LEVELS = routeC.LATENT, routeC.LEVELS
VOCAB = routeC.VOCAB
D_MODEL = 4096
DIM = 768

print("=" * 78)
print(f"当前配置：LATENT={LATENT} LEVELS={LEVELS} VOCAB={VOCAB}")
print("=" * 78)

print()
print("--- 1. 内容初始化码表（--init-code-emb）---")
p = ROOT / "checkpoints/fsq_full/code_emb_contentinit.pt"
if not p.exists():
    check("code_emb_contentinit.pt 存在", False, "文件不存在")
else:
    ci = torch.load(p, map_location="cpu", weights_only=False)
    w = ci["code_emb"]
    ok = tuple(w.shape) == (VOCAB, D_MODEL)
    check(f"形状 == ({VOCAB}, {D_MODEL})", ok,
          f"实际 {tuple(w.shape)}"
          + ("" if ok else "  <- 陈旧产物：它是在 VOCAB=4096 时生成的，"
                            "现在 --init-code-emb 会 assert 失败（好在不是静默）"))

print()
print("--- 2. FSQ 权重（FSQ_CKPT，形状过滤后还能载入几个张量）---")
p2 = ROOT / "checkpoints/fsq_full/tokenizer.pt"
if not p2.exists():
    check("tokenizer.pt 存在", False, "文件不存在")
else:
    ck = torch.load(p2, map_location="cpu", weights_only=False)
    st = ck["state"]
    ck_lat = st["enc.weight"].shape[0]
    check(f"检查点 latent == 当前 LATENT({LATENT})", ck_lat == LATENT,
          f"检查点 latent={ck_lat}"
          + ("" if ck_lat == LATENT else
             "  <- 陈旧：形状过滤后 enc 一个张量都载不进来，"
             "FSQ.enc 实际是随机初始化（日志只打「载入 0 个张量」，容易被忽略）"))

print()
print("--- 3. 码序列缓存（48_ 产出，供 --codes 用）---")
p3 = ROOT / "checkpoints/fsq_full/codes.pt"
if not p3.exists():
    check("codes.pt 存在", False)
else:
    cd = torch.load(p3, map_location="cpu", weights_only=False)
    check(f"codes.pt vocab == 当前 VOCAB({VOCAB})", cd["vocab"] == VOCAB,
          f"检查点 vocab={cd['vocab']}"
          + ("" if cd["vocab"] == VOCAB else "  <- 陈旧，码 id 超出 256 会越界"))

print()
print("--- 4. 隐藏态缓存的维度对得上吗 ---")
p4 = ROOT / "data/processed/_fsq_full_pos.pt"
if not p4.exists():
    check("_fsq_full_pos.pt 存在", False)
else:
    d = torch.load(p4, map_location="cpu", weights_only=False)
    x = d["x"]
    check(f"隐藏态维度 == {DIM}", x.shape[1] == DIM, f"实际 {x.shape[1]}")
    check("lens.sum() == 位置数", int(d["lens"].sum()) == x.shape[0],
          f"{int(d['lens'].sum())} vs {x.shape[0]}")

print()
print("--- 5. 分桶后每个桶的长度跨度（跨度大会让 token 预算失准）---")
if p4.exists():
    lens = d["lens"]
    order = torch.argsort(lens)
    b = 256
    spans = []
    for i in range(0, len(order), b):
        seg = lens[order[i:i + b]]
        spans.append(int(seg.max() - seg.min()))
    check("桶内长度跨度 ≤ 全长的 20%", max(spans) <= 0.2 * int(lens.max()),
          f"最大跨度 {max(spans)}，全长 {int(lens.max())}，"
          f"最长码 {int(lens.max()) // routeC.POOL}")

print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败   用时 {time.time()-t0:.1f}s")
print("=" * 78)
if FAIL:
    print("需要处理的陈旧产物：")
    for f in FAIL:
        print("  -", f)
    raise SystemExit(1)
print("配置与产物一致")
