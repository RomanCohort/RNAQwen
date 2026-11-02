"""编码全部序列的 ERNIE-RNA 隐藏态 —— memmap 版。

上一版把隐藏态累积在内存里最后 torch.cat：
  149,598 条 x 平均约 330 nt = 4,940 万位置 x 768 x 2 字节 = 约 76 GB
  实测跑到 6.4%(9,616 条)就吃了 6 GB，按比例全量要 94 GB —— 必然 OOM。

改成：
  1. 先扫一遍算总位置数
  2. np.memmap 预分配 (total, 768) fp16
  3. 分批编码，直接写进 memmap，内存占用与总量无关
  4. 索引（uids/lens/offsets）单独存一个小文件

ERNIE-RNA 每条序列加了 [CLS] 和 [SEP]：encoder 位置数 = 序列长度 + 2。
这个偏移不处理会让所有切片整体错位一格 —— 之前踩过一次。
uid 必须与 58_ 一致：md5(sequence)[:16] 加前缀 seq:。用 hash() 的话每个进程都不同。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SRC = ROOT / "data/processed/pretrain_rna.jsonl"
MM = ROOT / "data/processed/_fsq_all_hidden.f16"
IDX = ROOT / "data/processed/_fsq_all_index.pt"
BATCH = 16
DIM = 768


def main() -> int:
    seqs, uids, lens = [], [], []
    seen = set()
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = (r.get("seq") or "").upper()
            if not s or s in seen:
                continue
            seen.add(s)
            seqs.append(s)
            uids.append(r.get("uid") or
                        "seq:" + hashlib.md5(s.encode()).hexdigest()[:16])
            lens.append(len(s))
    total = sum(lens)
    print(f"[data] {len(seqs):,} 条唯一序列   总位置 {total:,}   "
          f"memmap 需要 {total*DIM*2/1024**3:.2f} GB", flush=True)

    mm = np.memmap(MM, dtype=np.float16, mode="w+", shape=(total, DIM))
    print(f"[mmap] {MM}", flush=True)

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    t0 = time.time()
    pos = 0
    bad = 0
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            chunk = seqs[i:i + BATCH]
            o = enc(chunk)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, s in enumerate(chunk):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(s) + 2:
                    hh = hh[1:-1]
                if hh.shape[0] != len(s):
                    bad += 1
                    continue
                mm[pos:pos + hh.shape[0]] = hh.half().cpu().numpy()
                pos += hh.shape[0]
            del o, h, mk
            if (i // BATCH) % 40 == 0:
                done = i + len(chunk)
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"  [enc] {done:,}/{len(seqs):,}  {rate:.1f}/s  "
                      f"位置 {pos:,}/{total:,}  "
                      f"ETA {(len(seqs)-done)/max(rate,1e-9)/3600:.1f} h", flush=True)
            if (i // BATCH) % 400 == 0:
                mm.flush()

    mm.flush()
    print(f"\n[out] 写入位置 {pos:,} / 预期 {total:,}   丢弃 {bad} 条", flush=True)
    lens_t = torch.tensor(lens)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), lens_t]), 0)[:-1]
    torch.save(dict(lens=lens_t, offsets=offsets, uids=uids,
                    n_pos=pos, dim=DIM, mmap=str(MM)), IDX)
    print(f"[save] {IDX}")
    print(f"[save] {MM}  {MM.stat().st_size/1024**3:.2f} GB")
    print(f"用时 {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
