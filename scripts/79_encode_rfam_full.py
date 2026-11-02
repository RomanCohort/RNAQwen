"""把 all.fa 的**全部** RNA 编成隐藏态缓存（升级 2）。

为什么要全量
------------
现有的 _rfam_hidden.f16 只有 19,726 条，而 61_rfam_qa.py 里那批是**按家族平衡采样**
出来的（每家族 <= 40），目的是把「永远答最常见家族」的命中率压到 2% 以下 ——
原本 rRNA 占 51%，RF00177 一个家族就 43.4%。

但新数据集 (rnaqa_seqtask) 的答案全部从序列复算，**标签不再参与**，
家族平衡这个约束自动失效。所以 128,155 条有 Rfam 注释的、乃至 all.fa 全部的
145,201 条都可以用。该增加的是 RNA，不是标签。

| 量            | 值                                          |
|---------------|---------------------------------------------|
| all.fa        | 145,201 条   47,547,438 位置   最长 1022 均值 327 |
| 已编码        |  19,726 条    6,680,378 位置   9.6 GiB      |
| 全量编码      |               47,547,438 位置  68.0 GiB     |
| 还差          | 125,475 条    40,867,060 位置  58.5 GiB     |

和 73_ 的两处差别（都是为全量而改）
----------------------------------
1. uid 来自 all.fa 本身，不再从 rnaqa_rfam.jsonl 取 —— 目标集合变了。
2. **偏移量按「实际写入的条数」算。** 73_ 用预分配的 lens 算 offsets，
   如果某条因形状不符被丢弃（bad），写入位置会比 offsets 少，
   后面所有序列的隐藏态就整体错位 —— 而且不会报错。这里改成
   记录真正写进去的 (uid, len) 列表，再算 offsets。

用法：
    python scripts/79_encode_rfam_full.py --limit 400      # 冒烟
    python scripts/79_encode_rfam_full.py                  # 全量
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FA = ROOT / "data/external/rfam/all.fa"
MM = ROOT / "data/processed/_rfa_full_hidden.f16"
IDX = ROOT / "data/processed/_rfa_full_index.pt"
DIM = 768
MAX_NT = 1024


def key(u: str) -> str:
    return u[4:] if u.startswith("seq:") else u


def read_fa(p: Path) -> tuple[list[str], list[str]]:
    """顺序保序返回 (uids, seqs)。同一条序列只留一次（uid 就是 md5，天然去重）。"""
    uids, seqs = [], []
    cur, buf = None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    uids.append(cur)
                    seqs.append("".join(buf).upper())
                cur = key(line[1:].strip().split()[0])
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        uids.append(cur)
        seqs.append("".join(buf).upper())
    # 保序去重（理论上不会有重复，因为 uid = md5(seq)[:16]）
    seen, U, S = set(), [], []
    for u, s in zip(uids, seqs):
        if u in seen:
            continue
        seen.add(u)
        U.append(u)
        S.append(s)
    return U, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只编前 N 条（冒烟）")
    ap.add_argument("--out", default="")
    ap.add_argument("--idx", default="")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="写之前至少留这么多磁盘")
    args = ap.parse_args()

    mm_path = Path(args.out) if args.out else MM
    idx_path = Path(args.idx) if args.idx else IDX

    uids, seqs = read_fa(FA)
    print(f"[fa] {FA.name}  序列 {len(uids):,} 条（去重后）")

    # 长度过滤：50_ 的 MAX_NT=1024
    keep = [i for i, s in enumerate(seqs) if 4 <= len(s) <= MAX_NT]
    drop_long = sum(1 for s in seqs if len(s) > MAX_NT)
    drop_short = sum(1 for s in seqs if len(s) < 4)
    if drop_long or drop_short:
        print(f"[filter] 丢 {drop_long} 条 > {MAX_NT}，{drop_short} 条 < 4")
    uids = [uids[i] for i in keep]
    seqs = [seqs[i] for i in keep]

    if args.limit:
        uids, seqs = uids[:args.limit], seqs[:args.limit]
        print(f"[limit] 只处理前 {len(uids):,} 条")

    lens = [len(s) for s in seqs]
    total = sum(lens)
    need_gb = total * DIM * 2 / 1024 ** 3
    print(f"[data] {len(seqs):,} 条   总位置 {total:,}   长度 {min(lens)}~{max(lens)} "
          f"均值 {total/len(lens):.0f}")
    print(f"[data] memmap 需要 {need_gb:.2f} GiB")

    free_gb = shutil.disk_usage(mm_path.parent).free / 1024 ** 3
    print(f"[disk] {mm_path.parent} 空闲 {free_gb:.1f} GiB")
    if free_gb < need_gb + args.min_free_gb:
        print(f"** 空间不够：需要 {need_gb:.1f} + 余量 {args.min_free_gb} "
              f"> 空闲 {free_gb:.1f} **")
        return 2

    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(total, DIM))
    print(f"[mmap] {mm_path}")

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    t0 = time.time()
    pos = 0
    kept_u, kept_l = [], []
    bad: list[tuple[str, int, int]] = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
            uu = uids[i:i + args.batch]
            o = enc(chunk)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, s in enumerate(chunk):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(s) + 2:      # [CLS] ... [SEP]
                    hh = hh[1:-1]
                if hh.shape[0] != len(s):
                    bad.append((uu[j], len(s), int(hh.shape[0])))
                    continue
                mm[pos:pos + hh.shape[0]] = hh.half().cpu().numpy()
                pos += hh.shape[0]
                kept_u.append(uu[j])
                kept_l.append(len(s))
            del o, h, mk
            if (i // args.batch) % 40 == 0:
                done = i + len(chunk)
                el = max(time.time() - t0, 1e-9)
                print(f"  [enc] {done:,}/{len(seqs):,}  {done/el:.1f}/s  "
                      f"位置 {pos:,}/{total:,}  "
                      f"ETA {(len(seqs)-done)/(done/el)/60:.1f} min", flush=True)
            if (i // args.batch) % 400 == 0:
                mm.flush()
    mm.flush()

    if bad:
        print(f"\n[bad] {len(bad)} 条长度不符被丢弃（前 5）:")
        for u, want, got in bad[:5]:
            print(f"      {u}  期望 {want}  实得 {got}")
    if pos < total:
        print(f"[trunc] 实际写入 {pos:,} < 预分配 {total:,}，"
              f"把文件截到 {pos*DIM*2/1024**3:.2f} GiB")
        del mm
        with open(mm_path, "r+b") as f:
            f.truncate(pos * DIM * 2)
        mm = np.memmap(mm_path, dtype=np.float16, mode="r", shape=(pos, DIM))
        mm.flush()

    # 关键：offsets 用**实际写入**的 kept_l 算，不是预分配的 lens
    lens_t = torch.tensor(kept_l, dtype=torch.long)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), lens_t]), 0)[:-1]
    torch.save(dict(lens=lens_t, offsets=offsets, uids=kept_u,
                    n_pos=pos, dim=DIM, mmap=str(mm_path)), idx_path)

    el = time.time() - t0
    print(f"\n[out]  写入 {pos:,} 位置   序列 {len(kept_u):,} 条   丢弃 {len(bad)}")
    print(f"[save] {idx_path}")
    print(f"[save] {mm_path}  {mm_path.stat().st_size/1024**3:.2f} GiB")
    print(f"       offsets 末项+末长 = {int(offsets[-1])+kept_l[-1]:,}  "
          f"应等于 n_pos {pos:,}  ->  {'OK' if int(offsets[-1])+kept_l[-1]==pos else '**错位**'}")
    print(f"用时 {el/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
