"""为 rfam 的 19,726 条 RNA 建隐藏态缓存。

rnaqa_rfam.jsonl 里没有 sequence 字段（61_ 只写了 uid/question/answer/...），
而 uid 是 `seq:<md5(sequence)[:16]>`。序列要从 data/external/rfam/all.fa 里按 id 取。

产出与 _fsq_full_pos.pt 兼容的格式：{x, lens, offsets}，顺序 = uids 顺序，
另外存 uids 和 uid→行号 的映射，供 50_ 的 _mk_new 用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FA = ROOT / "data/external/rfam/all.fa"
QA = ROOT / "data/processed/rnaqa_rfam.jsonl"
MM = ROOT / "data/processed/_rfam_hidden.f16"
IDX = ROOT / "data/processed/_rfam_index.pt"
BATCH = 16
DIM = 768


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    # 需要哪些 uid
    need = {}
    with open(QA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            need.setdefault(r["uid"], r.get("sequence") or "")
    print(f"[need] rfam QA 涉及 {len(need):,} 个 uid")

    # 从 all.fa 取序列
    got, cur = {}, None
    with open(FA, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                cur = line[1:].strip()
            elif cur:
                got[cur] = (got.get(cur, "") + line.strip()).upper()
    print(f"[fa]   all.fa 提供 {len(got):,} 条序列")

    # 归一化键：rnaqa_rfam.jsonl 的 uid 是裸 md5（61_ 里 split(":",1)[1] 剥掉了前缀），
    # all.fa 的 id 是 "seq:<md5>"。两边必须归一化才能 join。
    def key(u):
        return u[4:] if u.startswith("seq:") else u
    got_n = {key(k): v for k, v in got.items()}
    need_n = {key(k): k for k in need}
    uids = [need_n[k] for k in need_n if k in got_n]
    miss = len(need) - len(uids)
    print(f"[join] 命中 {len(uids):,}   缺 {miss}")
    if not uids:
        print("**一条都没对上** —— 检查 uid 格式"); return 1

    seqs = [got_n[key(u)] for u in uids]
    lens = [len(s) for s in seqs]
    total = sum(lens)
    print(f"[data] 总位置 {total:,}   memmap {total*DIM*2/1024**3:.2f} GB")

    mm = np.memmap(MM, dtype=np.float16, mode="w+", shape=(total, DIM))
    print(f"[mmap] {MM}")

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    t0, pos, bad = time.time(), 0, 0
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch):
            chunk = seqs[i:i + args.batch]
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
                    bad += 1; continue
                mm[pos:pos + hh.shape[0]] = hh.half().cpu().numpy()
                pos += hh.shape[0]
            del o, h, mk
            if (i // args.batch) % 40 == 0:
                done = i + len(chunk); el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"  [enc] {done:,}/{len(seqs):,}  {rate:.1f}/s  "
                      f"位置 {pos:,}/{total:,}  ETA {len(seqs)/max(rate,1e-9)/3600:.2f} h",
                      flush=True)
            if (i // args.batch) % 400 == 0:
                mm.flush()
    mm.flush()

    lens_t = torch.tensor(lens)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), lens_t]), 0)[:-1]
    torch.save(dict(lens=lens_t, offsets=offsets, uids=uids, n_pos=pos, dim=DIM,
                    mmap=str(MM)), IDX)
    print(f"\n[out] 写入 {pos:,} / 预期 {total:,}   丢弃 {bad}")
    print(f"[save] {IDX}\n[save] {MM}  {MM.stat().st_size/1024**3:.2f} GB")
    print(f"用时 {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
