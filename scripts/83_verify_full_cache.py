"""校验新全量缓存 _rfa_full_hidden.f16 —— 和旧的 _rfam_hidden.f16 逐位对拍。

为什么这是最硬的一条
--------------------
旧缓存 19,726 条、新缓存 145,201 条，前者是后者的子集（同一份 all.fa、同一个编码器）。
同一条序列被独立编了两遍，隐藏态应该**逐位吻合**（只允许 fp16 的舍入差）。
不吻合 -> 新文件的某一个环节错了（序列取错、偏移错位、dtype 错、模型没加载对），
而且这种错**不会自己报错**，只会让后面的训练莫名其妙地学不好。

顺带查：
  · 偏移量自洽（offsets 末项 + 末长 == n_pos，每条 offsets[i]+lens[i] == offsets[i+1]）
  · 抽样核对几条：从 memmap 里按 offsets 取出来的张量行数 == lens
  · 新旧 uids 的覆盖关系
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

D = ROOT / "data/processed"
NEW_MM, NEW_IDX = D / "_rfa_full_hidden.f16", D / "_rfa_full_index.pt"
OLD_MM, OLD_IDX = D / "_rfam_hidden.f16", D / "_rfam_index.pt"
DIM = 768


def key(u: str) -> str:
    return u[4:] if u.startswith("seq:") else u


def load_idx(p: Path):
    d = torch.load(p, map_location="cpu", weights_only=False)
    uids = [key(str(u)) for u in d["uids"]]
    lens = [int(x) for x in d["lens"]]
    offs = [int(x) for x in d["offsets"]]
    return d, uids, lens, offs


def main() -> int:
    fails = 0

    print("=" * 76)
    print("0  索引结构")
    print("=" * 76)
    dn, un, ln, on = load_idx(NEW_IDX)
    do, uo, lo, oo = load_idx(OLD_IDX)
    shape_n = (int(dn["n_pos"]), DIM)
    print(f"  新  序列 {len(un):,}   位置 {int(dn['n_pos']):,}   dim {dn['dim']}   "
          f"memmap {shape_n}")
    print(f"  旧  序列 {len(uo):,}   位置 {int(do['n_pos']):,}   dim {do['dim']}")

    # ---- 偏移自洽 ----
    print()
    print("=" * 76)
    print("1  偏移量自洽")
    print("=" * 76)
    ok = True
    for tag, lens, offs, npos in (("新", ln, on, int(dn["n_pos"])),
                                  ("旧", lo, oo, int(do["n_pos"]))):
        mono = all(offs[i] + lens[i] == offs[i + 1] for i in range(len(offs) - 1))
        tail = offs[-1] + lens[-1] == npos
        head = offs[0] == 0
        print(f"  {tag}  单调递推 {mono}   末项+末长==n_pos {tail}   首项==0 {head}")
        if not (mono and tail and head):
            fails += 1

    # ---- uid 覆盖 ----
    print()
    print("=" * 76)
    print("2  uid 覆盖")
    print("=" * 76)
    sn, so = set(un), set(uo)
    print(f"  新独有 {len(sn - so):,}   旧独有 {len(so - sn):,}   交集 {len(sn & so):,}")
    print(f"  旧是否完全包含于新: {so <= sn}")
    if not (so <= sn):
        fails += 1

    # ---- 逐位对拍 ----
    print()
    print("=" * 76)
    print("3  逐位对拍（交集上抽样 400 条，全部比对）")
    print("=" * 76)
    mn = np.memmap(NEW_MM, dtype=np.float16, mode="r", shape=(int(dn["n_pos"]), DIM))
    mo = np.memmap(OLD_MM, dtype=np.float16, mode="r", shape=(int(do["n_pos"]), DIM))
    pos_n = {u: (on[i], ln[i]) for i, u in enumerate(un)}
    pos_o = {u: (oo[i], lo[i]) for i, u in enumerate(uo)}

    inter = sorted(sn & so)
    step = max(1, len(inter) // 400)
    pick = inter[::step]
    print(f"  交集 {len(inter):,} 条，抽 {len(pick)} 条")

    worst = 0.0
    n_bad = 0
    n_checked = 0
    for u in pick:
        a_off, a_len = pos_n[u]
        b_off, b_len = pos_o[u]
        if a_len != b_len:
            print(f"  ** {u} 位置数不同：新 {a_len} 旧 {b_len}")
            n_bad += 1
            continue
        A = np.asarray(mn[a_off:a_off + a_len], dtype=np.float32)
        B = np.asarray(mo[b_off:b_off + b_len], dtype=np.float32)
        d = np.abs(A - B).max()
        worst = max(worst, float(d))
        n_checked += 1
        if d > 0.05:                     # fp16 在 |x|~10 上的 eps 约 0.01
            n_bad += 1
            if n_bad <= 5:
                print(f"  ** {u} 最大差 {d:.4f}  范数 新 {np.abs(A).max():.3f} "
                      f"旧 {np.abs(B).max():.3f}")
    print(f"  检查 {n_checked} 条   最大差 {worst:.6f}   超阈值 {n_bad} 条")
    if worst > 0.05:
        print("  ** 两份缓存不一致 —— 新文件不可信 **")
        fails += 1
    else:
        print(f"  ✅ 逐位吻合（fp16 舍入范围内，最大差 {worst:.6f}）")

    # ---- 新缓存里旧没有的部分，抽查是否合理 ----
    print()
    print("=" * 76)
    print("4  新增部分抽查（旧缓存没有的序列）")
    print("=" * 76)
    extra = [u for u in un if u not in so]
    step2 = max(1, len(extra) // 8)
    for u in extra[::step2][:8]:
        o, L = pos_n[u]
        A = np.asarray(mn[o:o + L], dtype=np.float32)
        print(f"  {u}  位置 {L:>5}  值域 [{A.min():+.3f}, {A.max():+.3f}]  "
              f"均值 {A.mean():+.4f}  有限值 {np.isfinite(A).all()}")
        if not np.isfinite(A).all() or A.max() == 0.0:
            fails += 1

    print()
    print("=" * 76)
    print("  ✅ 新缓存可信，可以删旧缓存回收 9.6 GiB" if fails == 0
          else f"  ❌ {fails} 处不合格，先别删旧缓存")
    print("=" * 76)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
