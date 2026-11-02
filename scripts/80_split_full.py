"""给全部 145,201 条 RNA 定划分，并标出「首/末 8-mer 未在训练集出现过」的测试子集。

为什么不做簇划分
----------------
先前打算按 20-mer 连通分量划簇（57_ 的做法）。**这个方向是错的**：
`first 8` 只有 4^8 = 65,536 种可能，而我们有 145,201 条序列 ——
鸽笼原理下平均每个 8-mer 对应 2.2 条序列，**不存在"测试集首 8 在训练集里没见过"的划分**。

而这并不构成问题，因为 route C 不是泛化任务：模型被要求从码信道**读**出答案，
不是从见过的样本**推**出答案。答案按定义就在输入里。

真正要排除的是另一件事：模型绕开码信道，靠「这个 8-mer 在语料里常见」的先验蒙对。
量它的办法不需要聚类：
  · 在「首 8 未在训练集出现过」的测试子集上单独报准确率 —— 和整体一致就排除了记忆
  · C 臂（码向量置零）本来就单独量了先验本身

划分口径
--------
  seq8 的 4,481 条      沿用原 split（保持与 B_seq8 可比，且它本来就是按簇划的）
  rfam 的 19,726 条     沿用原 split（保持与 C_merged 可比）
  其余 125,475 条       按 uid 的 md5 稳定散列 90/10 分 train/test

产出
----
  data/processed/split_full.json     uid -> split
  data/processed/fingerprint.json    uid -> {seen8_head, seen8_tail}
                                     只对 test 有意义（train 里必然"见过"）

用法：python scripts/80_split_full.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed"
FA = ROOT / "data/external/rfam/all.fa"
OUT_SPLIT = D / "split_full.json"
OUT_FP = D / "fingerprint.json"
TEST_FRAC = 0.10
VAL_FRAC = 0.02


def key(u: str) -> str:
    return u[4:] if u.startswith("seq:") else u


def effective(s: str, pool: int = 4) -> str:
    """模型实际能看到的序列（FSQCode 会截掉尾部，见 78_）。"""
    return s[:len(s) // pool * pool]


def read_fa(p: Path) -> dict[str, str]:
    got, cur, buf = {}, None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    got[cur] = "".join(buf).upper()
                cur = key(line[1:].strip().split()[0])
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        got[cur] = "".join(buf).upper()
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC,
                    help="从**新增**序列里再切一份 val 给早停用。"
                         "已沿用的划分不改（保住与 C_merged 的可比性）")
    args = ap.parse_args()

    fa = read_fa(FA)
    print(f"[all.fa] {len(fa):,} 条序列")

    # ---- 1. 沿用已有的划分 ----
    split: dict[str, str] = {}
    for name, f in (("seq8", D / "rnaqa_seq8.jsonl"), ("rfam", D / "rnaqa_rfam.jsonl")):
        n = 0
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            u = key(r["uid"])
            sp = r.get("split", "train")
            if u in fa and u not in split:
                split[u] = sp
                n += 1
        print(f"[{name}] 沿用 {n:,} 条")
    print(f"[carry] 合计沿用 {len(split):,} 条  {dict(Counter(split.values()))}")

    # ---- 2. 其余的按 md5 散列（三档：val / test / train）----
    new = [u for u in fa if u not in split]
    n_test = n_val = 0
    for u in new:
        h = int(hashlib.md5(u.encode()).hexdigest()[:12], 16) / 0x1000000000000
        if h < args.val_frac:
            sp = "val"
            n_val += 1
        elif h < args.val_frac + args.test_frac:
            sp = "test"
            n_test += 1
        else:
            sp = "train"
        split[u] = sp
    print(f"[hash] 新分 {len(new):,} 条  "
          f"train {len(new)-n_test-n_val:,} / val {n_val:,} / test {n_test:,}")

    tot = Counter(split.values())
    print(f"[total] {len(split):,} 条  {dict(tot)}")
    assert set(split) == set(fa), "划分覆盖和 all.fa 不一致"

    # ---- 3. 首/末 8-mer 指纹 ----
    print("\n[fingerprint] 统计首/末 8-mer 在训练集里的出现情况")
    tr = [u for u in fa if split[u] == "train"]
    te = [u for u in fa if split[u] == "test"]
    head_tr: set[str] = set()
    tail_tr: set[str] = set()
    for u in tr:
        e = effective(fa[u])
        head_tr.add(e[:8])
        tail_tr.add(e[-8:])
    print(f"  训练集 {len(tr):,} 条 -> 首 8 唯一值 {len(head_tr):,}   末 8 唯一值 {len(tail_tr):,}")

    fp: dict[str, dict[str, bool]] = {}
    n_h = n_t = 0
    for u in te:
        e = effective(fa[u])
        h8, t8 = e[:8], e[-8:]
        r = {"seen8_head": h8 in head_tr, "seen8_tail": t8 in tail_tr}
        fp[u] = r
        n_h += r["seen8_head"]
        n_t += r["seen8_tail"]

    print(f"  测试集 {len(te):,} 条：")
    print(f"    首 8 在训练集见过 {n_h:,} = {100*n_h/len(te):.1f}%   "
          f"未见过 {len(te)-n_h:,} = {100*(len(te)-n_h)/len(te):.1f}%")
    print(f"    末 8 在训练集见过 {n_t:,} = {100*n_t/len(te):.1f}%   "
          f"未见过 {len(te)-n_t:,} = {100*(len(te)-n_t)/len(te):.1f}%")
    print()
    print("  ** 注意：'未见过'这个子集不是空集，也不用做簇划分。")
    print("     在这个子集上准确率和整体一致 -> 模型确实在读码信道，不是背先验。")

    OUT_SPLIT.write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    OUT_FP.write_text(json.dumps(fp, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_SPLIT}  ({OUT_SPLIT.stat().st_size/1024**2:.1f} MB)")
    print(f"[save] {OUT_FP}  ({OUT_FP.stat().st_size/1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
