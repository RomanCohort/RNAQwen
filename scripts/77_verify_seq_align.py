"""校验 RNA 序列和隐藏态缓存逐位对齐 —— 建新数据集之前必须先过这一关。

为什么必须先做
--------------
新数据集「已验证模板 × 新 RNA」的答案是用 **字符串运算** 从序列算出来的：
    first 6 = seq[:6]      last 4 = seq[-4:]      revcomp = comp(seq[:6])[::-1]
而模型看到的是缓存里的隐藏态。序列错一位，答案就和模型看到的内容错位 ——
而且**不会报错**，只会让准确率莫名其妙地低。

三道检查
--------
1. 长度逐条相等        len(sequence) == cache lens
2. md5 自洽             rfam 的 uid = md5(sequence)[:16]（序列就是从它派生出来的）
3. 答案可复算           seq8 那些模板的答案必须精确等于用 sequence 算出来的值

用法：python scripts/77_verify_seq_align.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

FA = ROOT / "data/external/rfam/all.fa"
COMP = str.maketrans("ACGUacgu", "UGCAugca")


def key(u: str) -> str:
    return u[4:] if u.startswith("seq:") else u


def read_fa(p: Path) -> dict[str, str]:
    got, cur, buf = {}, None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    got[key(cur)] = "".join(buf)
                cur = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        got[key(cur)] = "".join(buf)
    return got


# 可以直接从序列复算答案的模板
def recompute(q: str, s: str):
    s = s.strip().upper()
    if q == "What are the first 4 nucleotides of this RNA?":
        return s[:4]
    if q == "What are the first 6 nucleotides of this RNA?":
        return s[:6]
    if q == "What are the first 8 nucleotides of this RNA?":
        return s[:8]
    if q == "What is the reverse complement of the first 4 nucleotides?":
        return s[:4].translate(COMP)[::-1]
    if q == "What is the reverse complement of the first 6 nucleotides?":
        return s[:6].translate(COMP)[::-1]
    if q == "What are the nucleotides at positions 1 through 6 of this RNA?":
        return s[:6]
    if q == "What is the dinucleotide at positions 1 and 2 of this RNA?":
        return s[:2]
    if q == "What are the last 4 nucleotides of this RNA?":
        return s[-4:]
    if q == "What are the last 6 nucleotides of this RNA?":
        return s[-6:]
    if q == "What are the last 8 nucleotides of this RNA?":
        return s[-8:]
    return None


def main() -> int:
    fails = 0

    # ================= 1. rfam：uid = md5(sequence)[:16]，序列从 all.fa 取 =========
    print("=" * 74)
    print("rfam —— 序列来自 data/external/rfam/all.fa（= 73_ 编码时用的同一份）")
    print("=" * 74)
    d2 = torch.load(ROOT / "data/processed/_rfam_index.pt", map_location="cpu",
                    weights_only=False)
    uids2 = [str(u) for u in d2["uids"]]
    lens2 = [int(x) for x in d2["lens"]]
    print(f"[cache] {len(uids2):,} 条  位置数 {min(lens2)}~{max(lens2)}")

    fa = read_fa(FA)
    print(f"[fa]    {FA.name} 提供 {len(fa):,} 条")

    miss = [u for u in uids2 if u not in fa]
    print(f"[join]  缓存 uid 在 fa 里找不到的：{len(miss)}")
    if miss:
        print(f"        {miss[:5]}")
        fails += len(miss)

    len_bad, md5_bad, n_ok = [], [], 0
    for u, L in zip(uids2, lens2):
        s = fa.get(u)
        if s is None:
            continue
        if len(s) != L:
            len_bad.append((u, L, len(s)))
        if hashlib.md5(s.encode()).hexdigest()[:16] != u:
            md5_bad.append(u)
        else:
            n_ok += 1
    print(f"[len]   长度逐条相等 {len(uids2)-len(miss)-len(len_bad):,} / "
          f"{len(uids2)-len(miss):,}   不符 {len(len_bad)}")
    if len_bad:
        for u, L, n in len_bad[:5]:
            print(f"        {u}  cache {L}  fa {n}")
        fails += len(len_bad)
    print(f"[md5]   uid == md5(sequence)[:16] 成立 {n_ok:,} / {len(uids2)-len(miss):,}   "
          f"不成立 {len(md5_bad)}")
    if md5_bad:
        print(f"        {md5_bad[:5]}")
        fails += len(md5_bad)

    # ================= 2. 主缓存（seq8 / freetext）=============================
    print()
    print("=" * 74)
    print("主缓存 —— _fsq_full_pos.pt，顺序 = 50_ load_records() 的文件顺序")
    print("=" * 74)
    d = torch.load(ROOT / "data/processed/_fsq_full_pos.pt", map_location="cpu",
                   weights_only=False)
    lens = [int(x) for x in d["lens"]]
    allr = rc.load_records(None, sort=False)
    print(f"[cache] {len(lens):,} 条  位置数 {min(lens)}~{max(lens)}")
    print(f"[recs]  load_records() {len(allr):,} 条  DATA={rc.DATA.name}")

    if len(allr) != len(lens):
        print(f"  ** 条数不一致：records {len(allr)} vs cache {len(lens)}")
        fails += 1
    else:
        bad = [(r["uid"], lens[i], len(r["sequence"]))
               for i, r in enumerate(allr) if len(r["sequence"]) != lens[i]]
        print(f"[len]   按文件顺序逐条相等 {len(allr)-len(bad):,} / {len(allr):,}   "
              f"不符 {len(bad)}")
        for u, L, n in bad[:5]:
            print(f"        {u[:28]}  cache {L}  rec {n}")
        fails += len(bad)

    # ================= 3. 答案可复算 ==========================================
    print()
    print("=" * 74)
    print("答案可复算 —— 用 sequence 算出来的值必须精确等于 jsonl 里的 answer")
    print("=" * 74)
    for name, seqof in (("rnaqa_seq8", None), ("rnaqa_merged", None)):
        p = ROOT / f"data/processed/{name}.jsonl"
        if not p.exists():
            continue
        stat: dict[str, list] = {}
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["question"]
            if r["source"] != "seq8" and name == "rnaqa_merged":
                continue
            s = r.get("sequence")
            if not s:
                continue
            eq = recompute(q, s)
            if eq is None:
                continue
            v = stat.setdefault(q, [0, 0, []])
            v[1] += 1
            if r["answer"].strip().upper() == eq.upper():
                v[0] += 1
            elif len(v[2]) < 2:
                v[2].append((r["answer"], eq, s[:24]))
        print(f"  --- {name} ---")
        for q, (ok, n, ex) in sorted(stat.items(), key=lambda x: -x[1][1]):
            flag = "✅" if ok == n else "❌"
            print(f"  {flag} {ok:>6}/{n:<6}  {q[:56]}")
            for a, e, s in ex:
                print(f"         answer {a!r} != recomputed {e!r}   seq[:24]={s!r}")
            if ok != n:
                fails += (n - ok)

    print()
    print("=" * 74)
    if fails == 0:
        print("  ✅ 全部一致 —— 可以按 sequence 建新数据集")
    else:
        print(f"  ❌ {fails} 处不一致 —— 先查清楚再建数据")
    print("=" * 74)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
