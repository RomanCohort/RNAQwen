"""重建 HuggingFace 的暂存目录 `_hf_upload/`（脚本 49_ 从它推送）。

为什么要有这个脚本
------------------
上一版 `_hf_upload/` 是**手工拼的**：没有清单、没有排除规则、没人知道
哪个文件为什么在里面。今天新增了几十个脚本、几个数据集、两个已训成的
artifacts，还有一份大改过的 PLAN —— 手工再拼一遍只会更乱。

所以这里用**显式清单 + 显式排除**，并且 `--dry` 能先看要推什么。

排除的东西和理由（都是有意排除，不是漏了）
------------------------------------------
    _rfa_full_hidden.f16        68 GiB。派生数据，73_/79_ 能从 all.fa 重生成（5.3 小时）。
                                **推理不需要它** —— 推理是「序列 -> ERNIE-RNA -> 码」，
                                隐藏态只在训练时用。
    _fsq_full_pos.pt            5.9 GiB，同上
    models/                     15 GB 底座，外部可下载
    runs/*/ckpt/*.pt            训练中间态，每个 300 MB，没有交付价值
    rnaqa_seq30_full.jsonl      2.4 GB，145,201 条的全量 QA，可由 58_ 重生成
    单个文件 > 400 MB           跳过并告警

用法
    python scripts/93_build_hf_upload.py --dry
    python scripts/93_build_hf_upload.py
    python scripts/93_push_hf.py            # 然后推
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_hf_upload"
MAX_MB = 400

# 目录级收录（整个目录递归）
INCLUDE_DIRS = [
    "scripts",
    "src",
    "configs",
    "checkpoints/fsq_full",
    "runs/B_seq8/artifacts",       # 8B 唯一通过三臂验证的那个（A 80.0%）
    "runs/C_merged/artifacts",     # 合并失败的那个，留作证据
]

# 文件级收录
INCLUDE_FILES = [
    "PLAN_v2.md", "README.md", "RUN.md", "PROGRESS.md",
    "data/processed/rnaqa_seqtask.jsonl",        # 校正后 242k / 24,207 RNA
    "data/processed/rnaqa_seqtask_full.jsonl",   # 校正后 1.5M / 149,682 RNA
    "data/processed/rnaqa_seq8.jsonl",
    "data/processed/rnaqa_merged.jsonl",
    "data/processed/rnaqa_rfam.jsonl",
    "data/processed/rnaqa_freetext.jsonl",
    "data/processed/rnaqa_seq30_split.jsonl",
    "data/processed/split_full.json",            # 全量划分
    "data/processed/fingerprint.json",           # 首/末 8-mer 指纹
    "data/external/rfam/hits32.tbl",
]

# 报告：json 全收，日志只收小文件
INCLUDE_REPORT_GLOB = ["*.json"]
INCLUDE_REPORT_LOG_MAX_MB = 5

# 路径里含这些片段的，一律不收
EXCLUDE_PARTS = {
    "__pycache__", "_speedtest", "_hf_upload",
    "models",                                   # 15 GB 底座
}
EXCLUDE_NAMES = {
    "_rfa_full_hidden.f16", "_rfam_hidden.f16", "_fsq_full_pos.pt",
    "_fsq_probe_x.pt", "_rfa_full_index.pt", "_fsq_all_index.pt",
    "rnaqa_seq30_full.jsonl", "rnaqa_n6_merged.jsonl",
    "_probe.txt", "_smoke35b.jsonl",
}
# 路径里含这些片段的，一律不收（训练中间态）
EXCLUDE_SUBSTR = ("\\ckpt\\", "/ckpt/", "_smoke", "_tmp", ".part", ".aria2")


def wanted() -> list[Path]:
    out: list[Path] = []
    for d in INCLUDE_DIRS:
        p = ROOT / d
        if p.exists():
            out += [f for f in p.rglob("*") if f.is_file()]
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            out.append(p)
    rp = ROOT / "reports"
    if rp.exists():
        for pat in INCLUDE_REPORT_GLOB:
            out += list(rp.glob(pat))
        for f in rp.glob("*.log"):
            if f.stat().st_size <= INCLUDE_REPORT_LOG_MAX_MB * 1_000_000:
                out.append(f)
    # 过滤
    keep: list[Path] = []
    for f in out:
        rel = f.relative_to(ROOT)
        if set(rel.parts) & EXCLUDE_PARTS:
            continue
        if f.name in EXCLUDE_NAMES:
            continue
        if any(s in str(rel) for s in EXCLUDE_SUBSTR):
            continue
        keep.append(f)
    return sorted(set(keep))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="先清空 _hf_upload（默认只覆盖，保留已有文件）")
    args = ap.parse_args()

    files = wanted()
    big = [f for f in files if f.stat().st_size > MAX_MB * 1_000_000]
    if big:
        print(f"** 以下 {len(big)} 个文件超过 {MAX_MB} MB，跳过：")
        for f in big:
            print(f"     {f.stat().st_size/1e6:>8.1f} MB  {f.relative_to(ROOT)}")
        files = [f for f in files if f not in big]

    tot = sum(f.stat().st_size for f in files)
    print(f"\n收录 {len(files)} 个文件，合计 {tot/1e6:.1f} MB")
    from collections import Counter
    by_top = Counter(str(f.relative_to(ROOT)).split("\\")[0].split("/")[0] for f in files)
    for k, v in by_top.most_common():
        sz = sum(f.stat().st_size for f in files
                 if str(f.relative_to(ROOT)).startswith(k)) / 1e6
        print(f"   {k:<16} {v:>4} 个   {sz:>9.1f} MB")
    print("\n最大的 10 个：")
    for f in sorted(files, key=lambda x: -x.stat().st_size)[:10]:
        print(f"   {f.stat().st_size/1e6:>8.1f} MB  {f.relative_to(ROOT)}")

    if args.dry:
        print("\n[dry] 没动盘")
        return 0

    if args.clean and SRC.exists():
        shutil.rmtree(SRC)
        print(f"[clean] 清空 {SRC}")
    SRC.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        rel = f.relative_to(ROOT)
        dst = SRC / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    # 清单写进去，别人能看出这一版推了什么
    (SRC / "UPLOAD_MANIFEST.txt").write_text(
        f"files {n}\nbytes {tot}\n\n" + "\n".join(
            f"{f.stat().st_size:>12}  {f.relative_to(ROOT)}" for f in files),
        encoding="utf-8")
    print(f"\n[copied] {n} 个文件 -> {SRC}   {tot/1e6:.1f} MB")
    print("[next]   python scripts/49_push_hf.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
