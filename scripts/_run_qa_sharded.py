"""分片跑 `12_generate_qa.py`，做到**可续跑 + 不丢已完成的活**。

为什么需要
----------
`12_generate_qa.py` 是整批一次 `gen.chat()`：15,200 个任务一把梭，
中途任何一个请求重试耗尽就整体抛异常，前面几小时全白干。
这里把它切成 N 条记录一片，每片一个子进程、一个独立输出文件：

    work/qa_slices/slice_0007.jsonl   输入切片（原样拷贝，只取一个区间）
    work/qa_shards/shard_0007.jsonl   该片的产出
    work/qa_shards/shard_0007.dropped.jsonl

片文件存在且行数正确 = 这片已完成 → 跳过。所以**中断后重跑就是续跑**。
全部完成后按片序拼成最终文件（顺序与输入一致）。

跑法：
    python scripts/_run_qa_sharded.py \
        --in data/raw/rnacentral_litsumm.jsonl \
        --out data/processed/rnaqa_rnacentral.jsonl \
        --shard-size 200 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GEN = ROOT / "scripts" / "12_generate_qa.py"


def n_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=ROOT / "work" / "qa_shards")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=0, help="0 = 到最后")
    ap.add_argument("--model", default="deepseek:deepseek-chat")
    ap.add_argument("--variants", default="ais,dc")
    ap.add_argument("--kinds", default="open,closed")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4, help="一片最多试几次")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    lines = [ln for ln in args.inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    slices = args.work / "slices"
    shards = args.work / "shards"
    slices.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)

    n_shard = (total + args.shard_size - 1) // args.shard_size
    end = args.shard_end or n_shard
    print(f"[shard] 输入 {total} 条 → {n_shard} 片（每片 {args.shard_size}）；本轮处理片 "
          f"[{args.shard_start}, {end})", flush=True)

    t_all = time.time()
    for k in range(args.shard_start, end):
        sl = slices / f"slice_{k:04d}.jsonl"
        sh = shards / f"shard_{k:04d}.jsonl"
        body = lines[k * args.shard_size:(k + 1) * args.shard_size]
        if not body:
            continue
        if n_lines(sh) == len(body):
            continue
        sl.write_text("\n".join(body) + "\n", encoding="utf-8")
        ok = False
        for attempt in range(args.retries):
            t0 = time.time()
            cmd = [PY, str(GEN), "--in", str(sl), "--out", str(sh),
                   "--model", args.model, "--variants", args.variants,
                   "--kinds", args.kinds, "--n-pairs", str(args.n_pairs),
                   "--concurrency", str(args.concurrency), "--keep-dropped"]
            print(f"[shard] {k}/{n_shard-1} 试 {attempt+1}：{len(body)} 条 ...", flush=True)
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
            if r.returncode == 0 and n_lines(sh) == len(body):
                ok = True
                print(f"[shard] {k} 完成 {len(body)} 条 / {time.time()-t0:.1f}s", flush=True)
                break
            print(f"[shard] {k} 失败 rc={r.returncode}（{time.time()-t0:.1f}s）\n"
                  f"--- stdout tail ---\n{tail}\n--- stderr tail ---\n"
                  f"{(r.stderr or '')[-500:]}", flush=True)
            time.sleep(3 * (attempt + 1))
        if not ok:
            print(f"[shard] {k} 多次失败 → 停止（已完成的分片会保留，重跑即续）", flush=True)
            return 1

    # ---- 合并 ----
    out_lines: list[str] = []
    dropped_lines: list[str] = []
    n_qa = 0
    for k in range(n_shard):
        sh = shards / f"shard_{k:04d}.jsonl"
        if sh.exists():
            out_lines.extend(ln for ln in sh.read_text(encoding="utf-8").splitlines() if ln.strip())
        dp = shards / f"shard_{k:04d}.dropped.jsonl"
        if dp.exists():
            dropped_lines.extend(ln for ln in dp.read_text(encoding="utf-8").splitlines() if ln.strip())
    if not out_lines:
        print("[shard] 没有任何分片产出，不合并")
        return 1
    for ln in out_lines:
        try:
            n_qa += len(json.loads(ln).get("qa", []))
        except json.JSONDecodeError:
            pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    side = args.out.with_suffix(".dropped.jsonl")
    side.write_text(("\n".join(dropped_lines) + "\n") if dropped_lines else "", encoding="utf-8")
    print(f"[shard] 合并 {len(out_lines)} 条记录 / {n_qa} 组 QA -> {args.out}", flush=True)
    print(f"[shard] 闸门明细 {len(dropped_lines)} 条 -> {side}", flush=True)
    print(f"[shard] 总用时 {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
