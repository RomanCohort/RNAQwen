"""全量 RNA-QA 生成的**分片驱动器**：断电/重启最多只丢一片。

为什么必须这样
-------------
`12_generate_qa.py` 是"内存里全部生成完 → 最后一次性写盘"。15,200 次调用要 1~2 小时，
这中间任何一次会话重启 / 进程被杀，全部白跑（这个项目已经发生过多次）。
而且它用 `ThreadPoolExecutor.map`，一个请求重试耗尽就整批抛异常。

设计
----
* 每片 `--shard-size`（默认 200）条记录 → 800 次调用 ≈ ¥1.2（高峰）
* 输入切片 `data/processed/shards/slices/rnaqa_slice_{i:03d}.jsonl`（原样拷贝，可复现）
* 产出     `data/processed/shards/rnaqa_shard_{i:03d}.jsonl`
* 用量     `data/processed/shards/rnaqa_shard_{i:03d}.usage.json`
* 启动先扫盘：**分片存在且行数 == 切片行数 → 跳过**（断点续跑）
* 每片跑完查一次余额；低于 `--min-balance` 立刻停（不跑穿）
* 全部完成后按片序合并 → `--out`

单片 worker 用 `_qa_cost_probe.py`（它在运行时给 `_one` 打补丁累计 usage，
**不改 src**，且口径仍是 `12_generate_qa.py` 本身）。

跑法：
    python scripts/_gen_qa_sharded.py \
        --in data/raw/rnacentral_litsumm.jsonl \
        --out data/processed/rnaqa_rnacentral.jsonl \
        --shard-size 200 --concurrency 8 --min-balance 2.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen import secrets  # noqa: E402

PY = sys.executable
WORKER = ROOT / "scripts" / "_qa_cost_probe.py"


def n_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def balance() -> float | None:
    try:
        key = secrets.resolve_api_key(provider="deepseek")
        req = urllib.request.Request("https://api.deepseek.com/user/balance",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode())
        return float(d["balance_infos"][0]["total_balance"])
    except Exception as exc:  # noqa: BLE001
        print(f"[bal] 查询失败 {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return None


def usage_of(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("usage") or {}
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "data" / "processed" / "shards")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=0, help="0 = 到最后")
    ap.add_argument("--model", default="deepseek:deepseek-flash")
    ap.add_argument("--variants", default="ais,dc")
    ap.add_argument("--kinds", default="open,closed")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--min-balance", type=float, default=2.0)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    lines = [ln for ln in args.inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    slices = args.shard_dir / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    n_shard = (total + args.shard_size - 1) // args.shard_size
    end = args.shard_end or n_shard

    b0 = balance()
    print(f"[gen] 输入 {total} 条 → {n_shard} 片 × {args.shard_size} 条/片；"
          f"模型 {args.model}；并发 {args.concurrency}", flush=True)
    print(f"[gen] 起始余额 ¥{b0 if b0 is None else round(b0,2)}；"
          f"处理片 [{args.shard_start}, {end})；停机阈值 ¥{args.min_balance}", flush=True)
    if b0 is not None and b0 < args.min_balance:
        print("[gen] 起始余额已低于阈值 → 直接停下报告")
        return 2

    t_all = time.time()
    stopped_for_balance = False
    if not args.merge_only:
        for k in range(args.shard_start, end):
            body = lines[k * args.shard_size:(k + 1) * args.shard_size]
            if not body:
                continue
            sl = slices / f"rnaqa_slice_{k:03d}.jsonl"
            sh = args.shard_dir / f"rnaqa_shard_{k:03d}.jsonl"
            us = args.shard_dir / f"rnaqa_shard_{k:03d}.usage.json"
            if n_lines(sh) == len(body):
                print(f"[gen] 片 {k:03d} 已完成（{len(body)} 条）→ 跳过", flush=True)
                continue
            sl.write_text("\n".join(body) + "\n", encoding="utf-8")

            ok = False
            for attempt in range(args.retries):
                t0 = time.time()
                cmd = [PY, str(WORKER), "--in", str(sl), "--out", str(sh),
                       "--limit", "0", "--model", args.model,
                       "--variants", args.variants, "--kinds", args.kinds,
                       "--n-pairs", str(args.n_pairs), "--concurrency", str(args.concurrency),
                       "--keep-dropped", "--report", str(us)]
                print(f"[gen] 片 {k:03d}/{n_shard-1} 试 {attempt+1}：{len(body)} 条 "
                      f"({len(body)*len(args.variants.split(','))*len(args.kinds.split(','))} 次调用)",
                      flush=True)
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                if r.returncode == 0 and n_lines(sh) == len(body):
                    ok = True
                    u = usage_of(us)
                    print(f"[gen] 片 {k:03d} 完成 {len(body)} 条 / {time.time()-t0:.0f}s "
                          f"usage={json.dumps(u, ensure_ascii=False)}", flush=True)
                    break
                tail = "\n".join((r.stdout or "").strip().splitlines()[-8:])
                print(f"[gen] 片 {k:03d} 失败 rc={r.returncode}（{time.time()-t0:.0f}s）\n"
                      f"--- stdout tail ---\n{tail}\n--- stderr tail ---\n"
                      f"{(r.stderr or '')[-600:]}", flush=True)
                time.sleep(5 * (attempt + 1))

            bb = balance()
            cum = {}
            for j in range(n_shard):
                for kk, vv in usage_of(args.shard_dir / f"rnaqa_shard_{j:03d}.usage.json").items():
                    cum[kk] = cum.get(kk, 0) + vv
            print(f"[gen] 累计余额 ¥{bb if bb is None else round(bb,2)}；累计 usage={json.dumps(cum, ensure_ascii=False)}",
                  flush=True)

            if not ok:
                print(f"[gen] 片 {k:03d} 多次失败 → 停止（已完成的分片保留，重跑即续）", flush=True)
                return 1
            if bb is not None and bb < args.min_balance:
                print(f"[gen] 余额 ¥{bb:.2f} < ¥{args.min_balance} → 停机报告", flush=True)
                stopped_for_balance = True
                break

    # ---- 合并（只合并存在的分片；缺片会明确报出来）----
    out_lines: list[str] = []
    dropped_lines: list[str] = []
    missing: list[int] = []
    n_qa = 0
    for k in range(n_shard):
        sh = args.shard_dir / f"rnaqa_shard_{k:03d}.jsonl"
        body_n = len(lines[k * args.shard_size:(k + 1) * args.shard_size])
        if n_lines(sh) != body_n:
            missing.append(k)
            continue
        out_lines.extend(ln for ln in sh.read_text(encoding="utf-8").splitlines() if ln.strip())
        dp = args.shard_dir / f"rnaqa_shard_{k:03d}.dropped.jsonl"
        if dp.exists():
            dropped_lines.extend(ln for ln in dp.read_text(encoding="utf-8").splitlines() if ln.strip())
    for ln in out_lines:
        try:
            n_qa += len(json.loads(ln).get("qa", []))
        except json.JSONDecodeError:
            pass

    cum_total: dict[str, float] = {}
    for k in range(n_shard):
        for kk, vv in usage_of(args.shard_dir / f"rnaqa_shard_{k:03d}.usage.json").items():
            cum_total[kk] = cum_total.get(kk, 0) + vv
    b1 = balance()

    print(f"\n[gen] ===== 汇总 =====")
    print(f"[gen] 已完成分片 {n_shard-len(missing)}/{n_shard}；缺片 {missing}")
    print(f"[gen] 合并 {len(out_lines)} 条记录 / {n_qa} 组 QA -> {args.out}")
    print(f"[gen] 闸门明细 {len(dropped_lines)} 条")
    print(f"[gen] 累计 usage: {json.dumps(cum_total, ensure_ascii=False)}")
    if b0 is not None and b1 is not None:
        print(f"[gen] 余额 ¥{b0:.2f} → ¥{b1:.2f}；实测花费 ¥{b0-b1:.2f}")
    print(f"[gen] 总用时 {(time.time()-t_all)/60:.1f} min")

    if out_lines:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        side = args.out.with_suffix(".dropped.jsonl")
        side.write_text(("\n".join(dropped_lines) + "\n") if dropped_lines else "", encoding="utf-8")
        print(f"[gen] 写入 -> {args.out}\n[gen] 闸门 -> {side}")

    summary = {
        "input_records": total, "shard_size": args.shard_size, "n_shards": n_shard,
        "shards_done": n_shard - len(missing), "shards_missing": missing,
        "records_written": len(out_lines), "qa_total": n_qa,
        "dropped_total": len(dropped_lines), "usage": cum_total,
        "balance_before": b0, "balance_after": b1,
        "spent_cny": (b0 - b1) if (b0 is not None and b1 is not None) else None,
        "model": args.model, "variants": args.variants, "kinds": args.kinds,
        "n_pairs": args.n_pairs, "concurrency": args.concurrency,
        "stopped_for_balance": stopped_for_balance,
        "elapsed_min": round((time.time() - t_all) / 60, 1),
    }
    (args.shard_dir / "gen_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen] 汇总 -> {args.shard_dir / 'gen_summary.json'}")

    if missing:
        print(f"[fail] 还有 {len(missing)} 片没完成 → 不视为成功")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
