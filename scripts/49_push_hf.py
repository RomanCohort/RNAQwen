"""按目录分组推送到 HuggingFace：RomanCohort/RNAQwen

两个已经踩过的坑（都是我的设计错误，记下来）：

  1. `upload_folder`（一次性原子提交）—— 读完 130 MB 本地文件后网络 0 字节，9 分钟不动。
     本地 ReadTransferCount 134.96 MB，WriteTransferCount 0.06 MB。挂住就全丢，且不可见。

  2. 逐文件上传 —— 198 个文件 = 198 次 create_commit，**429 Too Many Requests**。
     上传通道本身没问题（实测 65-82 KB/s），坏的是**提交次数**。

所以：**按目录分组，一次 commit 带一批文件**，总共 ~8 次提交。
429 单独处理：指数退避，必要时长时间等待。

用法:
    python scripts/49_push_hf.py --list
    python scripts/49_push_hf.py --dry
    python scripts/49_push_hf.py --cooldown 120      # 先等 2 分钟再开始
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_hf_upload"
REPO_ID = "RomanCohort/RNAQwen"
REPO_TYPE = "model"
SKIP_PARTS = {"_speedtest", "__pycache__"}
SKIP_NAMES = {"_probe.txt"}


def groups() -> list[tuple[str, list[Path]]]:
    """按目录分组 —— 每组一次 commit，把提交次数从 198 压到 ~8。"""
    files = [p for p in SRC.rglob("*") if p.is_file()
             and p.name not in SKIP_NAMES
             and not (SKIP_PARTS & set(p.relative_to(SRC).parts))]
    bucket: dict[str, list[Path]] = {}
    for p in files:
        rel = p.relative_to(SRC)
        parts = rel.parts
        if len(parts) == 1:
            key = "根目录"
        elif parts[0] == "scripts" and len(parts) > 2:
            key = f"scripts/{parts[1]}"                        # scripts/_scratch 单独一组
        elif parts[0] == "reports":
            key = f"reports/{p.stat().st_size // 1_000_000}MB"  # 报告按大小切，避免一次太多
        elif parts[0] == "data" and len(parts) > 2:
            key = f"data/{parts[1]}"
        elif parts[0] == "data":
            key = "data/大文件" if p.stat().st_size > 5_000_000 else "data/根"
        else:
            key = parts[0]
        bucket.setdefault(key, []).append(p)
    # 大文件单独成组（它们的 LFS 上传占绝大多数时间，单独一组便于重试）
    out = []
    for k, v in sorted(bucket.items()):
        big = [p for p in v if p.stat().st_size > 5_000_000]
        small = [p for p in v if p.stat().st_size <= 5_000_000]
        if small:
            out.append((k, sorted(small, key=lambda p: p.stat().st_size)))
        for p in big:
            out.append((f"{k}/{p.name}", [p]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--cooldown", type=int, default=0, help="开始前先等这么多秒（躲 429）")
    ap.add_argument("--attempts", type=int, default=5)
    args = ap.parse_args()

    gs = groups()
    total = sum(p.stat().st_size for _, v in gs for p in v)
    print(f"[src] {len(gs)} 组 / {sum(len(v) for _,v in gs)} 文件 / {total/1024/1024:.1f} MB", flush=True)
    for k, v in gs:
        print(f"   {k:<34} {len(v):>3} 文件  {sum(p.stat().st_size for p in v)/1024/1024:>7.2f} MB")
    if args.list:
        return 0

    token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"[auth] {who.get('name')} ({who.get('type')})", flush=True)
    except Exception as e:
        print(f"[auth] 失败 {type(e).__name__}: {str(e)[:200]}")
        return 1

    have: dict[str, int] = {}
    try:
        info = api.repo_info(args.repo, repo_type=REPO_TYPE, files_metadata=True)
        for s in (info.siblings or []):
            have[s.rfilename] = getattr(s, "size", None) or -1
        print(f"[repo] 远端已有 {len(have)} 个文件", flush=True)
    except Exception:
        api.create_repo(args.repo, repo_type=REPO_TYPE, exist_ok=True)

    if args.dry:
        print("[dry] 校验通过，未上传。")
        return 0

    if args.cooldown:
        print(f"[cool] 等 {args.cooldown}s 躲 429 ...", flush=True)
        time.sleep(args.cooldown)

    ok = fail = 0
    t_all = time.time()
    for gi, (name, files) in enumerate(gs, 1):
        todo = [p for p in files
                if have.get(p.relative_to(SRC).as_posix()) != p.stat().st_size]
        if not todo:
            print(f"[{gi}/{len(gs)}] skip  {name}", flush=True)
            continue
        ops = [CommitOperationAdd(path_in_repo=p.relative_to(SRC).as_posix(),
                                  path_or_fileobj=str(p)) for p in todo]
        wait = 20
        for att in range(1, args.attempts + 1):
            t0 = time.time()
            try:
                api.create_commit(repo_id=args.repo, repo_type=REPO_TYPE,
                                  operations=ops, commit_message=f"add {name}")
                dt = time.time() - t0
                mb = sum(p.stat().st_size for p in todo) / 1024 / 1024
                print(f"[{gi}/{len(gs)}] ok    {name:<34} {len(todo):>3} 文件 {mb:>7.2f} MB "
                      f"{dt:>6.1f}s  {mb*1024/dt:>6.1f} KB/s", flush=True)
                ok += 1
                break
            except Exception as e:
                msg = str(e)
                is429 = "429" in msg
                print(f"[{gi}/{len(gs)}] retry {att}/{args.attempts}  {name}  "
                      f"{'429 ' if is429 else ''}{type(e).__name__}", flush=True)
                if att == args.attempts:
                    fail += 1
                else:
                    print(f"        等 {wait}s ...", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 3, 600)      # 429 要等久一点

    print(f"\n[done] 组成功 {ok}  失败 {fail}  用时 {(time.time()-t_all)/60:.1f} min", flush=True)
    print(f"       https://huggingface.co/{args.repo}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
