"""重下 RNA-FM，硬校验 sha256。

为什么必须这么做（第二次栽同一个坑）：
  models/rnafm/model.safetensors 大小 398,117,588 与上游**逐字节相同**，
  但 sha256 不符，205 个张量里 **155 个全零** —— 整个 12 层 encoder 是空的。
  `from_pretrained` 不报错，LOAD REPORT 只提 lm_head/ss_head/pooler，
  于是模型静默地变成「只看位置、不看序列」。所有下游结果都是废的。

规矩（与 Qwen3-8B 那次一样）：
  1) 单写者，下到 .part；
  2) **强制** sha256 比对上游声明值；
  3) 通过后才原子改名；
  4) 再独立验证一次「换序列输出会变」。少了第 4 步，哈希对了也可能用错文件。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
DEST = ROOT / "models" / "rnafm" / "model.safetensors"
PART = DEST.with_suffix(".safetensors.part")
URL = "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors"
EXPECT_SHA = "ec69333eb894dbefb580e690b0cbe1cbae9845fefd2bc78da914cd14226a2828"
EXPECT_SIZE = 398117588


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def archive_corrupt() -> None:
    if not DEST.exists():
        return
    cur = sha256_of(DEST)
    if cur == EXPECT_SHA:
        print("[skip] 现有文件哈希已正确，无需重下")
        return
    bad = DEST.with_name("model.safetensors.corrupt")
    if bad.exists():
        bad.unlink()
    DEST.rename(bad)
    print(f"[keep] 损坏文件留档 -> {bad.name} (sha256={cur[:16]}...)")


def download_aria2() -> bool:
    exe = shutil.which("aria2c")
    if not exe:
        print("[aria2] 未找到 aria2c")
        return False
    # 断点续传：--continue=true 会读 .part.aria2 控制文件接着下。
    # 之所以敢续，是因为下面 --file-allocation=none —— .part 里只有真实下到的数据，
    # 不会有预分配留下的零填充空洞。**上次坏就坏在预分配 + 续传。**
    # 哈希闸门在最后，续传期间拿到垃圾也无所谓，反正过不了 sha256。
    cmd = [exe, "-x", "16", "-s", "16", "-k", "1M", "--continue=true",
           "--file-allocation=none",
           "--max-tries=0", "--retry-wait=10", "--timeout=60",
           "--summary-interval=60", "--console-log-level=warn",
           "-d", str(PART.parent), "-o", PART.name, URL]
    print("[aria2] " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"[aria2] 退出码 {rc}，用时 {(time.time()-t0)/60:.1f} min")
    if rc != 0:
        print("[aria2] 非零退出。**别删 .part** —— 重跑本脚本会接着下。")
    return PART.exists() and PART.stat().st_size > 0


def download_urllib() -> bool:
    """回退：单连接。支持用 Range 续传（aria2 挂了的时候不至于前功尽弃）。"""
    have = PART.stat().st_size if PART.exists() else 0
    headers = {"User-Agent": "curl/8.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"[urllib] 回退单连接下载，从 {have/1e6:.1f} MB 续传")
    else:
        print("[urllib] 回退单连接下载")
    try:
        # ⚠️ urllib 的 timeout 只吃标量。写 (30, 120) 是 requests 的习惯 -> TypeError。
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(PART, "ab" if have else "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) + have
            got, t0, last = have, time.time(), 0.0
            while True:
                b = r.read(1024 * 1024)
                if not b:
                    break
                f.write(b)
                got += len(b)
                if time.time() - last > 15:
                    last = time.time()
                    el = max(time.time() - t0, 0.01)
                    print(f"  {got/1e6:8.1f}/{total/1e6:.1f} MB  "
                          f"{(got-have)/el/1e6:6.2f} MB/s  "
                          f"eta {(total-got)/max((got-have)/el,1)/60:5.1f} min", flush=True)
    except Exception as e:
        print("[urllib] 失败:", e)
        return PART.exists()
    return PART.exists()


def main() -> int:
    print("=" * 72)
    print("重下 RNA-FM 并硬校验")
    print("  目标  :", DEST)
    print("  期望  : size=%d  sha256=%s" % (EXPECT_SIZE, EXPECT_SHA))
    print("=" * 72)

    archive_corrupt()
    if DEST.exists() and sha256_of(DEST) == EXPECT_SHA:
        print("[done] 已经是好的，直接进验证")
        return verify()

    if PART.exists() and not PART.with_name(PART.name + ".aria2").exists():
        # 没有 aria2 控制文件的 .part = 上一次是 urllib 下的，续不了，只能重来
        print("[note] 发现无控制文件的 .part，无法续传，删掉重下")
        PART.unlink()

    ok = download_aria2() or download_urllib()
    if not ok:
        print("[fail] 下载没成功")
        return 1

    size = PART.stat().st_size
    print(f"[verify] 大小 {size} (期望 {EXPECT_SIZE}) {'OK' if size == EXPECT_SIZE else '不符'}")
    got = sha256_of(PART)
    print(f"[verify] sha256 {got}")
    if got != EXPECT_SHA:
        print("[FAIL] 哈希不符 —— **绝不改名**")
        # 关键分支：
        #   大小已满 + 哈希错 = aria2 自认为下完了，但内容里混着零填充空洞
        #   （预分配 + 中断的经典产物）。这种情况**续传救不了** —— 控制文件会说
        #   "所有分片都完成了"，于是每次续传都立刻返回同样的坏文件，卡死循环。
        #   必须连控制文件一起清掉，下次从头下。
        #   大小未满 = 真的只是没下完，保留 .part 让它续。
        if size >= EXPECT_SIZE:
            ctl = PART.with_name(PART.name + ".aria2")
            print(f"[FAIL] 大小已满但内容错 -> 判定为「预分配空洞」，清掉 .part 与控制文件重来")
            print(f"       清 {PART.name}")
            PART.unlink(missing_ok=True)
            if ctl.exists():
                print(f"       清 {ctl.name}")
                ctl.unlink()
            print("       **重跑本脚本即可**（会从头下，不会再续进这个坏状态）")
        else:
            print(f"[FAIL] 只下了 {size/1e6:.1f} MB，未完成 -> 保留 .part，重跑本脚本会接着下")
        return 1
    print("[verify] 哈希一致")

    if DEST.exists():
        DEST.unlink()
    PART.rename(DEST)          # 同盘 rename 是原子的
    print(f"[done] 原子改名 -> {DEST.name}")
    return verify()


def verify() -> int:
    """独立验证：换序列，encoder 输出必须变。"""
    print("\n" + "=" * 72)
    print("独立验证：序列敏感性（哈希对了也可能装错文件）")
    print("=" * 72)
    sys.path.insert(0, str(ROOT / "src"))
    import rnaqwen  # noqa: F401
    import torch
    from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

    MP = str(DEST.parent)
    m = RnaFmModel.from_pretrained(MP).eval()
    tok = RnaTokenizer.from_pretrained(MP)

    w = m.state_dict()["embeddings.word_embeddings.weight"].float()
    print("  word_embeddings absmax = %.6e  nnz = %d/%d"
          % (float(w.abs().max()), int((w.abs() > 1e-8).sum()), w.numel()))

    import glob
    from safetensors.torch import load_file
    raw = {}
    for f in glob.glob(os.path.join(MP, "*.safetensors")):
        raw.update(load_file(f))
    zeros = [k for k in raw if raw[k].numel() and float(raw[k].float().abs().max()) == 0.0]
    print("  全零张量 = %d / %d" % (len(zeros), len(raw)))

    seqs = ["ACGUACGUACGUACGUACGU", "GGGGGGGGGGGGGGGGGGGG", "UUUUUUUUUUUUUUUUUUUU"]
    enc = tok(seqs, return_tensors="pt", padding=True)
    with torch.no_grad():
        o = m(**enc).last_hidden_state
    d1 = float((o[0] - o[1]).abs().max())
    d2 = float((o[0] - o[2]).abs().max())
    print("  ACGU vs GGGG  max|Δ| = %.6e" % d1)
    print("  ACGU vs UUUU  max|Δ| = %.6e" % d2)
    print("  输出形状 =", tuple(o.shape))

    ok = (not zeros) and d1 > 1e-6 and d2 > 1e-6 and float(w.abs().max()) > 1e-6
    print("\n[VERDICT]", "PASS —— RNA-FM 已修好，可以重跑 Stage 1" if ok
          else "FAIL —— 别急着训，先查清楚")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
