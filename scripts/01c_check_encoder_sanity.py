"""编码器健全性检查 —— 换任何 backbone 之前必须先跑这个。

为什么需要它（2026-10-09 事故）：
    `models/rnafm/model.safetensors` 被 aria2 预分配毁掉 —— 大小 398,117,588 与上游
    **逐字节相同**，205 个张量里 **155 个全零**，`word_embeddings` 全零，
    于是 encoder 对序列**零敏感**（不同序列输出 max|Δ| = 0.000000e+00）。
    而 `from_pretrained` 一声不响，LOAD REPORT 只提 lm_head/ss_head/pooler。

    更糟的是当时的探针 `01_probe_rnafm.py` **只查输出形状** `(1,83,640)` —— 
    **一个全零的模型照样过**。我据此报过 `[OK]`。

这个脚本查的是**语义**，不是形状：
    1. 结构   —— 有没有全零张量；word_embeddings 是否非零
    2. 哈希   —— 给了 --expect-sha256 就强制核对
    3. 敏感性 —— 同长度不同序列 -> 输出必须变   （抓"只看位置"）
    4. 次序   —— 同一条序列打乱 -> 输出必须变   （抓"只看碱基组成"）
    5. 确定性 —— 同一条序列跑两次 -> 输出必须逐位相同（抓随机性/漏 eval）

跑法：
    python scripts/01c_check_encoder_sanity.py --model models/rnafm
    python scripts/01c_check_encoder_sanity.py --model models/ernierna \
        --expect-sha256 02274cb7... --expect-size 345071800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402

from rnaqwen.models.encoder import ARCHS, RnaEncoder, read_model_type  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_weights(model_path: Path) -> Path | None:
    for n in ("model.safetensors", "pytorch_model.bin"):
        p = model_path / n
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--arch", default=None, choices=sorted(ARCHS),
                    help="不填就按 config.json 的 model_type 自动识别")
    ap.add_argument("--expect-sha256", default=None,
                    help="给了就强制核对 —— 这是唯一能挡住「大小对、内容空」的手段")
    ap.add_argument("--expect-size", type=int, default=None)
    ap.add_argument("--nt", type=int, default=200, help="测试序列长度")
    ap.add_argument("--device", default="cpu", help="默认 cpu：这个检查不需要 GPU")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fails: list[str] = []
    report: dict = {"model": str(args.model)}

    print("=" * 74)
    print(f"编码器健全性检查: {args.model}")
    print("=" * 74)

    # ---------- 0) 哈希 / 大小 ----------
    w = find_weights(args.model)
    if w is None:
        print("[FAIL] 目录里没有 model.safetensors / pytorch_model.bin")
        return 1
    size = w.stat().st_size
    report["weights"] = {"file": w.name, "size": size}
    print(f"  权重文件   : {w.name}")
    print(f"  大小       : {size:,}")

    if args.expect_size is not None:
        ok = size == args.expect_size
        print(f"  期望大小   : {args.expect_size:,}   {'OK' if ok else '**不符**'}")
        report["expected_size"] = args.expect_size
        if not ok:
            fails.append(f"大小不符: {size} != {args.expect_size}")

    if args.expect_sha256:
        print("  计算 sha256 ...", flush=True)
        got = sha256_of(w)
        ok = got == args.expect_sha256
        print(f"  sha256     : {got}")
        print(f"  期望       : {args.expect_sha256}")
        print(f"  -> {'一致 OK' if ok else '**不一致**'}")
        report["sha256"] = got
        report["expected_sha256"] = args.expect_sha256
        if not ok:
            fails.append("sha256 不符 —— 文件内容不是上游那份")

    mt = read_model_type(args.model)
    report["model_type"] = mt
    print(f"  model_type : {mt}")

    # ---------- 1) 结构：全零张量 ----------
    print()
    print("-" * 74)
    print("1) 结构：有没有全零张量")
    print("-" * 74)
    from safetensors.torch import load_file

    if w.suffix == ".safetensors":
        sd = load_file(str(w))
    else:
        sd = torch.load(str(w), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)

    zeros = [k for k in sd if sd[k].numel() and float(sd[k].float().abs().max()) == 0.0]
    print(f"  张量总数 {len(sd)}，全零 {len(zeros)}")
    report["n_tensors"] = len(sd)
    report["zero_tensors"] = zeros

    # 只有真正参与前向、且不该为零的才判失败。
    # 合法例外：LayerNorm.bias 初始化就是 0（但**加载后的**不该是 0），
    # 还有各种 head 的 decoder.bias。所以这里只对 encoder 主干严格。
    critical = [k for k in zeros
                if any(t in k for t in ("word_embeddings", "position_embeddings",
                                        "attention.self", "intermediate.dense",
                                        "output.dense"))]
    if critical:
        print(f"  **主干里有 {len(critical)} 个全零张量** —— 这就是损坏的特征：")
        for k in critical[:20]:
            print("     ZERO", k, tuple(sd[k].shape))
        if len(critical) > 20:
            print(f"     ... 还有 {len(critical)-20} 个")
        fails.append(f"主干里有 {len(critical)} 个全零张量")
    elif zeros:
        print(f"  （全零的 {len(zeros)} 个都在非主干位置，列出前 10：）")
        for k in zeros[:10]:
            print("     ", k, tuple(sd[k].shape))
    else:
        print("  没有任何全零张量 OK")

    we_keys = [k for k in sd if "word_embeddings" in k]
    we_ok = False
    for k in we_keys:
        t = sd[k].float()
        nz = int((t.abs() > 1e-8).sum())
        am = float(t.abs().max())
        print(f"  {k}: shape={tuple(sd[k].shape)} absmax={am:.6e} nnz={nz}/{t.numel()}")
        if am > 1e-6 and nz == t.numel():
            we_ok = True
    if not we_ok:
        fails.append("word_embeddings 零/非满 —— 模型看不见序列")

    # ---------- 2) 真正加载 ----------
    print()
    print("-" * 74)
    print("2) 加载（arch 自动识别 / 显式指定）")
    print("-" * 74)
    try:
        enc = RnaEncoder(str(args.model), arch=args.arch)
    except Exception as e:
        print(f"  [FAIL] 加载失败: {type(e).__name__}: {e}")
        return 1
    print("  " + enc.describe())
    report["arch"] = enc.arch.name
    report["hidden_size"] = enc.hidden_size

    # ---------- 2.5) 键覆盖：文件里的权重真的装进模型了吗 ----------
    #
    # 这一条是 2026-10-09 同一天踩的**第二个**坑：
    # 拿 ModelScope 那份做对照时，键名带 `rnafm.` 前缀没剥，encoder 全部 MISSING，
    # 于是测的是**随机初始化**的模型 —— 随机模型天然对序列敏感，
    # 我因此报了一次假的 `序列敏感: YES`。
    # 所以「敏感性」必须在**确认权重装上了**之后才成立。
    print()
    print("-" * 74)
    print("2.5) 键覆盖：checkpoint 的权重能装进模型吗")
    print("-" * 74)
    mt = report.get("model_type") or ""
    prefixes = ["", f"{mt}.", "model.", "rnafm.", "ernierna.", "rnaernie."]
    norm: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        for pfx in prefixes:
            if pfx and k.startswith(pfx):
                norm[k[len(pfx):]] = v
                break
        else:
            norm[k] = v
    msd_keys = set(enc.model.state_dict())
    covered = msd_keys & set(norm)
    uncovered = sorted(msd_keys - set(norm))
    # pooler 天然不在 checkpoint 里；各种 head 的 bias 也不参与我们的前向
    meaningful = [k for k in uncovered
                  if "pooler" not in k and not k.endswith((".bias",))]
    print(f"  模型键 {len(msd_keys)}，被 checkpoint 覆盖 {len(covered)}")
    print(f"  未覆盖 {len(uncovered)} 个: {uncovered[:8]}{' ...' if len(uncovered) > 8 else ''}")
    print(f"  其中影响前向的（剔除 pooler / *.bias）: {meaningful if meaningful else '无'}")
    report["keys_missing"] = uncovered
    report["keys_missing_meaningful"] = meaningful
    if meaningful:
        fails.append(f"checkpoint 覆盖不全，{len(meaningful)} 个权重要走随机初始化: "
                     f"{meaningful[:5]}")

    # 加载后模型里的全零张量（和读文件不同 —— load 可能补上随机初始化的）
    msd = enc.model.state_dict()
    mzeros = [k for k in msd if msd[k].numel()
              and float(msd[k].float().abs().max()) == 0.0]
    print(f"  加载后模型里的全零张量: {len(mzeros)}  {mzeros[:6]}")
    report["zero_tensors_after_load"] = mzeros
    mcrit = [k for k in mzeros if "word_embeddings" in k or "attention.self" in k]
    if mcrit:
        fails.append(f"加载后主干仍有全零张量: {mcrit[:5]}")

    # ---------- 3~5) 语义检查（核心）----------
    print()
    print("-" * 74)
    print("3) 语义检查 —— 形状检查挡不住的那个坑")
    print("-" * 74)

    rng = random.Random(0)
    nt = args.nt
    base = "".join(rng.choice("ACGU") for _ in range(nt))
    shuffled = list(base)
    rng.shuffle(shuffled)
    shuffled = "".join(shuffled)
    poly = "G" * nt                      # 同长度，组成完全不同
    # 保证打乱后确实不同
    if shuffled == base:
        shuffled = "A" + base[1:]

    grp = [base, poly, shuffled, base]
    labels = ["base", "polyG", "shuffled", "base(重复)"]
    out = enc(grp)
    h = out.hidden.float()
    report["shape"] = list(h.shape)
    print(f"  输出形状: {tuple(h.shape)}   mask 和: {out.mask.sum(-1).tolist()}")

    d_poly = float((h[0] - h[1]).abs().max())
    d_shuf = float((h[0] - h[2]).abs().max())
    d_same = float((h[0] - h[3]).abs().max())
    print(f"  base vs polyG      max|Δ| = {d_poly:.6e}   （同长度不同序列 -> 必须 > 0）")
    print(f"  base vs shuffled   max|Δ| = {d_shuf:.6e}   （同组成不同次序 -> 必须 > 0）")
    print(f"  base vs base(重复) max|Δ| = {d_same:.6e}   （确定性 -> 必须 == 0）")

    report.update({"delta_poly": d_poly, "delta_shuffled": d_shuf, "delta_repeat": d_same})

    THRESH = 1e-3
    if not d_poly > THRESH:
        fails.append(f"序列敏感性失败: 同长度不同序列 max|Δ|={d_poly:.3e} "
                     f"（<= {THRESH}，模型看不见序列）")
    if not d_shuf > THRESH:
        fails.append(f"次序敏感性失败: 打乱后 max|Δ|={d_shuf:.3e} "
                     f"（<= {THRESH}，模型只看碱基组成不看顺序）")
    if d_same != 0.0:
        fails.append(f"确定性失败: 同输入两次 max|Δ|={d_same:.3e}（应为 0）")

    # ---------- 结论 ----------
    print()
    print("=" * 74)
    if fails:
        print("VERDICT: FAIL")
        for f in fails:
            print("   ✗", f)
        print()
        print("  **不要拿这个 encoder 去训练。** 先查文件是不是坏的：")
        print("     和上游声明的大小/sha256 对一遍。")
    else:
        print("VERDICT: PASS —— encoder 健全，可以拿去训")
    print("=" * 74)

    report["verdict"] = "FAIL" if fails else "PASS"
    report["failures"] = fails
    out_path = args.out or (ROOT / "reports" / "encoder_sanity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"报告 -> {out_path}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
