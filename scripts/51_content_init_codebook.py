"""按内容初始化码表：把每个码的初始向量设成它代表的 RNA 内容的投影。

为什么这是缺的那一件：
    `rna_emb` 原本是 N(0, 0.02) 随机初始化 —— 第 0 步时每条 RNA 拿到的是一串
    无意义的 4096 维随机向量。模型在这种输入上唯一能学会的就是**忽略它**。
    （证据：47_ 漂移 12.5% 时 vs零表 22.4% 说明位置上有东西在被用，
      但 vs随机表 59.5% 说明那个东西的内容是噪声。）
    文献 2601.22244 的 "initialization from data" 就是治这个的，之前列了没做。

做法：
    1. 用已训好的 FSQ，把全部 201 万个位置编码成码
    2. 对每个码 c，取所有映射到 c 的 ERNIE-RNA 隐藏态的均值 mu_c (768 维)
    3. 用一个固定的随机投影 P (768 -> 4096)，令 rna_emb[c] = P @ normalize(mu_c)
       并缩放到与码表初始规模同量级
    4. head 也一并初始化，让 head(q_c) ≈ rna_emb[c]

自检（不训练，1 分钟）：
    同一个 6-mer 的两处，向量余弦应显著高于随机两处。
    若没有，说明初始化没带来结构，别往下走。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LATENT, LEVELS = 12, 2      # 占位，实际从检查点读（见 main）
VOCAB = LEVELS ** LATENT
D_MODEL = 4096
HID = ROOT / "data/processed/_fsq_full_pos.pt"
FSQ_CKPT = ROOT / "checkpoints/fsq_full/tokenizer.pt"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
OUT = ROOT / "checkpoints/fsq_full/code_emb_contentinit.pt"


def fsq_codes(enc_w, enc_b, x, levels=LEVELS):
    z = x @ enc_w.t() + enc_b
    t = torch.tanh(z)
    k = torch.round((t + 1) / 2 * (levels - 1)).long()
    c = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for d in range(k.shape[1]):
        c = c * levels + k[:, d]
    return c


def main() -> int:
    # global 必须在函数**最前面**：下面的 default=D_MODEL 会先读它，
    # 放到后面就报 "name 'D_MODEL' is used prior to global declaration"
    global D_MODEL, HID, FSQ_CKPT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=D_MODEL,
                    help="目标底座的 hidden_size。8B=4096，Qwen3.6-35B-A3B=2048")
    ap.add_argument("--tgt-norm", type=float, default=1.28,
                    help="码表行范数的目标值。**应对齐底座 embed_tokens 的实测行范数**："
                         "Qwen3-8B 实测 1.3758（原来写 1.28 是 N(0,0.02) 的理论值，差 7%%）；"
                         "Qwen3.6-35B-A3B 实测 0.5547")
    ap.add_argument("--out", default="")
    ap.add_argument("--hid", default="", help="换隐藏态缓存（默认 _fsq_full_pos.pt）")
    ap.add_argument("--fsq", default="",
                    help="换码表检查点（默认 checkpoints/fsq_full/tokenizer.pt）；"
                         "换成 fsq_stream/tokenizer.pt 时必须同时换一个 --out")
    args = ap.parse_args()
    D_MODEL = args.d_model
    if args.hid:
        HID = Path(args.hid)
    if args.fsq:
        FSQ_CKPT = Path(args.fsq)
    out_p = Path(args.out) if args.out else OUT
    print(f"[cfg] 目标 d_model={D_MODEL}  行范数目标={args.tgt_norm}  "
          f"隐藏态 {HID.name}  码表 {FSQ_CKPT.name}  -> {out_p}")

    d = torch.load(HID, map_location="cpu", weights_only=False)
    x, lens, offsets = d["x"], d["lens"], d["offsets"]
    ck = torch.load(FSQ_CKPT, map_location="cpu", weights_only=False)
    # 从检查点读 latent/levels，别再硬编码 —— 硬编码是第 5 轮那个陈旧产物坑的来源
    global LATENT, LEVELS, VOCAB
    LATENT, LEVELS = int(ck["latent"]), int(ck["levels"])
    VOCAB = LEVELS ** LATENT
    print(f"[cfg] 从检查点读：LATENT={LATENT} LEVELS={LEVELS} VOCAB={VOCAB}")
    ew, eb = ck["state"]["enc.weight"].float(), ck["state"]["enc.bias"].float()
    print(f"[in] 位置 {tuple(x.shape)}   FSQ enc {tuple(ew.shape)}")

    # 分块编码成码，并累加每个码的隐藏态均值
    sums = torch.zeros(VOCAB, x.shape[1], dtype=torch.float64)
    cnt = torch.zeros(VOCAB, dtype=torch.long)
    step = 200_000
    for i in range(0, x.shape[0], step):
        xb = x[i:i + step].to(DEV).float()
        c = fsq_codes(ew.to(DEV), eb.to(DEV), xb).cpu()
        sums.index_add_(0, c, xb.cpu().double())
        cnt.index_add_(0, c, torch.ones_like(c, dtype=torch.long))
        del xb, c
    used = cnt > 0
    print(f"[codes] 用到的码 {int(used.sum())}/{VOCAB}   位置 {int(cnt.sum())}")
    mu = torch.zeros(VOCAB, x.shape[1])
    mu[used] = (sums[used] / cnt[used].unsqueeze(1).double()).float()

    # 未见过的码：用全局均值（不该出现，但要有个定义）
    miss = ~used
    if miss.any():
        mu[miss] = (sums[used].sum(0) / cnt[used].sum().double()).float()

    # 固定随机投影 768 -> 4096，按 Qwen 词表行的量级缩放
    g = torch.Generator().manual_seed(0)
    P = torch.randn(D_MODEL, x.shape[1], generator=g) / (x.shape[1] ** 0.5)
    v = mu @ P.t()                                    # (VOCAB, 4096)
    v = v - v.mean(0, keepdim=True)                   # 去掉公共分量，避免所有码同向
    tgt = args.tgt_norm          # 对齐底座 embed_tokens 的实测行范数（不是理论值）
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-6) * tgt

    # ---- 自检：同 6-mer 的两处，向量余弦应高于随机两处 ----
    print("\n=== 自检：初始化有没有带来结构 ===")
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                s = r.get("sequence") or ""
                if s and len(s) <= 1024:
                    recs.append(s)
    K = 6
    mers = []
    for s in recs:
        b = s.encode()
        L = len(b)
        vv = torch.full((L,), -1, dtype=torch.long)
        for j in range(L - K + 1):
            a = 0
            for ch in b[j:j + K]:
                a = a * 4 + ({65: 0, 67: 1, 71: 2, 84: 3, 85: 0}.get(ch, 0))
            vv[j] = a
        mers.append(vv)
    mers = torch.cat(mers)
    assert mers.numel() == x.shape[0], f"对齐失败 {mers.numel()} vs {x.shape[0]}"

    code = torch.empty(x.shape[0], dtype=torch.long)
    for i in range(0, x.shape[0], step):
        code[i:i + step] = fsq_codes(ew, eb, x[i:i + step].float())
    ok = mers >= 0
    m, c = mers[ok], code[ok]
    g2 = torch.Generator().manual_seed(1)
    sel = torch.randperm(m.numel(), generator=g2)[:8000]
    m, c = m[sel], c[sel]
    vn = torch.nn.functional.normalize(v[c], dim=1)
    # 同一个 6-mer 的两处（跨序列，位置要远）
    same = (m.unsqueeze(0) == m.unsqueeze(1)) & (torch.arange(len(m)).unsqueeze(0)
            != torch.arange(len(m)).unsqueeze(1))
    idx = torch.nonzero(same)
    if idx.numel() == 0:
        print("  没找到同 6-mer 的对，跳过")
    else:
        g3 = torch.Generator().manual_seed(2)
        pick = idx[torch.randperm(len(idx), generator=g3)[:20_000]]
        cos_same = (vn[pick[:, 0]] * vn[pick[:, 1]]).sum(1).mean()
        rp = torch.randint(0, len(m), (20_000, 2), generator=g3)
        cos_rand = (vn[rp[:, 0]] * vn[rp[:, 1]]).sum(1).mean()
        print(f"  同 6-mer 两处的余弦   {float(cos_same):+.4f}")
        print(f"  随机的余弦            {float(cos_rand):+.4f}")
        print(f"  差（结构信号）        {float(cos_same-cos_rand):+.4f}")
        if float(cos_same) > float(cos_rand) + 0.02:
            print("  ✅ 初始化带来了结构 —— 同 6-mer 的向量确实更近")
        else:
            print("  ❌ 初始化没带来结构，先别训")

    torch.save(dict(code_emb=v, mu=mu, latent=LATENT, levels=LEVELS,
                    d_model=D_MODEL, mode="content_init"), out_p)
    print(f"\n[save] {out_p}   {tuple(v.shape)}   "
          f"行范数均值 {float(v.norm(dim=1).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
