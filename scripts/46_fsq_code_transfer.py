"""真正有判别力的那个测试：码的「含义」能不能跨序列迁移？

前两个指标都被否掉了：
  · within-code 方差 / 重建 err —— 循环。FSQ 是确定性映射、AE 照重构误差训的，
    within<1 是构造保证的，不是发现。
  · 覆盖率 —— 没有判别力。词表远小于训练位置数时，哈希也会给出 90% 利用率和
    95% 覆盖；词表远大于时样本不够。44_/45_ 两次都栽在这上面。

能区分的只有一个：**同一个码在没见过的序列里，是否指向同一种局部结构。**
靶子用局部 k-mer —— 它直接来自输入序列，不是 encoder 学出来的，所以不循环。

判据：
  在训练序列上学一张 code -> 该码最常见的局部 6-mer 的表，
  然后在**没见过的序列**上考：
      用码去预测该位置的 6-mer 的准确率   vs   把码随机打乱后的对照
  打乱对照保留码的边缘分布，抹掉"码<->位置"的对应关系。
  若准确率 >> 打乱对照，说明码承载了可跨序列迁移的局部结构 información，是词表。
  若 ≈ 对照，说明码到了新序列上就没有意义了 —— 接进 LLM 学不到东西。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LATENT, LEVELS = 12, 2                    # vocab 4096
VOCAB = LEVELS ** LATENT
K = 6
STEPS = 1500
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
N_SEQ, N_TEST = 256, 64


def fsq(z, levels):
    t = (torch.tanh(z) + 1) / 2
    s = t * (levels - 1)
    k = torch.round(s)
    return (k / (levels - 1) * 2 - 1) + (s - s.detach()), k.long()


class AE(nn.Module):
    def __init__(self, dim, latent, levels, hidden=256):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.levels = levels
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        q, k = fsq(self.enc(x), self.levels)
        c = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            c = c * self.levels + k[:, d]
        return self.dec(q), c


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    recs = [r for r in recs if r.get("sequence")]
    recs.sort(key=lambda r: len(r["sequence"]))
    n_all = 256
    seqs = [recs[int(i * (len(recs) - 1) / (n_all - 1))]["sequence"] for i in range(n_all)]

    d = torch.load(ROOT / "data/processed/_fsq_probe_x.pt", map_location="cpu",
                   weights_only=False)
    x, sid = d["x"].to(DEV), d["sid"].to(DEV)

    # 逐条比对：encoder 每条序列给出 L+2 个位置（[CLS] + 核苷酸 + [SEP]）。
    # 这与观测一致：115886 - 115374 = 512 = 2 x 256。
    cnt = torch.bincount(sid).tolist()
    keep = [i for i in range(len(seqs)) if cnt[i] == len(seqs[i]) + 2]
    drop = len(seqs) - len(keep)
    if drop:
        print(f"[filter] 丢弃 {drop} 条长度不匹配的序列，保留 {len(keep)}")
    remap = {o: n for n, o in enumerate(keep)}
    keep_pos = torch.tensor([s in remap for s in sid.tolist()], device=DEV)
    x = x[keep_pos]
    sid = torch.tensor([remap[s] for s in sid[keep_pos].tolist()], device=DEV)
    seqs = [seqs[i] for i in keep]
    N_SEQ = len(seqs)
    N, D = x.shape

    g = torch.Generator().manual_seed(7)
    test_ids = set(torch.randperm(N_SEQ, generator=g)[:max(1, N_SEQ // 4)].tolist())
    seq_tr = torch.tensor([s not in test_ids for s in sid.tolist()], device=DEV)

    torch.manual_seed(0)
    ae = AE(D, LATENT, LEVELS).to(DEV)
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    idx = torch.nonzero(seq_tr).squeeze(1)
    for _ in range(STEPS):
        p = idx[torch.randint(0, idx.numel(), (512,), device=DEV)]
        F.mse_loss(ae(x[p])[0], x[p]).backward()
        opt.step(); opt.zero_grad()
    ae.eval()
    with torch.no_grad():
        _, code = ae(x)

    # 构造每个位置的局部 6-mer 编号（[CLS]/[SEP] 和末尾不足 K 的位置记为 -1）
    mer_of_pos = torch.full((N,), -1, dtype=torch.long)
    pos = 0
    for s in seqs:
        b = s.encode()
        L = len(b)
        pos += 1                                   # [CLS]
        for j in range(L):
            if j + K <= L:
                v = 0
                for ch in b[j:j + K]:
                    v = v * 4 + ({65: 0, 67: 1, 71: 2, 84: 3, 85: 0}.get(ch, 0))
                mer_of_pos[pos] = v
            pos += 1
        pos += 1                                   # [SEP]
    # 位置必须与 encoder 的 mask 逐条对齐，否则 k-mer 靶子是错位的
    assert pos == N, f"位置对齐失败: 构造 {pos} != encoder 位置 {N}"
    mer_of_pos = mer_of_pos.to(DEV)

    tr_pos = torch.nonzero(seq_tr & (mer_of_pos >= 0)).squeeze(1)
    te_pos = torch.nonzero(~seq_tr & (mer_of_pos >= 0)).squeeze(1)
    print(f"[data] 训练位置 {tr_pos.numel()}  测试位置 {te_pos.numel()}  "
          f"（局部 {K}-mer，词表 {4**K}）")

    tc, tm = code[tr_pos], mer_of_pos[tr_pos].to(DEV)
    ec, em = code[te_pos], mer_of_pos[te_pos].to(DEV)

    # code -> 该码在训练序列里最常见的 6-mer
    lookup = torch.full((VOCAB,), -1, dtype=torch.long, device=DEV)
    for c in torch.unique(tc):
        sel = tm[tc == c]
        counts = torch.bincount(sel, minlength=4 ** K)
        lookup[c] = int(counts.argmax())
    hit = lookup[ec] >= 0
    acc = float((lookup[ec][hit] == em[hit]).float().mean()) if hit.any() else 0.0
    cov = float(hit.float().mean())

    # 对照：打乱 code<->位置 的对应（保留 code 边缘分布）
    g2 = torch.Generator().manual_seed(99)
    accs = []
    for _ in range(5):
        perm = torch.randperm(tc.numel(), generator=g2)
        tc2 = tc[perm]
        lk = torch.full((VOCAB,), -1, dtype=torch.long, device=DEV)
        for c in torch.unique(tc2):
            sel = tm[tc2 == c]
            lk[c] = int(torch.bincount(sel, minlength=4 ** K).argmax())
        h = lk[ec] >= 0
        accs.append(float((lk[ec][h] == em[h]).float().mean()))
    ctrl = sum(accs) / len(accs)

    # 第二个对照：全局最常见的 6-mer 常数预测
    base = float(torch.bincount(tm, minlength=4 ** K).max()) / tm.numel()

    print()
    print("=" * 78)
    print("跨序列含义迁移：用码预测局部 6-mer")
    print("=" * 78)
    print(f"  测试位置中码在训练集见过的比例: {100*cov:.1f}%")
    print(f"  按码查表预测 6-mer 准确率     : {100*acc:.2f}%")
    print(f"  打乱 code<->位置 对照          : {100*ctrl:.2f}%")
    print(f"  常数预测（训练集最常见 6-mer）: {100*base:.2f}%")
    print(f"  相对打乱对照的增益            : {100*(acc-ctrl):+.2f}pp")
    # 信息量：以比特计
    import math
    lift = acc / max(ctrl, 1e-9)
    print(f"  提升倍数                      : {lift:.2f}x")
    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    if acc > ctrl * 1.5 and acc > base * 1.3:
        print(f"  ✅ **码携带可跨序列迁移的局部结构信息**（{100*acc:.1f}% vs 打乱 {100*ctrl:.1f}%，"
              f"{lift:.1f}x）。")
        print("     同一个码在没见过的序列里指向同一类 6-mer —— 这是词表，不是哈希。")
        print("     接进 LLM 有可学的东西。")
    elif acc > ctrl * 1.15:
        print(f"  ⚠️ **弱迁移**：{100*acc:.1f}% vs 打乱 {100*ctrl:.1f}%（{lift:.2f}x）。")
        print("     码有结构但信噪比低。可能要靠加复用压力（commitment / 熵正则）才能变干净。")
    else:
        print(f"  ❌ **码到新序列上就没有含义了**（{100*acc:.1f}% vs 打乱 {100*ctrl:.1f}%，"
              f"{lift:.2f}x）。")
        print("     码只是把每条序列切成一串各自不同的片段 —— 对 LLM 而言是新的乱码。")
        print("     要救这条路线，必须换目标函数（加复用压力），而不是换个量化方式。")

    out = ROOT / "reports" / "fsq_code_transfer.json"
    out.write_text(json.dumps(dict(vocab=VOCAB, k=K, acc=acc, ctrl=ctrl, base=base,
                                   cov=cov, lift=lift), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
