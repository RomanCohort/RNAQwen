"""FSQ 的码是"词表"还是"哈希"—— v2，带样本量与随机基线控制

v1 的错：64 条序列 29k 位置，词表 65536 -> 平均每码 2.6 个位置，
        中位重用次数必然是 1。"52% 的码只出现一次"是**词表比样本大**的算术必然，
        不是关于 FSQ 的任何发现。

v2 的两个判据：
  ① 扫词表大小，看单次出现占比随「位置/码」比值怎么变。
     若在所有尺度下它都是"哈希"，曲线会一直贴 1；若是词表，会随比值塌下来。
  ② 留出序列覆盖率 + 随机基线。
     在 192 条上训，在 64 条没见过的序列上测：多少位置落在训练见过的码上。
     基线 = Σ_c p_train(c)·p_test(c)（同边缘分布的随机对照）。
     哈希 -> 实测≈基线；真词表 -> 实测 >> 基线。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rnaqwen.models.encoder import RnaEncoder  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SEQ, STEPS = 256, 1500
N_TEST = 64
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
# (latent, levels) -> vocab
CFGS = [(8, 2), (10, 2), (8, 3), (12, 2), (6, 4), (8, 4)]


def fsq(z, levels):
    t = (torch.tanh(z) + 1) / 2
    s = t * (levels - 1)
    k = torch.round(s)
    q = k / (levels - 1) * 2 - 1
    return q + (s - s.detach()), k.long()


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
        code = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            code = code * self.levels + k[:, d]
        return self.dec(q), code


def boot_baseline(tc, n_test, k, vocab, reps=8):
    """从训练边缘分布独立采样 n_test 个码，统计其中"训练计数>=k"的比例。"""
    p = tc / tc.sum()
    g = torch.Generator(device="cpu").manual_seed(1234)
    tot = 0.0
    for _ in range(reps):
        s = torch.multinomial(p.cpu(), n_test, replacement=True, generator=g).to(tc.device)
        tot += float(((tc[s] >= k).float()).mean())
    return tot / reps


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    recs = [r for r in recs if r.get("sequence")]
    recs.sort(key=lambda r: len(r["sequence"]))
    pick = [recs[int(i * (len(recs) - 1) / (N_SEQ - 1))] for i in range(N_SEQ)]
    seqs = [r["sequence"] for r in pick]

    CACHE = ROOT / "data/processed/_fsq_probe_x.pt"
    if CACHE.exists():
        d = torch.load(CACHE, map_location="cpu", weights_only=False)
        x, sid = d["x"].to(DEV), d["sid"].to(DEV)
        print(f"[cache] {CACHE.name}")
    else:
        enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()
        # 分批过编码器：256x1024 的 fp32 attention 单层就要 ~13GB，会直接 abort
        xs, sids = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                chunk = seqs[i:i + 16]
                o = enc(chunk)
                h = o.hidden.float()
                mk = (o.mask.squeeze(-1) > 0)
                if mk.shape[1] != h.shape[1]:
                    mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
                xs.append(h[mk].cpu())
                sids.append(torch.repeat_interleave(
                    torch.arange(i, i + len(chunk)), mk.sum(1).cpu()))
                del o, h, mk
        x = torch.cat(xs).to(DEV)
        sid = torch.cat(sids).to(DEV)
        del xs, sids
        torch.save(dict(x=x.cpu(), sid=sid.cpu()), CACHE)
        del enc
        import gc; gc.collect(); torch.cuda.empty_cache()
    N, D = x.shape
    # 随机切分（按序列）—— 按长度切会变成"长度外推"实验，不是词表实验
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(seqs), generator=g)
    test_ids = set(perm[:N_TEST].tolist())
    tr = torch.tensor([s not in test_ids for s in sid.tolist()], device=DEV)
    te = ~tr
    print(f"[data] {len(seqs)} 条序列 / {N} 位置  训练除 {int(tr.sum())} 测试 {int(te.sum())}")
    print(f"       测试位占比 {100*float(te.float().mean()):.1f}%")

    print()
    print("=" * 84)
    print("① 重用分布随「位置/码」比值的变化（同一批数据，只换词表）")
    print("=" * 84)
    print(f"{'词表':>8} {'lat':>4} {'l':>3} {'用码数':>8} {'用码%':>7} {'位置/用码':>10} "
          f"{'中位重用':>9} {'仅1次%':>8} {'词内方v':>8} {'err':>7}")
    rows = []
    preds = {}
    for lat, lv in CFGS:
        vocab = lv ** lat
        torch.manual_seed(0)
        ae = AE(D, lat, lv).to(DEV)
        opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
        idx_tr = torch.nonzero(tr).squeeze(1)
        for _ in range(STEPS):
            p = idx_tr[torch.randint(0, idx_tr.numel(), (512,), device=DEV)]
            xb = x[p]
            rec, _ = ae(xb)
            F.mse_loss(rec, xb).backward()
            opt.step(); opt.zero_grad()
        ae.eval()
        with torch.no_grad():
            rec, code = ae(x)
            err = float(((rec - x) ** 2).sum(1).mean() / float(x.var(dim=0).sum()))
        cnt = torch.bincount(code, minlength=vocab)
        used = int((cnt > 0).sum())
        med = float(cnt[cnt > 0].float().median())
        once = float((cnt == 1).sum()) / max(used, 1)
        sums = torch.zeros(vocab, D, device=DEV)
        sums.index_add_(0, code, x)
        means = sums / cnt.clamp(min=1).unsqueeze(1)
        within = float(((x - means[code]) ** 2).sum(1).mean() / float(x.var(dim=0).sum()))
        print(f"{vocab:>8} {lat:>4} {lv:>3} {used:>8} {100*used/vocab:>6.1f}% "
              f"{N/used:>10.1f} {med:>9.0f} {100*once:>7.1f}% {within:>8.3f} {err:>7.4f}")
        rows.append(dict(vocab=vocab, latent=lat, levels=lv, used=used, median=med,
                         once=once, within=within, err=err, pos_per_code=N / used))
        preds[vocab] = (code, tr, te)

    print()
    print("=" * 84)
    print("② 留出序列覆盖率（192 训 / 64 测）+ 随机基线")
    print("=" * 84)
    print(f"{'词表':>8} {'位置/码':>9} | {'>=1次':>8} {'基线':>8} {'超出':>7} | "
          f"{'>=5次':>8} {'基线':>8} {'超出':>7} | {'>=20次':>8} {'基线':>8}")
    verdict = []
    for r in rows:
        vocab = r["vocab"]
        code, tr, te = preds[vocab]
        tc_ = code[tr]; te_ = code[te]
        tcnt = torch.bincount(tc_, minlength=vocab).float()
        ecnt = torch.bincount(te_, minlength=vocab).float()
        pet = ecnt / ecnt.sum()
        line = f"{vocab:>8} {r['pos_per_code']:>9.1f} |"
        ex1 = ex5 = 0.0
        for k in (1, 5):
            cov = float(pet[tcnt >= k].sum())
            base = boot_baseline(tcnt, int(te_.numel()), k, vocab)
            ex = cov - base
            if k == 1:
                ex1 = ex
            else:
                ex5 = ex
            line += f" {100*cov:>7.1f}% {100*base:>7.1f}% {100*ex:>+6.1f}% |"
        cov20 = float(pet[tcnt >= 20].sum())
        base20 = boot_baseline(tcnt, int(te_.numel()), 20, vocab)
        line += f" {100*cov20:>7.1f}% {100*base20:>7.1f}%"
        print(line)
        verdict.append((vocab, r["pos_per_code"], ex1, ex5, cov20, base20, r["within"]))

    print()
    print("=" * 84)
    print("判定")
    print("=" * 84)
    print("  哈希的签名：超出基线 ≈ 0（见过与否和序列是不是新的无关）。")
    print("  词表的签名：超出基线 >> 0，且随词表变小而增大。")
    print()
    ok = [v for v in verdict if v[3] > 0.05]     # >=5 次覆盖超出 > 5pp
    if not ok:
        print("  ❌ **全部配置都接近哈希**。用量化离散码接进 LLM 没有可泛化的东西可学。")
    else:
        print("  ✅ 以下配置超出基线显著，码是有泛化性的：")
        for vocab, ppc, ex1, ex5, c20, b20, wi in ok:
            print(f"     词表 {vocab:>6}（位置/码 {ppc:>5.1f}）: >=5次覆盖超出 {100*ex5:+.1f}pp，"
                  f"词内方差 {wi:.3f}")
        best = max(ok, key=lambda v: v[3])
        print()
        print(f"  建议：**词表 {best[0]}**（位置/码 {best[1]:.1f}），"
              f">=5次覆盖超出 {100*best[3]:+.1f}pp。")

    out = ROOT / "reports" / "fsq_vocab_or_hash.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, n_seq=N_SEQ, n_pos=N,
                                   n_test_seqs=N_TEST), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
