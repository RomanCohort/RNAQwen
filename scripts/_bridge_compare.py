"""桥接方式对照实验：FSQ 离散码 vs ChatNT 式问题感知投影。

问题
----
当前路线把 ERNIE-RNA 的 768 维输出经 FSQ 量化成离散码（每 4 个位置 1 个码），
再查表 rna_emb[code] 变成 LLM 的输入向量。ChatNT (Nat Mach Intell 2025) 走的是
另一条路：ERNIE/NT 的连续 hidden → **问题感知的 Perceiver resampler** → 固定
数量的软提示 token。

两者都是「把序列接进 LLM」，但信息瓶颈不同：
  FSQ      —— 码完全固定，同一条序列永远给同一串码，与问题无关
  ChatNT   —— 投影层看得到英文问题，可以按问题抽取相关特征
论文实测：普通 Perceiver 0.71 MCC → 问题感知 0.77 MCC。

做法
----
不训练 35B。用同一份 ERNIE hidden + 同一份候选式题目，只对比「桥」：
训练一个小模型，把 (序列表示, 问题) 映射到 4 选 1 的答案字母。

  A 臂：FSQ 码 → 嵌入表 → 小 Transformer → 答案
  B 臂：连续 hidden → Perceiver resampler（问题感知）→ 小 Transformer → 答案

这样能在几分钟内看出「桥本身够不够用」，而不是等 35B 跑 10 小时。

用法
    python scripts/_bridge_compare.py --n-train 8000 --n-test 1500 --epochs 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 数据

def load_rows(path: Path, split: str, n: int, seed: int):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split and r.get("kind") == "candidate_family":
                rows.append(r)
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------- 两条桥

class FSQBridge(nn.Module):
    """A 臂：离散码 -> 嵌入表。码与问题无关（这正是要检验的瓶颈）。"""

    def __init__(self, n_codes=4096, dim=256):
        super().__init__()
        self.emb = nn.Embedding(n_codes, dim)
        self.dim = dim

    def forward(self, code_ids, q_vec):
        # 码取平均，问题用简单相加（模拟"同一串码服务所有问题"）
        h = self.emb(code_ids)              # [B, L, dim]
        return h.mean(1) + q_vec            # [B, dim]


class ChatNTBridge(nn.Module):
    """B 臂：连续 hidden -> 问题感知 resampler（gated cross-attention）。"""

    def __init__(self, enc_dim=768, dim=256, n_query=8, n_head=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, enc_dim)
        self.query = nn.Parameter(torch.randn(n_query, enc_dim) * 0.02)
        self.attn = nn.MultiheadAttention(enc_dim, n_head, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(enc_dim, dim)
        self.n_query = n_query
        self.dim = dim

    def forward(self, hidden, mask, q_vec):
        # hidden [B,L,enc_dim], mask [B,L], q_vec [B,dim]
        q = self.query.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        q = q + self.q_proj(q_vec).unsqueeze(1)     # 问题感知
        kpm = ~mask.bool()
        a, _ = self.attn(q, hidden, hidden, key_padding_mask=kpm)
        a = torch.sigmoid(self.gate) * a
        return self.out(a).mean(1)                  # [B, dim]


class Head(nn.Module):
    """两条桥共用的小分类头：向量 + 问题 -> A/B/C/D。"""

    def __init__(self, dim, n_opt=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, n_opt))

    def forward(self, v):
        return self.net(v)


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/rnaqa_v3_cand.jsonl")
    ap.add_argument("--encoder", default="models/ernierna")
    ap.add_argument("--fsq", default="checkpoints/fsq_stream/tokenizer.pt")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tr = load_rows(ROOT / args.data, "train", args.n_train, args.seed)
    te = load_rows(ROOT / args.data, "test", args.n_test, args.seed + 1)
    print(f"[data] train {len(tr)}  test {len(te)}", flush=True)

    # ---- 编码一次，两条桥共用 ----
    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / args.encoder), dtype=torch.float32).to(DEV).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"[enc] {args.encoder}  hidden {enc.hidden_size}", flush=True)

    # FSQ（冻结）
    import importlib.util as _iu
    spec = _iu.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
    rc = _iu.module_from_spec(spec)
    spec.loader.exec_module(rc)
    ck = torch.load(ROOT / args.fsq, map_location="cpu", weights_only=False)
    LAT, LEV = int(ck["latent"]), int(ck["levels"])
    # 流式码表结构：enc(768->12) 直接量化，池化在编码前做（没有 compress 层）
    enc_w = ck["state"]["enc.weight"].float().to(DEV)
    enc_b = ck["state"]["enc.bias"].float().to(DEV)

    @torch.no_grad()
    def fsq_encode(h):          # h [N, 768] -> code ids [N]
        t = torch.tanh(F.linear(h, enc_w, enc_b))
        s = (t + 1) / 2 * (LEV - 1)
        k = torch.round(s).long()
        code = torch.zeros(k.shape[0], dtype=torch.long, device=k.device)
        for d in range(LAT):
            code = code * LEV + k[:, d]
        return code

    # 全局固定长度，避免不同 batch 的 L 不一致导致 cat 失败
    MAXL = 1026

    def pad_to(t, maxl):
        B, L = t.shape[0], t.shape[1]
        if L >= maxl:
            return t[:, :maxl]
        pad = t.new_zeros(B, maxl - L, *t.shape[2:])
        return torch.cat([t, pad], 1)

    @torch.no_grad()
    def featurize(rows, tag):
        hid_all, mask_all, code_all, y_all = [], [], [], []
        B = 32
        for i in range(0, len(rows), B):
            chunk = rows[i:i + B]
            seqs = [r["sequence"][:1024] for r in chunk]
            o = enc(seqs)
            h, m = o.hidden.float(), o.mask.float()
            # bf16 存 hid，省一半内存（ERNIE 输出本来就是 fp32，bf16 足够）
            hid_all.append(pad_to(h, MAXL).to(torch.bfloat16).cpu())
            mask_all.append(pad_to(m, MAXL).to(torch.bool).cpu())
            for j, r in enumerate(chunk):
                valid = h[j][m[j].bool()]
                if valid.shape[0] < 4:
                    code_all.append(torch.zeros(1, dtype=torch.long))
                else:
                    n = valid.shape[0] // 4 * 4        # pool=4
                    pooled = valid[:n].reshape(n // 4, 4, -1).mean(1)
                    code_all.append(fsq_encode(pooled).cpu())
                y_all.append("ABCD".index(r["answer"]))
            if (i // B) % 50 == 0:
                print(f"  [{tag}] {i+len(chunk)}/{len(rows)}", flush=True)
        return (torch.cat(hid_all), torch.cat(mask_all),
                code_all, torch.tensor(y_all))

    # 特征缓存：编码一次要 ~5 分钟，反复调参时不该重算
    cache = ROOT / "tmp" / f"bridge_feats_{len(tr)}_{len(te)}_{args.seed}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cache.exists():
        tr_hid, tr_mask, tr_code, tr_y, te_hid, te_mask, te_code, te_y = \
            torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[feat] 命中缓存 {cache.name}", flush=True)
    else:
        tr_hid, tr_mask, tr_code, tr_y = featurize(tr, "train")
        te_hid, te_mask, te_code, te_y = featurize(te, "test")
        torch.save((tr_hid, tr_mask, tr_code, tr_y,
                    te_hid, te_mask, te_code, te_y), cache)
        print(f"[feat] {time.time()-t0:.0f}s  已缓存 {cache.name}", flush=True)

    def pad_codes(codes, maxlen=256):
        out = torch.zeros(len(codes), maxlen, dtype=torch.long)
        for i, c in enumerate(codes):
            n = min(len(c), maxlen)
            out[i, :n] = c[:n]
        return out

    tr_code_p = pad_codes(tr_code).to(DEV)
    te_code_p = pad_codes(te_code).to(DEV)

    # 问题向量：按**问题模板**做确定性投影，而不是纯随机。
    # 候选题的 question 前缀都一样（只有选项不同），如果直接用随机向量，
    # B 臂的「问题感知」就只是噪声。这里把所有出现过的 question 模板
    # 映射到一组固定的随机向量，同一模板永远得到同一个向量 —— 这样
    # 「看到问题」至少是稳定的条件信息，两条桥的比较才公平。
    q_dim = 256
    _seen_q: dict[str, int] = {}
    def qid(q: str) -> int:
        return _seen_q.setdefault(q, len(_seen_q))

    tr_ids = [qid(r["question"]) for r in tr]
    te_ids = [qid(r["question"]) for r in te]
    n_q = max(len(_seen_q), 1)
    g = torch.Generator().manual_seed(1234)
    table = torch.randn(n_q, q_dim, generator=g)
    tr_q = table[torch.tensor(tr_ids)].to(DEV)
    te_q = table[torch.tensor(te_ids)].to(DEV)
    print(f"[q] {n_q} 个问题模板", flush=True)
    tr_y, te_y = tr_y.to(DEV), te_y.to(DEV)

    results = {}
    for name, Bridge in [("FSQ(码)", FSQBridge), ("ChatNT(问题感知)", ChatNTBridge)]:
        print(f"\n===== {name} =====", flush=True)
        bridge = Bridge().to(DEV)
        head = Head(256).to(DEV)
        opt = torch.optim.AdamW(list(bridge.parameters()) + list(head.parameters()),
                                lr=args.lr, weight_decay=0.01)
        n = len(tr)
        best = 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=DEV)
            bridge.train(); head.train()
            tot, cnt = 0.0, 0
            for i in range(0, n, args.batch):
                idx = perm[i:i + args.batch]
                ic = idx.cpu()          # CPU 张量必须用 CPU 索引
                if name.startswith("FSQ"):
                    v = bridge(tr_code_p[idx], tr_q[idx])
                else:
                    v = bridge(tr_hid[ic].to(DEV).float(),
                               tr_mask[ic].to(DEV), tr_q[idx])
                logits = head(v)
                loss = F.cross_entropy(logits, tr_y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(idx); cnt += len(idx)
            # eval
            bridge.eval(); head.eval()
            hit = 0
            with torch.no_grad():
                for i in range(0, len(te), 256):
                    sl = slice(i, i + 256)
                    if name.startswith("FSQ"):
                        v = bridge(te_code_p[sl], te_q[sl])
                    else:
                        v = bridge(te_hid[sl].to(DEV).float(),
                                   te_mask[sl].to(DEV), te_q[sl])
                    hit += int((head(v).argmax(-1) == te_y[sl]).sum())
            acc = hit / len(te)
            best = max(best, acc)
            print(f"  ep{ep+1}  loss {tot/cnt:.4f}  test acc {100*acc:.2f}%  "
                  f"(随机 25%)", flush=True)
        results[name] = best

    print("\n" + "=" * 60)
    print("最佳 test 准确率（4 选 1，随机 25%）")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<22} {100*v:.2f}%")
    outp = ROOT / "reports" / "bridge_compare.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
