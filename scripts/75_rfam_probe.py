"""① 补测 rfam 的探针准确率 —— 和 seq8 同一套方法。

问题：那 8 个 seq8 模板是**筛出来的**（探针 >= 0.446）。rfam 的 2 个模板从来没测过。
现在合并数据里带着它们，等于在没验证的前提下把新任务塞进去。

方法（对齐 §10 的 code_answer_probe）：
    1. 用训练好的 FSQ（runs/B_seq8/artifacts/fsq.pt）把 rfam 的隐藏态编码成码
       —— 含 pool=4 的学习式压缩，和 LLM 看到的完全一致
    2. 特征 = 每条序列的码直方图（4096 维）
    3. 在训练序列上拟合线性分类器（码直方图 -> 家族标签）
    4. 在没见过的序列上考
    5. 对照：众数地板（永远答训练集最常见的家族）

判据：probe_acc >= 0.40 才留（和 seq8 的筛选门槛一致）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HID = ROOT / "data/processed/_rfam_hidden.f16"
IDX = ROOT / "data/processed/_rfam_index.pt"
QA = ROOT / "data/processed/rnaqa_rfam.jsonl"
ART = ROOT / "runs/B_seq8/artifacts"
POOL = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ART))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    art = ROOT / args.artifacts
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    VOC = LEV ** LAT
    print(f"[fsq] {art.name}  LATENT={LAT} LEVELS={LEV} VOCAB={VOC}")

    d = torch.load(IDX, map_location="cpu", weights_only=False)
    uids = d["uids"]
    lens, offs = d["lens"], d["offsets"]
    mm = np.memmap(HID, dtype=np.float16, mode="r", shape=(int(d["n_pos"]), int(d["dim"])))
    x_all = torch.from_numpy(mm)
    print(f"[hid] {tuple(mm.shape)}  序列 {len(uids):,}")

    # 每条序列的家族标签 + split（来自 rfam QA，每条序列两个模板但标签相同）
    label, split = {}, {}
    with open(QA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            label[r["uid"]] = r["answer"]
            split[r["uid"]] = r.get("split", "train")
    # 只用「Which Rfam family...」那条，别混两个模板
    fam = {}
    with open(QA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["question"].startswith("Which Rfam"):
                fam[r["uid"]] = (r["answer"], r.get("split", "train"))
    print(f"[qa]  有家族标签的序列 {len(fam):,}")

    # 只处理有标签、且在缓存里的
    todo = [(i, u) for i, u in enumerate(uids) if u in fam]
    print(f"[join] 缓存 ∩ 标签 = {len(todo):,}")
    if len(todo) < 100:
        print("**太少** —— 检查 uid 格式是否一致")
        # 看看两边的样例
        print("  缓存 uid 例:", uids[:3])
        print("  标签 uid 例:", list(fam)[:3])
        return 1

    # ---- 编码成码 + 直方图 ----
    class Comp:
        """只用 fsq 的 compress + enc，拿到 code id。"""
        def __init__(s, ck, lat, lev):
            s.lat, s.lev = lat, lev
            import torch.nn as nn
            s.compress = nn.Linear(POOL * ck["dim"], ck["dim"], bias=False)
            s.enc = nn.Linear(ck["dim"], lat)
        def to(s, dev):
            s.compress.to(dev).eval(); s.enc.to(dev).eval(); return s
        @torch.no_grad()
        def __call__(s, x):
            n = x.shape[0] // POOL
            if n == 0:
                return None
            x = x[:n * POOL]
            x = s.compress(x.reshape(n, POOL * x.shape[1]))
            t = torch.tanh(s.enc(x))
            k = torch.round((t + 1) / 2 * (s.lev - 1)).long()
            c = torch.zeros(k.shape[0], dtype=torch.long, device=k.device)
            for dd in range(k.shape[1]):
                c = c * s.lev + k[:, dd]
            return c

    ck = dict(fsq_ck["state"]); ck["dim"] = 768
    comp = Comp(ck, LAT, LEV).to(DEV)
    comp.compress.weight.data.copy_(fsq_ck["state"]["compress.weight"])
    comp.enc.weight.data.copy_(fsq_ck["state"]["enc.weight"])
    comp.enc.bias.data.copy_(fsq_ck["state"]["enc.bias"])

    t0 = time.time()
    feats, labels, splits = [], [], []
    for k, (i, u) in enumerate(todo):
        off, ln = int(offs[i]), int(lens[i])
        x = x_all[off:off + ln].to(DEV, dtype=torch.float32)
        c = comp(x)
        if c is None or c.numel() == 0:
            continue
        h = torch.bincount(c, minlength=VOC).float()
        feats.append(h)
        labels.append(fam[u][0])
        splits.append(fam[u][1])
        if (k + 1) % 4000 == 0:
            print(f"  [enc] {k+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)

    X = torch.stack(feats)
    print(f"[feat] {tuple(X.shape)}   用时 {time.time()-t0:.0f}s")

    tr = torch.tensor([s == "train" for s in splits])
    te = ~tr
    print(f"[split] train {int(tr.sum()):,}  test {int(te.sum()):,}")
    if int(te.sum()) < 50:
        print("**测试集太小**"); return 1

    yv = sorted(set(labels))
    y2i = {v: i for i, v in enumerate(yv)}
    y = torch.tensor([y2i[v] for v in labels])
    print(f"[label] {len(yv):,} 个不同家族")

    # ---- 众数地板 ----
    cnt_tr = Counter(y[tr].tolist())
    mode = cnt_tr.most_common(1)[0][0]
    floor = float((y[te] == mode).float().mean())
    print(f"[floor] 永远答训练集最常见家族 = {100*floor:.2f}%")

    # ---- 线性探针 ----
    Xn = X / X.sum(1, keepdim=True).clamp(min=1)
    W = torch.zeros(len(yv), X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(len(yv), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=args.lr)
    Xtr, ytr = Xn[tr].to(DEV), y[tr].to(DEV)
    Xte, yte = Xn[te].to(DEV), y[te].to(DEV)
    for ep in range(args.epochs):
        logit = Xtr @ W.t() + b
        loss = torch.nn.functional.cross_entropy(logit, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                a = float((torch.cat([logit.argmax(1), (Xte @ W.t() + b).argmax(1)])
                           == torch.cat([ytr, yte])).float().mean())
            print(f"  ep {ep+1:>4}  loss {float(loss):.4f}  合计 acc {100*a:.2f}%", flush=True)

    with torch.no_grad():
        acc = float(((Xte @ W.t() + b).argmax(1) == yte).float().mean())
    print()
    print("=" * 70)
    print(f"  探针准确率（没见过的序列）  **{100*acc:.2f}%**")
    print(f"  众数地板                     {100*floor:.2f}%")
    print(f"  提升                         {100*(acc-floor):+.2f} pp")
    print("=" * 70)
    print()
    print("  判据：probe_acc >= 0.40 才留（和 seq8 同一门槛）")
    if acc >= 0.40:
        print(f"  ✅ **{100*acc:.1f}% >= 40%  —— rfam 模板该留**")
    elif acc > floor + 0.10:
        print(f"  ⚠️ {100*acc:.1f}% 明显高于地板但不到 40% —— 有信号，弱于 seq8 的 8 个模板")
    else:
        print(f"  ❌ {100*acc:.1f}% 接近地板 —— **和那 22 个计数模板同类，不该进训练**")

    (ROOT / "reports/rfam_probe.json").write_text(json.dumps(dict(
        artifacts=args.artifacts, n_seq=len(labels), n_train=int(tr.sum()),
        n_test=int(te.sum()), n_classes=len(yv), probe_acc=acc, floor=floor,
        lift=acc - floor, verdict=("keep" if acc >= 0.40 else "drop")),
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
