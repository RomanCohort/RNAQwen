"""码信道的**天花板**：只看码序列，30 个模板能答对多少？

为什么必须先量这个
------------------
66_ 证明了码表没塌（全量 3885/4096）、跨序列 6-mer 迁移 7.79x。但那只说明
「码携带局部结构」，不说明「码里装着这批问答的答案」。

于是有两个完全不同的岔路，再多训几步也分不开：
  (a) 码里有答案、LLM 没学会读   -> 动 LLM 侧（对齐 / 继续预训练）
  (b) 码里根本没有答案           -> 动 tokenizer（别池化 4x / 换目标）

这个脚本用**不需要 LLM** 的探针把两者分开：在**训练序列**上拟合「码直方图 -> 答案」，
在**没见过的序列**上考（划分与 LLM 评测同一份 rnaqa_seq30_split.jsonl）。

三类探针（都只用码，不用任何隐藏态）：
  数值型（计数 / GC% / 占比）  : 岭回归 hist -> 数值，按模板原格式输出
  序列型（first/last k nt、反向互补）: 码 -> 该码最常见的 4-mer 查表，拼出前/后 k nt
  类别型（What type of RNA is this?）: 码直方图的最近质心

另外补上 66_ 留的口子：**窗口内四个位置各自的可预测性**。
7.79x 只用了窗口首位那个 6-mer；若 compress 只编码了窗口首位，这张表会露馅。

跑法:
    python scripts/69_code_answer_probe.py --ckpt runs/e2e_A_v3/ckpt/step_016000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 0}
B4 = {"A": 0, "C": 1, "G": 2, "U": 3}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_answer(a: str):
    """(类型, 值, 格式串)。类型: int | pct | float | seq | cat"""
    a = a.strip()
    if re.fullmatch(r"-?\d+", a):
        return "int", int(a), "{}"
    if re.fullmatch(r"\d+(\.\d+)?%", a):
        d = len(a.split(".")[1]) if "." in a else 1
        return "pct", float(a.rstrip("%")), "{:." + str(d) + "f}%"
    if re.fullmatch(r"\d+(\.\d+)?", a):
        d = len(a.split(".")[1]) if "." in a else 0
        return "float", float(a), "{:." + str(d) + "f}"
    if re.fullmatch(r"[ACGU]+", a):
        return "seq", a.upper(), "{}"
    return "cat", a.upper(), "{}"


def seq_answer(q: str, s: str) -> str:
    """序列型模板的答案，直接从解码出的序列上切（模板见 56_ 的 30 个）。"""
    ql = q.lower()
    if not s:
        return ""
    if "reverse complement" in ql:
        m = re.search(r"first (\d+)", ql)
        k = int(m.group(1)) if m else 4
        frag = s[:k]
        if len(frag) < k:
            return ""
        comp = {"A": "U", "U": "A", "G": "C", "C": "G"}
        return "".join(comp[ch] for ch in frag)[::-1]
    m = re.search(r"first (\d+) nucleotides", ql)
    if m:
        return s[:int(m.group(1))]
    m = re.search(r"last (\d+) nucleotides", ql)
    if m:
        k = int(m.group(1))
        return s[-k:] if len(s) >= k else ""
    m = re.search(r"positions 1 through (\d+)", ql)
    if m:
        return s[:int(m.group(1))]
    if "positions 1 and 2" in ql:
        return s[:2]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/e2e_A_v3/ckpt/step_016000.pt")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=3.0, help="岭回归系数")
    ap.add_argument("--out", default="reports/code_answer_probe.json")
    args = ap.parse_args()
    t0 = time.time()

    tok48 = _load("t48", ROOT / "scripts/48_stage1_full_tokenizer.py")
    rc = _load("rc50", ROOT / "scripts/50_routeC_e2e.py")

    # ---------------------------------------------------------------- 码
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tc = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt", map_location="cpu",
                    weights_only=False)
    LAT, LEV = int(tc["latent"]), int(tc["levels"])
    VOC = LEV ** LAT
    head = rc.FSQCode(int(tc["dim"]), LAT, LEV, int(tc["dim"]), pool=args.pool)
    cur = head.state_dict()
    miss = [k for k in cur if k not in ck["fsq"]]
    head.load_state_dict({k: v for k, v in ck["fsq"].items()
                          if k in cur and tuple(cur[k].shape) == tuple(v.shape)}, strict=False)
    if miss:
        raise SystemExit(f"检查点缺 {miss}，读数不可信")
    head.eval()
    print(f"[head] {args.ckpt.name} step {ck['step']} pool {args.pool} vocab {VOC}")

    recs = tok48.load_all()
    seqs = [r["sequence"] for r in recs]
    uids = [r["uid"] for r in recs]
    seq_of = {r["uid"]: r["sequence"] for r in recs}
    d = torch.load(tok48.POS_CACHE, map_location="cpu", weights_only=False, mmap=True)
    x, lens = d["x"], d["lens"]
    offs, o = [], 0
    for L in lens.tolist():
        offs.append(o)
        o += L

    codes_of: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, (uid, L) in enumerate(zip(uids, lens.tolist())):
            n = L // args.pool
            if n == 0:
                codes_of[uid] = torch.zeros(0, dtype=torch.long)
                continue
            xs = x[offs[i]:offs[i] + n * args.pool].float().reshape(n * args.pool, -1)
            c, _, _, _ = head(xs)
            codes_of[uid] = c.long()
    print(f"[codes] {len(codes_of)} 条序列  {time.time()-t0:.0f}s")

    hist_of = {}
    for uid, c in codes_of.items():
        h = torch.bincount(c, minlength=VOC).float()
        hist_of[uid] = h / max(float(h.sum()), 1.0)

    # ---------------------------------------------------------------- 问答 + 划分
    train, test = [], []
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (train if r.get("split") == "train" else test).append(r)
    split_of = {r["uid"]: r["split"] for r in train + test}
    tr_uids = [u for u in uids if split_of.get(u) == "train"]
    te_uids = [u for u in uids if split_of.get(u) == "test"]
    print(f"[qa] train {len(train):,} / test {len(test):,}；"
          f"序列 train {len(tr_uids)} / test {len(te_uids)}")

    # ---------------------------------------------------------------- 窗口内四位
    print("\n=== 窗口内四个位置各自的可预测性（66_ 留的口子）===")
    cnts = [torch.zeros(VOC, 4) for _ in range(args.pool)]
    for uid in tr_uids:
        c, s = codes_of[uid], seq_of[uid]
        if c.numel() == 0:
            continue
        for j in range(args.pool):
            widx = [w for w in range(c.numel())
                    if w * args.pool + j < len(s) and s[w * args.pool + j] in BASE]
            if not widx:
                continue
            ci = c[torch.tensor(widx)]
            bs = torch.tensor([BASE[s[w * args.pool + j]] for w in widx])
            cnts[j].index_put_((ci, bs), torch.ones(len(widx)), accumulate=True)
    base_acc = []
    for j in range(args.pool):
        maj = cnts[j].argmax(dim=1)
        hit = tot = 0
        for uid in te_uids:
            c, s = codes_of[uid], seq_of[uid]
            widx = [w for w in range(c.numel())
                    if w * args.pool + j < len(s) and s[w * args.pool + j] in BASE]
            if not widx:
                continue
            bs = torch.tensor([BASE[s[w * args.pool + j]] for w in widx])
            pr = maj[c[torch.tensor(widx)]]
            hit += int((pr == bs).sum()); tot += len(widx)
        base_acc.append(hit / max(tot, 1))
    print("  " + "  ".join(f"窗口内第{j}位 {100*a:.1f}%" for j, a in enumerate(base_acc))
          + "   （随机 25.0%）")
    print("  => " + ("✅ 四位都还能读出来" if min(base_acc) > 0.30
                     else "⚠️ 至少有一位接近随机 —— 池化把窗口内信息压掉了"))

    # 每个码最常见的 4-mer
    mc = torch.zeros(VOC, 4 ** 4)
    for uid in tr_uids:
        c, s = codes_of[uid], seq_of[uid]
        for w in range(c.numel()):
            p = w * args.pool
            frag = s[p:p + 4]
            if p + 4 <= len(s) and all(ch in B4 for ch in frag):
                v = 0
                for ch in frag:
                    v = v * 4 + B4[ch]
                mc[int(c[w]), v] += 1
    mer4 = mc.argmax(dim=1)
    mer4[mc.sum(dim=1) == 0] = -1

    def decode_seq(uid: str) -> str:
        out = []
        for v0 in codes_of[uid].tolist():
            v = int(mer4[v0])
            if v < 0:
                continue
            w = ""
            for _ in range(4):
                w = "ACGU"[v % 4] + w
                v //= 4
            out.append(w)
        return "".join(out)

    # ---------------------------------------------------------------- 逐模板探针
    byt: dict[str, list] = defaultdict(list)
    for r in train:
        byt[r["question"]].append(r)

    # 有些模板不是每条序列都有（例如 first-50 那两个只有 80%）。为了让所有模板
    # 共用同一个 Gram 矩阵（4096x4096，算一次就够），取**交集**。
    cover = [set(r["uid"] for r in rows) for rows in byt.values()]
    common = set.intersection(*cover)
    sub_uids = [u for u in tr_uids if u in common]
    X = torch.stack([hist_of[u] for u in sub_uids])
    A = X.t() @ X + args.alpha * torch.eye(VOC)
    L = torch.linalg.cholesky(A)
    print(f"[ridge] 设计矩阵 {tuple(X.shape)}（覆盖 {len(common)}/{len(tr_uids)} 条共有序列）"
          f"  分解完 {time.time()-t0:.0f}s")

    models: dict[str, dict] = {}
    for q, rows in byt.items():
        kinds = Counter(parse_answer(r["answer"])[0] for r in rows)
        kind = kinds.most_common(1)[0][0]
        m: dict = dict(kind=kind, fmt=parse_answer(rows[0]["answer"])[2])
        ymap = {r["uid"]: parse_answer(r["answer"])[1] for r in rows}
        if kind in ("int", "pct", "float"):
            y = torch.tensor([ymap[u] for u in sub_uids], dtype=torch.float32)
            m["w"] = torch.cholesky_solve((X.t() @ y).unsqueeze(1), L).squeeze(1)
        elif kind == "cat":
            classes = sorted({parse_answer(r["answer"])[1] for r in rows})
            lab = torch.tensor([classes.index(ymap[u]) for u in sub_uids])
            cen = torch.zeros(len(classes), VOC)
            for ci in range(len(classes)):
                sel = X[lab == ci]
                cen[ci] = sel.mean(0) if len(sel) else 0.0
            m["classes"], m["cen"] = classes, cen
        models[q] = m

    # ---------------------------------------------------------------- 评测
    print(f"\n=== 探针（只看码）vs 模板 ===")
    print(f"{'模板':<52} {'探针':>7} {'n':>6}  类型")
    res, hits, tot = [], 0, 0
    for q, m in models.items():
        rows = [r for r in test if r["question"] == q and r["uid"] in hist_of]
        if not rows:
            continue
        h = 0
        for r in rows:
            uid = r["uid"]
            if m["kind"] in ("int", "pct", "float"):
                p = float(hist_of[uid] @ m["w"])
                got = m["fmt"].format(round(p) if m["kind"] == "int" else p)
            elif m["kind"] == "cat":
                v = hist_of[uid]
                sim = (m["cen"] @ v) / (m["cen"].norm(dim=1) * v.norm() + 1e-9)
                got = m["classes"][int(sim.argmax())]
            else:
                got = seq_answer(q, decode_seq(uid))
            h += (got == r["answer"].strip().upper())
        hits += h; tot += len(rows)
        res.append(dict(question=q, probe_acc=h / len(rows), n=len(rows), kind=m["kind"]))
        print(f"{q[:52]:<52} {100*h/len(rows):>6.1f}% {len(rows):>6}  {m['kind']}")

    print(f"\n  探针总体 {100*hits/max(tot,1):.1f}%  ({hits}/{tot})")
    print("  对照：LLM acc(A 真RNA) 16.7% / acc(C 不给RNA) 25.6%（n=90，同划分）；"
          "众数地板 3.3%")

    # ---------------------------------------------------------------- 计数天花板
    # 数值模板（16/30 个）探针几乎全灭。但「精确匹配 0.0%」不区分两件事：
    #   (i) 码里没有精确的全局计数     (ii) 岭回归 + 0.1% 精度本身做不到
    # 最干净的分离办法：**不训练任何东西**，直接用「码 -> 该码在训练窗口里的
    # 平均碱基个数」求和。它是「只看码 + 无状态」能达到的最优估计，
    # 误差里就包含码自身的信息损失。报 MAE / 精确率 / R²。
    print("\n=== 计数天花板：Σ(码的经验均值) 能不能还原真实计数 ===")
    per_code_sum = {b: torch.zeros(VOC) for b in "ACGU"}
    per_code_sum["GC"] = torch.zeros(VOC)
    per_code_cnt = torch.zeros(VOC)
    for uid in tr_uids:
        c, s = codes_of[uid], seq_of[uid]
        for w in range(c.numel()):
            frag = s[w * args.pool:(w + 1) * args.pool]
            ci = int(c[w])
            per_code_cnt[ci] += 1
            for b in "ACGU":
                per_code_sum[b][ci] += frag.count(b)
            per_code_sum["GC"][ci] += frag.count("G") + frag.count("C")
    est = {k: (v / per_code_cnt.clamp(min=1)) for k, v in per_code_sum.items()}

    count_rows = []
    for key in ("A", "C", "G", "U", "GC"):
        errs = []
        for uid in te_uids:
            c, s = codes_of[uid], seq_of[uid]
            pred = float(est[key][c].sum())
            if key == "GC":
                true = s.count("G") + s.count("C")
                errs.append((pred - true) / max(len(s), 1) * 100)   # 百分点
            else:
                errs.append(pred - s.count(key))                    # 碱基个数
        e = torch.tensor(errs)
        mae = float(e.abs().mean())
        exact = float((e.abs() < 0.5).float().mean())
        unit = "%" if key == "GC" else "个"
        count_rows.append(dict(target=key, mae=mae, medae=float(e.abs().median()),
                               exact=exact, sd=float(e.std()) if len(e) > 1 else 0.0))
        print(f"  {key:>2} 计数：MAE {mae:6.2f}{unit}   中位 {float(e.abs().median()):5.2f}"
              f"   |误差|<0.5 的比例 {100*exact:5.1f}%   误差 sd {float(e.std()):.2f}")
    print("  读法：MAE 小（~1-3 个）=> 码里带着计数，数值模板是可做的，LLM 没学会是它的问题；")
    print("        MAE 大（> 10 个）=> 码本身还原不了精确计数，16/30 个模板被 tokenizer 卡住。")

    outp = ROOT / args.out
    outp.write_text(json.dumps(dict(ckpt=str(args.ckpt), pool=args.pool, alpha=args.alpha,
                                    per_position_base_acc=base_acc,
                                    count_ceiling=count_rows,
                                    probe_overall=hits / max(tot, 1), n_test=tot,
                                    per_template=res), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[save] {outp}   用时 {(time.time()-t0)/60:.1f} min")
    print("\n判读：探针是**只看码**能达到的天花板。")
    print("  · 探针高、LLM 低 -> 码里有答案，LLM 没学会读 => 动 LLM 侧")
    print("  · 探针也低       -> 码里没有答案          => 动 tokenizer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
