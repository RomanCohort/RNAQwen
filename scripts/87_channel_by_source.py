"""按来源量「码信道到底带了多少信息」—— 用 teacher-forced CE，不生成。

为什么不能用 76_ 的 exact match 回答这个问题
--------------------------------------------
76_ 里 freetext 出现 A 臂和 C 臂逐点相等，当时我读成"信道贡献为零"。**那个推断不成立。**
freetext 的答案平均 228 字符，生成上限 48 token，判据是整段 exact match ——
这个组合对 freetext 几乎失明：它只看得见"是否一字不差"，看不见"差得少一点"。

所以 A == C 有两种解释，exact match 分不开：
    1. 信道真的没用
    2. 信道有用，但两臂都够不到 exact match 那条线
而且 A 命中的那少数几条，很可能是那个占 32.3% 的拒答吸引子 —— 两臂都吐它，
一致但不说明任何事。

换尺子：**答案 token 的 CE（teacher-forced）**。
它逐 token 比较，A 臂只要比 C 臂低一点就说明信道带了信息，不需要整段蒙对。
而且不用生成 —— 35B 上生成 40 秒一条，CE 只要一次前向。

三个臂 + 一个基准
----------------
    A  真实 RNA 的码向量
    B  换一条同来源 RNA 的码向量（切断码与内容的对应）
    C  码向量置零（切断整条信道）
    MODE  不训练任何东西：永远输出该模板在训练集里最常见的答案
          —— 这是"问题先验"能拿到的 CE。**CE_A 接近 MODE 就说明没读到序列。**

判读（paired，逐样本比较，比比均值灵敏）
    CE_A < CE_C 且 CE_A < MODE  -> 信道确实带信息
    CE_A ≈ CE_C ≈ MODE          -> 信道没被用上，模型只在读问题

用法：python scripts/87_channel_by_source.py --ckpt runs/e2e_C_merged/ckpt/step_015000.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ⚠️ 强制 stdout/stderr 用 utf-8。
#    控制台默认编码可能是 GBK（cp936），而本文件（以及仓库里 60+ 个脚本）的日志里
#    带 ✓ / ⚠ / ✅ 这类非 GBK 字符 —— 一个 print 就会抛 UnicodeEncodeError，
#    把跑了几十分钟的评测整条打死（2026-10-14 踩过：模型装载完、适配器也校验过，
#    死在守卫生成的那行日志上）。
#    放在脚本里自带免疫，不依赖调用方记得设 PYTHONIOENCODING —— 那种依赖迟早会漏。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import rnaqwen  # noqa: F401,E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c.isalnum() or c in ".%").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed/rnaqa_merged.jsonl")
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/channel_by_source.json")
    args = ap.parse_args()

    ck_path = ROOT / args.ckpt
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    print(f"[ckpt] {ck_path.name}  step {ck.get('step')}  "
          f"{ck.get('t_elapsed', 0)/3600:.2f} h")

    # ---- 缓存 ----
    d = torch.load(rc.HID, map_location="cpu", weights_only=False)
    allr = rc.load_records(None, sort=False)
    rc.register_cache("main", d["x"], d["lens"], d["offsets"], [r["uid"] for r in allr])
    hid2 = cargs.get("hid2", "")
    if hid2:
        import numpy as np
        d2 = torch.load(ROOT / cargs["hid2_idx"], map_location="cpu", weights_only=False)
        mm = np.memmap(ROOT / hid2, dtype=np.float16, mode="r",
                       shape=(int(d2["n_pos"]), int(d2["dim"])))
        rc.register_cache("aux", torch.from_numpy(mm), d2["lens"], d2["offsets"],
                          d2["uids"])
    print(f"[hid] 缓存条目 {len(rc.IDX):,}")

    tkc = torch.load(rc.FSQ_CKPT, map_location="cpu", weights_only=False)
    LAT, LEV = int(tkc["latent"]), int(tkc["levels"])
    POOL = int(cargs.get("pool", 4))
    LORA_R = int(cargs.get("lora_r", 8))
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT

    # ⚠️ 底座必须跟着 checkpoint 走，不能写死 8B —— 否则拿 35B 的档跑会**静默**
    #    加载 8B、再 strict=False 装适配器，不匹配的参数不报错、日志正常、数字全假。
    from rnaqwen.qwen36 import base_spec_from_ckpt_args
    base_dir, mk_kw = base_spec_from_ckpt_args(cargs)
    print(f"[cfg] LATENT={LAT} VOCAB={rc.VOCAB} pool={POOL} lora_r={LORA_R}")
    print(f"[cfg] 底座 {base_dir.name}  adapter={mk_kw['adapter']} "
          f"preset={mk_kw['lora_preset']} mode={mk_kw['base_mode']}")

    from transformers import AutoTokenizer
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tk = rc.Tok(tk0)
    print(f"[model] 载入 {base_dir.name} + adapter + 码表 …")
    model = rc.E2EModel(LORA_R, cargs.get("rna_emb", "table"),
                        latent=LAT, levels=LEV, **mk_kw)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    bad = list(getattr(miss, "unexpected_keys", None) or [])
    print(f"[lora] {len(ck['lora'])} 张量  unexpected {len(bad)}")
    if bad:
        print(f"  ** adapter 没挂上：{bad[:4]}")
        return 3
    # 反向也要查：可训参数是否被 checkpoint 全覆盖。只看 unexpected_keys 会漏掉
    # 「拿 8B 的档装进 35B」那种情况 —— 那时 unexpected=0，而适配器全是随机初始化。
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(ck["lora"].keys())
    if want - got:
        print(f"  ** 有 {len(want - got)} 个可训参数不在 checkpoint 里"
              f"（adapter 与底座对不上）：{sorted(want - got)[:4]}")
        return 3
    print(f"[lora] 可训参数 {len(want)} 个，全部被 checkpoint 覆盖 ")
    model.eval()
    TAU = LAT / 2

    # ---- 抽样 ----
    recs = [json.loads(l) for l in open(ROOT / args.data, encoding="utf-8") if l.strip()]
    pool = [r for r in recs if r.get("split") == args.split and r["uid"] in rc.IDX]
    if not pool:
        pool = [r for r in recs if r.get("split") == "val" and r["uid"] in rc.IDX]
    train = [r for r in recs if r.get("split") == "train"]

    picks: dict[str, list] = {}
    for src in sorted({r["source"] for r in pool}):
        v = [r for r in pool if r["source"] == src]
        byq: dict[str, list] = {}
        for r in v:
            byq.setdefault(r["question"], []).append(r)
        out, i = [], 0
        while len(out) < args.per_source:
            got = False
            for q, lst in byq.items():
                if i < len(lst) and len(out) < args.per_source:
                    out.append(lst[i]); got = True
            if not got:
                break
            i += 1
        picks[src] = out
    print(f"[data] {args.split} 共 {len(pool):,}，取 " +
          "  ".join(f"{k} {len(v)}" for k, v in picks.items()))

    # 模板众数答案（问题先验的天花板）
    mode_ans = {}
    for src, v in picks.items():
        for q in {r["question"] for r in v}:
            c = Counter(norm(r["answer"]) for r in train
                        if r["source"] == src and r["question"] == q)
            mode_ans[(src, q)] = c.most_common(1)[0][0] if c else ""

    def clen_of(u):
        return rc.IDX[u][2] // POOL

    @torch.no_grad()
    def ce_of(rec, arm, code_rec=None, mode_text=None):
        """rec 提供问题+答案（prompt 的内容），code_rec 提供码（默认就是 rec）。

        **arm B 必须是「用别人的码、配自己的题目和答案」。**
        如果把 r2 整个换掉，那换的就不只是 RNA，题目和答案也一起换了，
        量到的变成"换一道题"，不是"换一条 RNA"。
        """
        src_rec = code_rec or rec
        u = src_rec["uid"]
        n = clen_of(u)
        if arm == "C":
            v = torch.zeros(n, model.base_emb.weight.shape[1], device=DEV,
                            dtype=torch.float32)
        else:
            x = rc.slice_hid(u)
            v, _, _, _ = model.rna_vectors(x, tau=TAU)
        ans = mode_text if mode_text is not None else rec["answer"]
        tail = tk.enc(f" A: {ans}{tk.eos}")
        ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
               + [tk.base_vocab] * n
               + tk.enc(f"</RNA>{rec['question']}")
               + tk.assistant_ids + tk.no_think + tail)
        lab = [-100] * (len(ids) - len(tail)) + ids[len(ids) - len(tail):]
        ii = torch.tensor([ids], device=DEV)
        ll = torch.tensor([lab], device=DEV)
        am = torch.ones_like(ii)
        emb = model.embed(ii, v)
        lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
        tgt = ll[:, 1:]
        m = tgt >= 0
        if not m.sum():
            return 0.0, 0
        lp = F.log_softmax(lg, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return float(-lp[m].sum()), int(m.sum())

    print("\n=== 逐来源 teacher-forced CE（越低越好；只在答案 token 上算）===")
    summary = {}
    t0 = time.time()
    for src, v in picks.items():
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(len(v), generator=g).tolist()
        rows = []
        for i, rec in enumerate(v):
            ce = {}
            ntok = 0
            for arm in ("A", "B", "C"):
                # B 臂：用别人的码，题目和答案还是自己的
                code_rec = v[perm[i]] if arm == "B" else None
                s, ntok = ce_of(rec, arm, code_rec=code_rec)
                ce[arm] = s / max(ntok, 1)
            s, ntok = ce_of(rec, "A", mode_text=mode_ans[(src, rec["question"])])
            ce["MODE"] = s / max(ntok, 1)
            ce["ntok"] = ntok
            rows.append(ce)
            if (i + 1) % 20 == 0:
                print(f"  [{src}] {i+1}/{len(v)}  {time.time()-t0:.0f}s", flush=True)
        m = {k: sum(r[k] for r in rows) / len(rows)
             for k in ("A", "B", "C", "MODE")}
        wins = sum(1 for r in rows if r["A"] < r["C"] - 1e-6)
        wins_m = sum(1 for r in rows if r["A"] < r["MODE"] - 1e-6)
        summary[src] = dict(n=len(rows), ce=m, win_vs_C=wins / len(rows),
                            win_vs_MODE=wins_m / len(rows),
                            ntok=sum(r["ntok"] for r in rows) / len(rows))
        star = ""
        if m["A"] < m["C"] - 0.01 and m["A"] < m["MODE"] - 0.01:
            star = "✅ 信道带信息"
        elif m["A"] >= m["C"] - 0.01:
            star = "❌ 信道没被用上"
        else:
            star = "⚠️ 比零表好但不如问题先验"
        print(f"  {src:<12} CE_A {m['A']:.4f}  CE_B {m['B']:.4f}  CE_C {m['C']:.4f}  "
              f"MODE {m['MODE']:.4f}   A<C 的样本 {100*wins/len(rows):.0f}%   "
              f"A<MODE {100*wins_m/len(rows):.0f}%   {star}")

    print()
    print("=" * 78)
    print("  怎么读")
    print("=" * 78)
    print("  CE_A < CE_C   码向量置零会变差 -> 信道在用")
    print("  CE_A < MODE   **比「只读问题、永远答模板众数」还好** -> 确实从序列读到了东西")
    print("  CE_A ≈ MODE   说明模型的输出基本由问题决定，序列没贡献")
    print("  MODE 一列是关键：它是问题先验能拿到的 CE，exact match 看不到这一层")
    print("=" * 78)
    (ROOT / args.out).write_text(
        json.dumps({"ckpt": str(ck_path), "step": ck.get("step"),
                    "split": args.split, "sources": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


def _hard_exit(code: int) -> None:
    """绕过解释器收尾硬退。

    实测（2026-10-14）：python 抛异常退出后**不会真的死** —— 它挂在 ROCm 的上下文
    销毁上，进程一直留着，继续占着 18 GiB 显存，而且 `taskkill /F /T` 报
    「could not be terminated / There is no running instance of the task」，
    只有把**父 shell** 也杀掉才回收得掉。表现是"评测崩了，但显存一直不还"。
    os._exit 跳过会挂的那段析构；句柄和显存由操作系统回收。
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    _hard_exit(_code)
