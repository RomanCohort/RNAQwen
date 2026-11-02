"""训练日志里的 `ce 0.2~0.9` 到底在量什么？—— 答案 token 只占被算 loss 的少数。

问题
----
`50_routeC_e2e.py` 的 labels **从 assistant 头开始**（有意如此，否则生成时学不会停）。
那段固定文本是：

    <|im_end|><|im_start|>assistant\n  +  空 think 块(13 token)  +  " A:"  +  eos

合计约 **18~20 个几乎确定的 token**。而答案本体常常只有 **1~3 个 token**
（"How many times does the dinucleotide GC occur?" -> " 0"）。
于是日志里的 `ce` 被样板文本稀释：ce 0.4 完全可能意味着「样板学会了、答案一个字没学会」。

这正好就是 ChatMultiOmics 说的那句：
`LoRA primarily leverages pre-trained knowledge to generate well-formed answers
based on the output format learned from SFT data`。

所以这个脚本把 loss 拆成三段量（**口径与训练时逐字一致**：labels = ids[n_head:]，
n_head = len(ids) - len(assistant_ids) - len(answer_text)）：

    ce_all    训练日志里的那个数（样板 + 答案）
    ce_head   样板段（assistant 头 + 空 think 块 + " A:"）
    ce_ans    答案 token（真正决定对错的那几个）

判读：ce_ans 换成「有多少候选在竞争」（exp(ce_ans)），再跟该模板实际的取值数比。
训练集上 ce_ans 若仍然远大于 ln(取值数)，说明模型**连拟合都没做到**。

跑法:
    python scripts/68_answer_only_ce.py --artifacts runs/exp_14000 --n 120
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
HID = ROOT / "data/processed/_fsq_full_pos.pt"
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(ch for ch in s.strip().upper() if ch.isalnum() or ch in ".%").rstrip(".")


def pick_roundrobin(pool, n):
    byq = {}
    for r in pool:
        byq.setdefault(r["question"], []).append(r)
    picked, i = [], 0
    while len(picked) < n and i < 10 ** 6:
        got = False
        for q, lst in byq.items():
            if i < len(lst) and len(picked) < n:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="runs/exp_14000")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    ap.add_argument("--n", type=int, default=120, help="每个 split 取多少条")
    ap.add_argument("--scales", default="0,0.5,1.0,2.0",
                    help="把码表向量整体乘 λ 的剂量-反应；λ=0 即「不给 RNA」")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    art = ROOT / args.artifacts
    tag = art.name
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[art] {art}  LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB}")

    train, test = [], []
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (train if r.get("split") == "train" else test).append(r)
    _c: dict[str, Counter] = {}
    for r in train:
        _c.setdefault(r["question"], Counter())[norm(r["answer"])] += 1
    n_choices = {q: len(c) for q, c in _c.items()}

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {r["uid"]: int(d["offsets"][j]) for j, r in enumerate(allr)}
    len_of = {r["uid"]: int(lens[j]) for j, r in enumerate(allr)}

    from transformers import AutoTokenizer
    _tk0 = AutoTokenizer.from_pretrained(str(QWEN))
    tk = rc.Tok(_tk0)
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(str(art / "lora_adapter" / "adapter_model.safetensors"))
    set_peft_model_state_dict(model.llm, sd)
    print(f"[lora] 载入 {len(sd)} 个张量")
    model.eval()
    P = 4

    @torch.no_grad()
    def ce_of(r: dict, scale: float = 1.0) -> dict:
        clen = len_of[r["uid"]] // P
        x = hid[off_of[r["uid"]]:off_of[r["uid"]] + clen * P].to(DEV, dtype=torch.float32)
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)
        if scale != 1.0:
            v = v * scale
        head = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>") + [tk.base_vocab] * clen
                + tk.enc(f"</RNA>{r['question']}") + tk.assistant_ids + tk.no_think)
        ans_txt = f" A: {r['answer']}{tk.eos}"
        tail = tk.enc(ans_txt)
        ids = head + tail
        ii = torch.tensor([ids], device=DEV)
        emb = model.embed(ii, v)
        am = torch.ones_like(ii)
        lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[0, :-1]
        tgt = torch.tensor(ids[1:], device=DEV)
        lp = F.log_softmax(lg, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # 位置 p 的 logp 预测 ids[p+1]。训练时 labels 从 assistant 头开始：
        # n_head = len(ids) - len(assistant_ids) - len(enc(ans_txt))，对应 lp 的下标 n_head-1
        n_head = len(ids) - len(tk.assistant_ids) - len(tail)
        a0 = len(head) - 1                       # 答案首 token 由 lp[len(head)-1] 预测
        ce_all = float(-lp[n_head - 1:].mean())
        ce_head = float(-lp[n_head - 1:a0].mean())
        ce_ans = float(-lp[a0:].mean())
        return dict(ce_all=ce_all, ce_head=ce_head, ce_ans=ce_ans,
                    n_ans=len(tail), n_labelled=len(ids) - n_head)

    # ---- 剂量-反应：把码表向量整体乘 λ。λ=0 就是「不给 RNA」那一臂。----
    # 若 ce_ans(λ) 随 λ 单调变差，说明码向量不是「没被用上」而是**净负担**。
    scales = [float(x) for x in args.scales.split(",")]
    report = {}
    for sp, pool in (("train", train), ("test", test)):
        picked = [r for r in pick_roundrobin(pool, args.n) if r["uid"] in off_of]
        t0 = time.time()
        curve = []
        for sc in scales:
            agg = defaultdict(list)
            for r in picked:
                c = ce_of(r, scale=sc)
                for k, v in c.items():
                    agg[k].append(v)
            mean = {k: sum(v) / len(v) for k, v in agg.items()}
            curve.append(dict(scale=sc, ce_ans=mean["ce_ans"], ce_all=mean["ce_all"],
                              ce_head=mean["ce_head"], ppl_ans=math.exp(mean["ce_ans"])))
            print(f"  [{sp}] λ={sc:<4} ce_ans {mean['ce_ans']:7.4f}"
                  f"  (困惑度 {math.exp(mean['ce_ans']):8.1f})   ce_all {mean['ce_all']:.4f}"
                  f"   {time.time()-t0:.0f}s", flush=True)
        base = next(c for c in curve if c["scale"] == 1.0)
        print(f"\n[{sp}] n={len(picked)}  λ=1：ce_all {base['ce_all']:.4f} / "
              f"ce_head(样板) {base['ce_head']:.4f} / **ce_ans(答案) {base['ce_ans']:.4f}**"
              f"  被算 loss 的 token 均值 {sum(agg['n_labelled'])/len(picked):.1f}"
              f"（其中答案 {sum(agg['n_ans'])/len(picked):.1f}）")

        # 逐模板（λ=1）
        per_tpl = defaultdict(lambda: defaultdict(list))
        for r in picked:
            c = ce_of(r, scale=1.0)
            for k, v in c.items():
                if k not in ("n_ans", "n_labelled"):
                    per_tpl[r["question"]][k].append(v)
        rows = []
        print(f"\n  {'模板':<52} {'ce_ans':>8} {'ce_all':>8} {'取值数':>6}")
        for q, v in sorted(per_tpl.items(),
                           key=lambda kv: -sum(kv[1]["ce_ans"]) / len(kv[1]["ce_ans"])):
            ca = sum(v["ce_ans"]) / len(v["ce_ans"])
            ca_all = sum(v["ce_all"]) / len(v["ce_all"])
            rows.append(dict(question=q, ce_ans=ca, ce_all=ca_all, n=len(v["ce_ans"]),
                             n_choices=n_choices.get(q, 0)))
            print(f"  {q[:52]:<52} {ca:>8.3f} {ca_all:>8.3f} {n_choices.get(q,0):>6}")
        report[sp] = dict(n=len(picked), curve=curve,
                          n_labelled=sum(agg["n_labelled"]) / len(picked),
                          n_ans=sum(agg["n_ans"]) / len(picked),
                          per_template=rows)

    outp = ROOT / (args.out or f"reports/answer_only_ce_{tag}.json")
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {outp}")
    tr, te = report["train"], report["test"]
    tr1 = next(c for c in tr["curve"] if c["scale"] == 1.0)
    te1 = next(c for c in te["curve"] if c["scale"] == 1.0)
    print("\n判读：")
    print(f"  训练集 ce_ans {tr1['ce_ans']:.3f}（困惑度 {tr1['ppl_ans']:.1f}）  "
          f"测试集 ce_ans {te1['ce_ans']:.3f}（困惑度 {te1['ppl_ans']:.1f}）")
    if tr1["ce_ans"] > 3.0:
        print("  ❌ 训练集上答案段仍然很贵 —— 模型**连训练样本的答案都没学会**。")
        print("     日志里的 ce 0.2~0.9 是样板文本贡献的（那段本来就近乎确定）。")
    else:
        print("  ✅ 训练集答案学会了；问题在泛化。")
    print("\n  剂量-反应（把码表向量整体乘 λ，λ=0 即「不给 RNA」）：")
    for sp in ("train", "test"):
        s = "  ".join(f"λ{c['scale']:g}:{c['ce_ans']:.3f}" for c in report[sp]["curve"])
        print(f"    {sp:<5} {s}")
    print("    λ=1 比 λ=0 更差 => 码向量是净负担，不是「没被用上」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
