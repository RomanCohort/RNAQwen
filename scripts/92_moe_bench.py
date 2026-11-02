"""MoE 开销定性实验：时间是「启动延迟」还是「算力/带宽」主导？

为什么要做
----------
35B 实测 17.6 秒/步，而它是 3B 激活的 MoE，8B dense 只要 2.4 秒/步。
按算力/带宽我推不出 17.6 秒这个量级（粗算反量化 ~2 GB + 卷积替换几项乘加，
加起来几百毫秒），差两个数量级 —— **说明有个我没算到的大头**。
猜着优化会白费功夫，先做一次能定性的测量。

判据
----
固定**样本总数**，只改 batch：

    时间几乎不随 batch 变   -> 启动延迟瓶颈（320 次专家派发 + 数千次 kernel 启动）
                               解法：合并成 grouped GEMM
    时间随 batch 线性涨      -> 算力/带宽瓶颈
                               解法完全不同（减反量化流量 / 换精度 / 减层数）

做法
----
载入一次 35B（用和训练完全一样的装载路径），然后对 batch ∈ {1,2,4,8}
各跑若干个训练步，每步都走完整的前向+反向+optimizer，
并分别报告 forward / backward 的时间。

用法：python scripts/92_moe_bench.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import rnaqwen  # noqa: F401,E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
DATA = ROOT / "data/processed/rnaqa_seqtask_full.jsonl"
HID2 = ROOT / "data/processed/_rfa_full_hidden.f16"
HID2IDX = ROOT / "data/processed/_rfa_full_index.pt"
BATCHES = (1, 2, 4, 8)
STEPS_PER = {1: 24, 2: 12, 4: 6, 8: 3}      # 每个 batch 固定 ~24 个样本


def main() -> int:
    import numpy as np
    from transformers import AutoTokenizer

    # ---- 缓存 ----
    d = torch.load(rc.HID, map_location="cpu", weights_only=False)
    allr = rc.load_records(None, sort=False)
    rc.register_cache("main", d["x"], d["lens"], d["offsets"], [r["uid"] for r in allr])
    d2 = torch.load(HID2IDX, map_location="cpu", weights_only=False)
    import numpy as np
    mm = np.memmap(HID2, dtype=np.float16, mode="r",
                   shape=(int(d2["n_pos"]), int(d2["dim"])))
    rc.register_cache("aux", torch.from_numpy(mm), d2["lens"], d2["offsets"], d2["uids"])
    print(f"[hid] 缓存条目 {len(rc.IDX):,}", flush=True)

    tkc = torch.load(rc.FSQ_CKPT, map_location="cpu", weights_only=False)
    LAT, LEV = int(tkc["latent"]), int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    P = 4
    rc.TAU = LAT / 2

    # ---- 取一批真实样本（按 split 分层，长度按真实分布）----
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("split") != "train" or q["uid"] not in rc.IDX:
                continue
            rows.append(q)
            if len(rows) >= 4000:
                break
    ex = [({"uid": r["uid"]}, {"question": r["question"], "answer": r["answer"]},
           rc.IDX[r["uid"]][2] // P, rc.IDX[r["uid"]][1]) for r in rows]
    import collections
    cl = [e[2] for e in ex]
    print(f"[data] {len(ex):,} 条  码长 中位 {sorted(cl)[len(cl)//2]}  "
          f"均值 {sum(cl)/len(cl):.1f}  最大 {max(cl)}", flush=True)

    tk0 = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3-8B"))
    tk = rc.Tok(tk0)

    print("[model] 载入 35B int4 …", flush=True)
    model = rc.E2EModel(8, "table", latent=LAT, levels=LEV, pool=P,
                        base="qwen36-35b", base_mode="int4", lora_preset="C")
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(torch.load(rc.FSQ_CKPT, map_location="cpu",
                                         weights_only=False)["state"], strict=False)
    ci = torch.load(ROOT / "checkpoints/fsq_full/code_emb_contentinit_d2048.pt",
                    map_location="cpu", weights_only=False)["code_emb"]
    model.rna_emb.weight.data.copy_(ci.to(model.rna_emb.weight.device,
                                          model.rna_emb.weight.dtype))
    params = ([p for p in model.llm.parameters() if p.requires_grad]
              + list(model.fsq.parameters()) + [model.rna_emb.weight])
    opt = torch.optim.AdamW(params, lr=2e-4)
    model.train()

    def one_batch(bs, g):
        picked, tot = [], 0
        while len(picked) < bs:
            e = ex[torch.randint(0, len(ex), (1,), generator=g).item()]
            if picked and tot + e[2] > 600:
                break
            picked.append(e); tot += e[2]
        ids, lab, am = rc.build_batch(tk, picked)
        vecs, t_all, q_all = [], [], []
        for r, qa, clen, off in picked:
            v, c, t, q = model.rna_vectors(rc.slice_hid(r["uid"]), tau=rc.TAU)
            vecs.append(v); t_all.append(t); q_all.append(q)
        v = torch.cat(vecs)
        emb = model.embed(ids, v)
        return ids, lab, am, emb, v, t_all, q_all, len(picked), tot

    print("\n" + "=" * 78)
    print("固定样本总数 ~24，只改 batch。分开测 forward / backward")
    print("=" * 78)
    print(f"  {'batch':>6}{'步数':>6}{'样本':>6}{'码长合计':>9}"
          f"{'fwd/步':>10}{'bwd/步':>10}{'合计/步':>10}{'每样本':>10}")
    g = torch.Generator().manual_seed(0)
    out = {}
    for bs in BATCHES:
        n = STEPS_PER[bs]
        # 预热 1 步（不计时）
        for _ in range(1):
            ids, lab, am, emb, v, t_all, q_all, ns, tot = one_batch(bs, g)
            lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
            tgt = lab[:, 1:]
            m = tgt >= 0
            loss = F.cross_entropy(lg[m], tgt[m])
            loss.backward(); opt.zero_grad()
        torch.cuda.synchronize()
        t_fwd = t_bwd = 0.0
        n_samp = n_tok = 0
        for _ in range(n):
            ids, lab, am, emb, v, t_all, q_all, ns, tot = one_batch(bs, g)
            torch.cuda.synchronize(); t0 = time.time()
            lg = model.llm(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
            tgt = lab[:, 1:]
            m = tgt >= 0
            loss = F.cross_entropy(lg[m], tgt[m])
            torch.cuda.synchronize(); t1 = time.time()
            loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize(); t2 = time.time()
            t_fwd += t1 - t0; t_bwd += t2 - t1
            n_samp += ns; n_tok += tot
        f, b = t_fwd / n, t_bwd / n
        out[bs] = dict(fwd=f, bwd=b, total=f + b, n_samp=n_samp,
                       per_sample=(f + b) / (n_samp / n))
        print(f"  {bs:>6}{n:>6}{n_samp:>6}{n_tok:>9}"
              f"{f:>10.3f}{b:>10.3f}{f+b:>10.3f}{out[bs]['per_sample']:>10.3f}", flush=True)

    print()
    print("=" * 78)
    r1, r8 = out[1]["total"], out[8]["total"]
    p1, p8 = out[1]["per_sample"], out[8]["per_sample"]
    print(f"  batch 1 -> 8：每步时间 {r1:.2f}s -> {r8:.2f}s（{r8/r1:.2f}x）")
    print(f"                每样本  {p1:.3f}s -> {p8:.3f}s（{p8/p1:.2f}x）")
    if r8 / r1 < 2.0:
        print("  -> **启动延迟主导**：8 倍样本只花不到 2 倍时间")
        print("     解法：把 8 次专家前向合并成一次 grouped GEMM；加大 batch")
    elif p8 / p1 > 0.8:
        print("  -> **算力/带宽主导**：每样本时间基本不变")
        print("     解法：减反量化流量、换精度、或换更高效的 MoE 内核")
    else:
        print("  -> 混合：有启动开销，但也被算力限制")
    print("=" * 78)
    (ROOT / "reports/moe_bench.json").write_text(
        json.dumps({str(k): v for k, v in out.items()}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
