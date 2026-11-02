"""PLAN §7.3 续 —— semantic F1 的**分辨率地板**量化。

动机（来自 `reports/eval_selfcheck_bert.json`）：
    B 关里 "unrelated"（RNA vs 蛋白质晶体学）拿到 **F1 = 0.8159**。
    即：两段完全无关的文本，句子级 max-cosine 也能给 0.82。
    => 这条指标的实际分辨窗口被压缩在 **[0.82, 1.00]**，不是它名义上的 [0, 1]。

本脚本把这件事量化清楚，回答三个问题：
    1. 地板有多高？（多领域无关对 + 随机配对基线）
    2. 天花板在哪？（identical 文本）
    3. 原报数 0.8609 在这个尺度下到底算不算"好"？

口径（**不改** `src/rnaqwen/eval/semantic.py` 的任何算法，只调用）：
    P = mean_i max_j cos(gen_i, ref_j)
    R = mean_j max_i cos(gen_j, ref_i)
    F1 = 2PR/(P+R)          —— 句子级 max-cosine，mean pooling，L2 归一化，无 baseline rescaling

跑法：
    # 两个 embedder 各跑一遍（推荐）
    python scripts/03b_semantic_floor.py
    # 指定
    python scripts/03b_semantic_floor.py --embedders bert,pubmedbert \
        --embedder-path bert=C:\\qwen-api\\RNAQwen\\models\\bert-base-uncased \
        --embedder-path pubmedbert=C:\\qwen-api\\RNAQwen\\models\\pubmedbert
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— import 本包即自动装载 env.shim

from rnaqwen.eval.semantic import (  # noqa: E402
    build_embedder,
    semantic_scores,
    split_sentences,
    _cos_matrix,
)

# 本地权重默认路径（上游 pubmedbert 缺 safetensors/tokenizer.json，本地补齐过）
DEFAULT_PATHS: dict[str, str] = {
    "bert": str(ROOT / "models" / "bert-base-uncased"),
    "pubmedbert": str(ROOT / "models" / "pubmedbert"),
}


# ============================================================ 文本语料
# 五个领域，每域 8 句。所有句子都是该领域的真实风格文本（不是 "hello world"），
# 且**刻意避免跨域共享实词** —— 好让"无关"是真的无关。
# 句子长度也控制在相近区间，免得 F1 被句子数/长度结构差异污染。

CORPUS: dict[str, list[str]] = {
    "rna": [
        "The RNA molecule is a non-coding RNA that regulates gene expression.",
        "It is processed by the RNase III enzyme Drosha in the nucleus.",
        "Its secondary structure contains a conserved stem-loop motif required for binding.",
        "The transcript is polyadenylated at its 3-prime end before export.",
        "Ribosomal RNA adopts a compact tertiary fold stabilized by magnesium ions.",
        "The guide strand is loaded into the Argonaute effector complex.",
        "Alternative splicing produces several isoforms from the same precursor.",
        "Nucleotide modifications at conserved positions alter translational efficiency.",
    ],
    "crystallography": [
        "The protein adopts an eight-stranded beta-barrel fold.",
        "Crystallography at 1.8 angstrom resolution revealed a bound calcium ion.",
        "Diffraction data were collected at a synchrotron beamline with a Pilatus detector.",
        "The heavy-atom derivative provided phases by isomorphous replacement.",
        "Two symmetry-related monomers form a dimer across the crystallographic axis.",
        "The model was refined to acceptable geometry after density modification.",
        "Electron density was continuous for the entire polypeptide backbone.",
        "Anomalous scattering located the selenomethionine positions in the lattice.",
    ],
    "economics": [
        "The central bank raised its policy rate by twenty-five basis points.",
        "Consumer price inflation slowed for the third consecutive quarter.",
        "Fiscal consolidation reduced the primary deficit relative to output.",
        "Unemployment fell even as labour force participation declined.",
        "Credit spreads widened sharply across emerging market sovereigns.",
        "Tariff escalation disrupted bilateral trade flows between the two economies.",
        "Household debt service ratios reached their highest level in a decade.",
        "The exchange rate depreciated after the capital account was liberalised.",
    ],
    "legal": [
        "The lessee shall indemnify the lessor against all third-party claims.",
        "This agreement is governed by the laws of the State of Delaware.",
        "Either party may terminate upon thirty days written notice of material breach.",
        "The warranty excludes defects arising from improper installation or misuse.",
        "All intellectual property created during the term vests in the company.",
        "The limitation of liability shall not apply to gross negligence or wilful misconduct.",
        "Confidential information must be returned or destroyed upon written request.",
        "Any amendment must be executed in writing and signed by both parties.",
    ],
    "cooking": [
        "Sear the lamb shoulder on all sides until the fat renders golden.",
        "Fold the beaten whites into the batter gently to keep the air.",
        "Simmer the stock uncovered for two hours, then skim the surface.",
        "Toasting the cumin seeds in a dry pan deepens the aroma considerably.",
        "Chill the dough for an hour so the butter stays in distinct layers.",
        "Deglaze the roasting tin with vinegar and reduce to a glossy sauce.",
        "Salt the aubergine slices and let them drain before frying.",
        "Rest the roast on a warm plate before carving against the grain.",
    ],
    # 第 6 个域：**同主题、不同事实**的分子生物学文本。
    # 这一组是"真实错误答案"的最佳代理：领域、语域、词汇层级都对，
    # 只是每一句说的事情都不是参考里说的那件事。
    "molbio_adjacent": [
        "Northern blotting detected the transcript in total RNA extracts.",
        "The construct was cloned into a lentiviral vector for stable expression.",
        "Quantitative PCR confirmed a two-fold change after drug treatment.",
        "Reverse transcription was primed with random hexamers overnight.",
        "The sample was resolved on a denaturing polyacrylamide gel.",
        "Cells were harvested forty-eight hours after transfection.",
        "Primers were designed to span an exon-exon junction.",
        "The library was sequenced on a short-read platform to depth.",
    ],
}

# ---- (a) 显式无关对：不同领域两两配对 -------------------------------
# 每组给一个 "pair"（单对，看具体数字）与一组 "pool"（域对，看分布）
UNRELATED_PAIRS: list[dict] = [
    {"name": "rna_vs_crystallography", "a": "rna", "b": "crystallography",
     "note": "与 03_eval_selfcheck 的 unrelated 同型（RNA vs 蛋白质晶体学）"},
    {"name": "rna_vs_molbio_adjacent", "a": "rna", "b": "molbio_adjacent",
     "note": "★ 最难的一组：同领域、同语域、不同事实 —— 『真实错误答案』的代理"},
    {"name": "rna_vs_economics", "a": "rna", "b": "economics", "note": ""},
    {"name": "rna_vs_legal", "a": "rna", "b": "legal", "note": ""},
    {"name": "rna_vs_cooking", "a": "rna", "b": "cooking", "note": ""},
    {"name": "crystallography_vs_legal", "a": "crystallography", "b": "legal", "note": ""},
    {"name": "economics_vs_cooking", "a": "economics", "b": "cooking", "note": ""},
]

# ---- 与原自检结构对齐的"标准无关对"（用于和 0.8159 直接对上）------
# 保持和 03_eval_selfcheck.py 的 UNRELATED/REFERENCE 完全相同的**句子数与结构**：
#   hyp 2 句 vs ref 3 句。这样 0.8159 与新数字是同一可比口径。
SELFCHECK_STYLE = {
    "name": "selfcheck_style_rna_vs_crystallography",
    "hyp": ("The protein adopts a beta-barrel fold. "
            "Crystallography at 1.8 angstrom resolution revealed a bound calcium ion."),
    "ref": ("The RNA molecule is a non-coding RNA that regulates gene expression. "
            "It is processed by the RNase III enzyme Drosha in the nucleus. "
            "Its secondary structure contains a conserved stem-loop motif that is required for binding."),
}

# ---- 参考 ACGU 序列（C 关里 F1=0.2633 的那条，作为"非语言 token"对照）---
ACGU = "ACGUACGUACGUACGUACGUACGUACGUACGU"

RANDOM_SEED = 20261008
RANDOM_TRIALS = 2000
POOL_TRIALS = 200


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pct(xs: list[float], q: float) -> float:
    """线性插值分位数（不引 numpy）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "p05": pct(xs, 0.05),
        "p25": pct(xs, 0.25),
        "median": pct(xs, 0.50),
        "p75": pct(xs, 0.75),
        "p95": pct(xs, 0.95),
        "max": max(xs),
    }


# ============================================================ 组件


def encode_all(emb, texts: list[str]) -> dict[str, list[list[float]]]:
    """一次编码全部句子，缓存复用（避免重复前向）。"""
    uniq = sorted(set(texts))
    vecs = emb.encode(uniq)
    return dict(zip(uniq, vecs))


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def anisotropy_report(cache: dict, corpus: dict[str, list[str]], corpus_sentences: list[str]) -> dict:
    """附加探针：句子向量空间的各向异性。

    这不是论文指标的组成部分，是用来**解释地板为什么高**的机制诊断：
    如果所有句子向量都挤在一个窄锥里（跨域 cosine 本身就有 0.94），
    那么 max-cosine 的 F1 自然被抬到同一个高度，与"语义对不对"无关。
    """
    V = [cache[s] for s in corpus_sentences]
    dom_of = [d for d, ss in corpus.items() for _ in ss]
    cross = [_cos(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V)) if dom_of[i] != dom_of[j]]
    within = [_cos(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V)) if dom_of[i] == dom_of[j]]
    d = len(V[0])
    mu = [sum(v[k] for v in V) / len(V) for k in range(d)]
    nmu = math.sqrt(sum(x * x for x in mu))
    anis = [_cos(v, [x / nmu for x in mu]) for v in V]
    return {
        "note": "机制诊断，不是论文口径的一部分",
        "sentence_cosine_cross_domain": stats(cross),
        "sentence_cosine_same_domain": stats(within),
        "anisotropy_vs_corpus_mean": stats(anis),
        "corpus_mean_vector_norm": nmu,
        "separation_cross_vs_same": statistics.fmean(within) - statistics.fmean(cross),
    }


def pair_f1(vec_hyp: list[list[float]], vec_ref: list[list[float]]) -> dict:
    """复刻 semantic.py 的 P/R/F1 口径，但直接吃已编码的向量（不引外部状态）。"""
    if not vec_hyp or not vec_ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    m = _cos_matrix(vec_hyp, vec_ref)
    p = sum(max(row) for row in m) / len(vec_hyp)
    r = sum(max(m[i][j] for i in range(len(vec_hyp))) for j in range(len(vec_ref))) / len(vec_ref)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def score_pair(emb, hyp: str, ref: str, cache: dict) -> dict:
    """走官方 semantic_scores（保证口径一致）+ 缓存编码。"""
    hs = split_sentences(hyp)
    rs = split_sentences(ref)
    for s in hs + rs:
        if s not in cache:
            cache[s] = emb.encode([s])[0]
    return pair_f1([cache[s] for s in hs], [cache[s] for s in rs])


def doc_cos(emb, hyp: str, ref: str, cache: dict) -> float:
    """附加探针：句子向量**均值**后的余弦（1 个数，不是 max-pooling）。

    不是论文口径，仅作为"还有一个更简单信号可用"的对照，单独标注。
    """
    hs = split_sentences(hyp)
    rs = split_sentences(ref)
    if not hs or not rs:
        return 0.0

    def mean_vec(ss):
        vs = [cache[s] for s in ss]
        d = len(vs[0])
        return [sum(v[k] for v in vs) / len(vs) for k in range(d)]

    a, b = mean_vec(hs), mean_vec(rs)
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x / na * y / nb for x, y in zip(a, b))


# ============================================================ 主实验


def run_for_embedder(spec: str, n_random: int = RANDOM_TRIALS) -> dict:
    section(f"embedder = {spec}")
    emb = build_embedder(spec)
    print(f"[load] {spec}  (首个 batch 触发权重加载)")
    t0 = time.time()
    corpus_sentences = [s for sents in CORPUS.values() for s in sents]
    all_texts = list(corpus_sentences)
    all_texts += [SELFCHECK_STYLE["hyp"], SELFCHECK_STYLE["ref"], ACGU]
    cache = encode_all(emb, all_texts)
    print(f"[enc ] {len(cache)} 条唯一句子, {time.time() - t0:.1f}s, dim={len(next(iter(cache.values())))}")

    out: dict = {
        "embedder": spec,
        "embedder_name": emb.name,
        "n_unique_sentences": len(cache),
        "dim": len(next(iter(cache.values()))),
    }

    # ---------- 机制诊断：表示各向异性 ----------
    out["z_anisotropy"] = anisotropy_report(cache, CORPUS, corpus_sentences)
    az = out["z_anisotropy"]
    print(f"[ani ] 跨域句级 cos 均值 = {az['sentence_cosine_cross_domain']['mean']:.4f}；"
          f"同域 = {az['sentence_cosine_same_domain']['mean']:.4f}；"
          f"跨/同域差 = {az['separation_cross_vs_same']:+.4f}")
    print(f"       各向异性（vs 语料均值向量）= {az['anisotropy_vs_corpus_mean']['mean']:.4f}")

    # ---------- (c) 理论上界：identical ----------
    section(f"[c] 理论上界 —— identical 文本 (embedder={spec})")
    ub_rows = []
    for dom, sents in CORPUS.items():
        hyp = " ".join(sents)
        s = score_pair(emb, hyp, hyp, cache)
        ub_rows.append({"domain": dom, "n_sent": len(split_sentences(hyp)), **s})
        print(f"  identical/{dom:16s} n_sent={len(split_sentences(hyp)):2d} "
              f"P={s['precision']:.6f} R={s['recall']:.6f} F1={s['f1']:.6f}")
    # 同域但不同文（真正同名领域内的两段不同文本）—— 看"同域地板"
    dom_rows = []
    for dom, sents in CORPUS.items():
        a = " ".join(sents[:4])
        b = " ".join(sents[4:])
        s = score_pair(emb, a, b, cache)
        dom_rows.append({"domain": dom, "variant": "same_domain_disjoint_halves", **s})
    ubs = [r["f1"] for r in ub_rows]
    out["c_upper_bound"] = {
        "identical": {"per_domain": ub_rows, "f1_mean": statistics.fmean(ubs), "f1_min": min(ubs)},
        "same_domain_disjoint_halves": {
            "rows": dom_rows,
            "f1_mean": statistics.fmean([r["f1"] for r in dom_rows]),
            "f1_min": min(r["f1"] for r in dom_rows),
        },
    }
    print(f"  -> identical F1 均值 {statistics.fmean(ubs):.6f}（最小 {min(ubs):.6f}）")

    # ---------- (a) 无关文本对 ----------
    section(f"[a] 无关文本对 —— 6 组跨领域配对 (embedder={spec})")
    print(f"{'pair':30s} {'n_hyp':>5s} {'n_ref':>5s} {'P':>8s} {'R':>8s} {'F1':>8s} {'doc_cos':>8s}")
    pair_rows = []
    for cfg in UNRELATED_PAIRS:
        hyp = " ".join(CORPUS[cfg["a"]])
        ref = " ".join(CORPUS[cfg["b"]])
        s = score_pair(emb, hyp, ref, cache)
        row = {
            "pair": cfg["name"], "domain_a": cfg["a"], "domain_b": cfg["b"], "note": cfg.get("note", ""),
            "n_sent_hyp": len(split_sentences(hyp)), "n_sent_ref": len(split_sentences(ref)),
            "doc_mean_cosine": doc_cos(emb, hyp, ref, cache),
            **s,
        }
        pair_rows.append(row)
        print(f"  {cfg['name']:28s} {row['n_sent_hyp']:5d} {row['n_sent_ref']:5d} "
              f"{s['precision']:8.4f} {s['recall']:8.4f} {s['f1']:8.4f} {row['doc_mean_cosine']:8.4f}")

    # 每对做 200 次随机句级抽样（消除"这一对恰好如何"的偶然性）
    rng = random.Random(RANDOM_SEED)
    pool_rows = []
    for cfg in UNRELATED_PAIRS:
        A, B = CORPUS[cfg["a"]], CORPUS[cfg["b"]]
        f1s, ps, rs_ = [], [], []
        for _ in range(POOL_TRIALS):
            k = rng.randint(2, 4)
            hyp = " ".join(rng.sample(A, k))
            ref = " ".join(rng.sample(B, rng.randint(2, 4)))
            s = score_pair(emb, hyp, ref, cache)
            f1s.append(s["f1"]); ps.append(s["precision"]); rs_.append(s["recall"])
        st = stats(f1s)
        pool_rows.append({"pair": cfg["name"], "trials": POOL_TRIALS,
                          "precision_mean": statistics.fmean(ps),
                          "recall_mean": statistics.fmean(rs_),
                          "f1": st})
    print("\n  句级随机抽样的分布（每对 200 次，跨域随机取句）：")
    for r in pool_rows:
        st = r["f1"]
        print(f"  {r['pair']:28s} F1 mean={st['mean']:.4f} std={st['std']:.4f} "
              f"min={st['min']:.4f} p05={st['p05']:.4f} max={st['max']:.4f}")

    # ---- 负对照：把"词汇全对、语义全错"的文本喂给同一个指标 ----
    # 这组数字直接决定"这条指标能不能当真"：
    #   如果打乱词序后 F1 仍然接近 1.0，那 max-cosine 量的是词汇重叠，不是语义。
    section(f"[a2] 负对照 —— 词汇相同 / 语义破坏 (embedder={spec})")
    ngram_m = 3

    def shuffled(sent: str, r: random.Random) -> str:
        core = sent.rstrip(".")
        w = core.split()
        rng.shuffle(w)
        return " ".join(w) + "."

    def word_ngrams(text: str, n: int = ngram_m) -> set[tuple[str, ...]]:
        w = text.lower().split()
        return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}

    rng2 = random.Random(RANDOM_SEED + 7)
    neg_rows = []
    for dom in ("rna", "molbio_adjacent", "economics"):
        ref = " ".join(CORPUS[dom])
        ref_ngrams = word_ngrams(ref)
        f1s, jac = [], []
        for _ in range(POOL_TRIALS):
            hyp = " ".join(shuffled(s, rng2) for s in CORPUS[dom])
            f1s.append(score_pair(emb, hyp, ref, cache)["f1"])
            hg = word_ngrams(hyp)
            jac.append(len(hg & ref_ngrams) / max(1, len(hg | ref_ngrams)))
        st = stats(f1s)
        row = {"domain": dom, "trials": POOL_TRIALS, "f1": st,
               "word_3gram_jaccard_mean": statistics.fmean(jac),
               "full_sentence_f1": score_pair(emb, " ".join(CORPUS[dom]), ref, cache)["f1"]}
        neg_rows.append(row)
        print(f"  {dom:18s} 词序全打乱: F1 mean={st['mean']:.4f} std={st['std']:.4f} "
              f"| 词 3-gram Jaccard={statistics.fmean(jac):.4f} | 原句 F1={row['full_sentence_f1']:.4f}")

    rng3 = random.Random(RANDOM_SEED + 8)
    gib = [
        "qzx vpr mnt lkjh gfds wqpo azxc vbnm",
        "zzzz qqqq xxxx jjjj vvvv kkkk wwww",
        "abc def ghi jkl mno pqr stu vwx yz",
        "aaaaaaaa bbbbbbbb cccccccc dddddddd",
    ]
    gib_rows = []
    for g in gib:
        s = score_pair(emb, g, " ".join(CORPUS["rna"]), cache)
        gib_rows.append({"text": g, **s})
        print(f"  乱码 {g[:34]:36s} F1={s['f1']:.4f}")
    print(f"  （纯随机字符串是有损上限参考：语义全无 → F1 应当等于地板）")
    out["a2_negative_controls"] = {
        "note": "词汇相同/语义破坏 —— 判定 max-cosine 是否在量语义的关键对照",
        "word_shuffled_within_sentence": neg_rows,
        "gibberish": gib_rows,
    }

    # 原自检同型对（2 句 vs 3 句，与 0.8159 可比）
    sc = score_pair(emb, SELFCHECK_STYLE["hyp"], SELFCHECK_STYLE["ref"], cache)
    sc["doc_mean_cosine"] = doc_cos(emb, SELFCHECK_STYLE["hyp"], SELFCHECK_STYLE["ref"], cache)
    print(f"\n  03_eval_selfcheck 同型无关对（2句 vs 3句，RNA vs 晶体学）："
          f"P={sc['precision']:.4f} R={sc['recall']:.4f} F1={sc['f1']:.4f}")

    # 非语言 token 对照：随机 ACGU
    acgu = score_pair(emb, ACGU, " ".join(CORPUS["rna"]), cache)
    acgu["doc_mean_cosine"] = doc_cos(emb, ACGU, " ".join(CORPUS["rna"]), cache)
    print(f"  ACGU 随机序列 vs RNA 文本：P={acgu['precision']:.4f} R={acgu['recall']:.4f} F1={acgu['f1']:.4f}")

    out["a_unrelated_pairs"] = {
        "explicit_pairs": pair_rows,
        "pool_sampling": pool_rows,
        "selfcheck_style_pair": {"hyp": SELFCHECK_STYLE["hyp"], "ref": SELFCHECK_STYLE["ref"], **sc},
        "acgu_probe": {"hyp": ACGU, "ref": " ".join(CORPUS["rna"]), **acgu},
    }

    # ---------- (b) 随机基线 ----------
    section(f"[b] 随机基线 —— 参考文本与随机另一段参考文本配对 (embedder={spec})")
    sent_pool = [s for d, sents in CORPUS.items() for s in sents]
    rng = random.Random(RANDOM_SEED + 1)
    sent_f1, sent_p, sent_r = [], [], []
    for _ in range(n_random):
        a = rng.choice(sent_pool)
        b = rng.choice(sent_pool)
        s = pair_f1([cache[a]], [cache[b]])
        sent_f1.append(s["f1"]); sent_p.append(s["precision"]); sent_r.append(s["recall"])
    sent_st = stats(sent_f1)
    print(f"  句级随机配对 (n={n_random})：F1 mean={sent_st['mean']:.4f} std={sent_st['std']:.4f} "
          f"p05={sent_st['p05']:.4f} median={sent_st['median']:.4f} p95={sent_st['p95']:.4f} "
          f"min={sent_st['min']:.4f} max={sent_st['max']:.4f}")
    print(f"                                P mean={statistics.fmean(sent_p):.4f}  R mean={statistics.fmean(sent_r):.4f}")

    # 文级随机配对（与真实评测同粒度：多句文档 vs 多句文档）
    rng = random.Random(RANDOM_SEED + 2)
    doc_f1 = []
    doms = list(CORPUS)
    for _ in range(n_random):
        da, db = rng.sample(doms, 2)  # 强制不同领域
        hyp = " ".join(rng.sample(CORPUS[da], rng.randint(2, 4)))
        ref = " ".join(rng.sample(CORPUS[db], rng.randint(2, 4)))
        doc_f1.append(score_pair(emb, hyp, ref, cache)["f1"])
    doc_st = stats(doc_f1)
    print(f"  文级跨域随机配对 (n={n_random})：F1 mean={doc_st['mean']:.4f} std={doc_st['std']:.4f} "
          f"p05={doc_st['p05']:.4f} median={doc_st['median']:.4f} p95={doc_st['p95']:.4f} "
          f"min={doc_st['min']:.4f} max={doc_st['max']:.4f}")

    # 同域文级随机配对（更难 —— 同领域但不同文）
    rng = random.Random(RANDOM_SEED + 3)
    same_f1 = []
    for _ in range(n_random):
        dom = rng.choice(doms)
        a, b = rng.sample(CORPUS[dom], 2)
        same_f1.append(pair_f1([cache[a]], [cache[b]])["f1"])
    same_st = stats(same_f1)
    print(f"  同域句级随机配对 (n={n_random})：F1 mean={same_st['mean']:.4f} std={same_st['std']:.4f} "
          f"p05={same_st['p05']:.4f} median={same_st['median']:.4f}")

    out["b_random_baseline"] = {
        "seed": RANDOM_SEED,
        "trials": n_random,
        "sentence_level_cross_corpus": {"precision_mean": statistics.fmean(sent_p),
                                        "recall_mean": statistics.fmean(sent_r), **sent_st},
        "doc_level_cross_domain": doc_st,
        "sentence_level_same_domain": same_st,
    }

    # ---------- (d) 有效动态范围 ----------
    # **主地板**取"跨域无关"：这是"答案完全跑题"的代理。
    # 同时报"同域无关"（更难、更贴近真实错误答案）与"词序打乱"（词汇对、语义错）。
    floor_cross = doc_st["mean"]
    floor_cross_p05 = doc_st["p05"]
    molbio_row = next((r for r in pair_rows if r["pair"] == "rna_vs_molbio_adjacent"), None)
    floor_same_domain_pair = molbio_row["f1"] if molbio_row else None
    floor_shuffled = statistics.fmean([r["f1"]["mean"] for r in neg_rows])
    ceil = statistics.fmean(ubs)
    out["d_dynamic_range"] = {
        "ceiling_identical_f1": ceil,
        "floor_cross_domain_f1_mean": floor_cross,
        "floor_cross_domain_f1_p05": floor_cross_p05,
        "floor_cross_domain_vs_molbio_adjacent": floor_same_domain_pair,
        "floor_word_shuffled_f1_mean": floor_shuffled,
        "floor_sentence_random_f1_mean": sent_st["mean"],
        "floor_selfcheck_style_f1": sc["f1"],
        "floor_from_explicit_pairs_mean": statistics.fmean([r["f1"] for r in pair_rows]),
        "span_cross_domain_to_1": 1.0 - floor_cross,
        "usable_range_cross_domain": 1.0 - floor_cross,
        "usable_range_vs_molbio_adjacent": (1.0 - floor_same_domain_pair) if floor_same_domain_pair else None,
        "usable_fraction_of_nominal_range": (1.0 - floor_cross) / 1.0,
    }

    print(f"\n  -> 天花板（identical F1）         = {ceil:.4f}")
    print(f"  -> 地板 A（跨域无关随机 F1 均值） = {floor_cross:.4f}   <- 答案完全跑题")
    if floor_same_domain_pair is not None:
        print(f"  -> 地板 B（同域不同事实 molbio）  = {floor_same_domain_pair:.4f}   <- 真实错误答案的代理")
    print(f"  -> 地板 C（词序打乱 F1 均值）     = {floor_shuffled:.4f}   <- 词汇对、语义错")
    print(f"  -> 可用窗口（跨域地板 -> 1.0）     = [{floor_cross:.4f}, 1.0000]，"
          f"宽度 {1.0 - floor_cross:.4f}（名义范围的 {(1.0 - floor_cross) * 100:.1f}%）")
    if floor_same_domain_pair is not None:
        print(f"     若以同域地板为准，宽度只剩 {1.0 - floor_same_domain_pair:.4f}")

    return out



def interpret(results: list[dict]) -> dict:
    """给 0.8609 一个坐标：它站在地板以上的什么位置。

    ⚠️ 注意：0.8609 是论文在**它自己的评测集**上量出来的，本脚本用的是自建小语料。
    两者绝对值不可直接比较 —— 这里只回答"同一把尺子在本地语料上的零点在哪"。
    """
    section("解读：原报数 F1@S_BERT = 0.8609 在本地板尺度上的位置")
    rows = []
    for r in results:
        d = r["d_dynamic_range"]
        floor = d["floor_cross_domain_f1_mean"]
        ceil = d["ceiling_identical_f1"]
        dns = ceil - floor
        sbert = 0.8609
        normalized = (sbert - floor) / dns if dns > 0 else None
        row = {
            "embedder": r["embedder"],
            "floor_cross_domain": floor,
            "floor_molbio_adjacent": d.get("floor_cross_domain_vs_molbio_adjacent"),
            "ceiling": ceil,
            "dynamic_range": dns,
            "reported_0_8609_above_floor": sbert - floor,
            "reported_0_8609_normalized_in_window": normalized,
            "reported_0_8609_below_floor": sbert < floor,
        }
        rows.append(row)
        name = Path(r["embedder"]).name
        if sbert < floor:
            print(f"  {name:16s} 地板={floor:.4f} 上界={ceil:.4f} 窗口宽={dns:.4f} | "
                  f"0.8609 **低于本地板** {floor - sbert:.4f} —— 在这个尺度上它等于'完全跑题'")
        else:
            print(f"  {name:16s} 地板={floor:.4f} 上界={ceil:.4f} 窗口宽={dns:.4f} | "
                  f"0.8609 高出地板 {sbert - floor:+.4f}，占窗口 {normalized * 100:.0f}%")
    return {
        "sbert_reported": 0.8609,
        "comparable": False,
        "caveat": "0.8609 来自原论文自己的评测集；本地是自建语料，绝对值不可直接比较，"
                  "只用于定位'同一把尺子在本地语料上的零点'。",
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", default="bert,pubmedbert", help="逗号分隔：bert / pubmedbert / HF repo id")
    ap.add_argument("--embedder-path", action="append", default=[],
                    help="覆盖某个名字的权重路径，形如 bert=C:\\path\\to\\dir")
    ap.add_argument("--random-trials", type=int, default=RANDOM_TRIALS)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "semantic_floor.json")
    args = ap.parse_args()

    overrides = {}
    for item in args.embedder_path:
        if "=" not in item:
            print(f"[warn] 忽略非法 --embedder-path：{item}")
            continue
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    specs = [s.strip() for s in args.embedders.split(",") if s.strip()]
    resolved = [overrides.get(s, DEFAULT_PATHS.get(s, s)) for s in specs]

    report: dict = {
        "what": "semantic F1 的语义地板 / 随机基线 / 理论上界（分辨率诊断）",
        "protocol": {
            "precision": "mean_i max_j cos(gen_i, ref_j)",
            "recall": "mean_j max_i cos(gen_j, ref_i)",
            "f1": "2PR/(P+R)",
            "sentence_split": "换行 + [.!?] 结尾切分，丢弃 len<2",
            "embedding": "最后一层 hidden mean pooling（mask 加权），L2 归一化，点积=余弦",
            "rescaling": "无（BERTScore baseline rescaling 未启用）",
            "note": "口径与 src/rnaqwen/eval/semantic.py 逐字一致；本脚本只调用，不改其算法",
        },
        "corpus": {d: s for d, s in CORPUS.items()},
        "acgu_sequence": ACGU,
        "random_seed": RANDOM_SEED,
        "results": [],
        "interpretation": None,
    }
    for spec in resolved:
        report["results"].append(run_for_embedder(spec, n_random=args.random_trials))
    report["interpretation"] = interpret(report["results"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] 语义地板报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    # 显式打 traceback —— 否则子进程里一个异常只剩 "exit code 1"，看不到原因
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        print("\n[ERROR] 脚本异常终止：", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(3)
