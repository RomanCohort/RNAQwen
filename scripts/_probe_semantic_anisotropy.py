"""一次性诊断：PubMedBERT / BERT 的表示各向异性 + MLM 合理性探针。

只读，不改任何东西。用来排除"model.safetensors 转换坏了"这个可能。
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

from rnaqwen.eval.semantic import build_embedder, split_sentences  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

CORPUS = import_module("03b_semantic_floor").CORPUS


def probe(path: str, label: str) -> None:
    print(f"\n{'-' * 70}\n{label}  ({path})\n{'-' * 70}", flush=True)
    emb = build_embedder(path)
    sents = [s for d in CORPUS.values() for s in d]
    V = emb.encode(sents)
    dom_of = [d for d, ss in CORPUS.items() for _ in ss]

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    cross = [cos(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V)) if dom_of[i] != dom_of[j]]
    within = [cos(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V)) if dom_of[i] == dom_of[j]]
    print(f"句子级 cosine 跨域: mean={statistics.fmean(cross):.4f} min={min(cross):.4f} max={max(cross):.4f}")
    print(f"句子级 cosine 同域: mean={statistics.fmean(within):.4f} min={min(within):.4f} max={max(within):.4f}")

    # 各向异性：句子向量 vs 全语料均值向量 的余弦（BERT-flow 论文的指标）
    d = len(V[0])
    mu = [sum(v[k] for v in V) / len(V) for k in range(d)]
    nmu = math.sqrt(sum(x * x for x in mu))
    anis = [cos(v, [x / nmu for x in mu]) for v in V]
    print(f"各向异性 (vs 语料均值向量): mean={statistics.fmean(anis):.4f} min={min(anis):.4f}")
    # 秩/维数健康度：句子向量集合的奇异值谱（防止坍缩成常数）
    n = len(V)
    mean_sub = [[V[i][k] - mu[k] for k in range(d)] for i in range(n)]
    # 用 Gram 矩阵近似（n x n）取特征值
    G = [[sum(mean_sub[i][k] * mean_sub[j][k] for k in range(d)) for j in range(n)] for i in range(n)]
    # 幂迭代取前 2 个特征值
    def power(G, iters=300):
        x = [1.0 / math.sqrt(n)] * n
        for _ in range(iters):
            y = [sum(G[i][j] * x[j] for j in range(n)) for i in range(n)]
            nrm = math.sqrt(sum(v * v for v in y)) or 1.0
            x = [v / nrm for v in y]
        lam = sum(x[i] * sum(G[i][j] * x[j] for j in range(n)) for i in range(n))
        return lam
    l1 = power(G)
    G2 = [[G[i][j] - l1 * 0 + 0 for j in range(n)] for i in range(n)]
    # 收缩第二个特征值：G - l1 * x x^T
    x1 = [1.0 / math.sqrt(n)] * n
    for _ in range(300):
        y = [sum(G[i][j] * x1[j] for j in range(n)) for i in range(n)]
        nrm = math.sqrt(sum(v * v for v in y)) or 1.0
        x1 = [v / nrm for v in y]
    for i in range(n):
        for j in range(n):
            G2[i][j] = G[i][j] - l1 * x1[i] * x1[j]
    l2 = power(G2)
    tot = sum(G[i][i] for i in range(n))
    print(f"句向量谱: top1={l1:.3f} top2={l2:.3f}  top1/总能量={l1 / tot:.4f}  (前两维占 {100 * (l1 + l2) / tot:.1f}%)")

    # MLM 合理性探针：生物医学 vs 通用
    print("\n[MLM 探针]")
    tok = AutoTokenizer.from_pretrained(path)
    mlm = AutoModelForMaskedLM.from_pretrained(path).eval()
    probes = [
        ("The patient was diagnosed with [MASK] cancer.", "生物医学"),
        ("The [MASK] molecule regulates gene expression.", "生物医学"),
        ("Crystallography at 1.8 [MASK] resolution.", "晶体学"),
    ]
    with torch.no_grad():
        for text, tag in probes:
            enc = tok(text, return_tensors="pt")
            out = mlm(**enc).logits
            pos = int((enc["input_ids"] == tok.mask_token_id).nonzero()[0, 1])
            top = torch.topk(out[0, pos], 5).indices.tolist()
            words = [tok.decode([t]).strip() for t in top]
            print(f"  [{tag}] {text}\n      -> {words}")


if __name__ == "__main__":
    probe(str(ROOT / "models" / "pubmedbert"), "PubMedBERT")
    probe(str(ROOT / "models" / "bert-base-uncased"), "bert-base-uncased")
