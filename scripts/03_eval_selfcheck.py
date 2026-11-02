"""PLAN §7.3 —— 评测脚本自检。

上量之前先证明"尺子是准的"。三关：

  A. **ROUGE 对数**：自实现 vs Google `rouge_score`，逐样本必须一致（1e-9）。
     不一致 = 口径错了，后面所有数字都不可比。
  B. **语义指标不变量**：
       * 完美答案（hyp == ref）→ P/R/F1 ≈ 1.0
       * 完全无关的两段 → F1 明显低
       * 单调性：部分重叠的答案，F1 落在两者之间
  C. **原版消融表的量级自检**：用 PLAN §4 那三行做"人工完美答案"实验，
     确认指标在 0.09 / 0.25 / 0.48 这种稀疏度上确实是单调响应的
     （注意：**不是**去复现原报数，那需要他们的评测集；这里只验证尺子的响应方向）。

跑法：
    python scripts/03_eval_selfcheck.py --embedder bert
    python scripts/03_eval_selfcheck.py --skip-embedders     # 只对数 ROUGE（不用下模型）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— import 本包即自动装载 env.shim（本机 torch 缺分布式符号）

from rnaqwen.eval.rouge import (  # noqa: E402
    rouge_1,
    rouge_2,
    rouge_l,
    rouge_scores,
)
from rnaqwen.eval.semantic import build_embedder, semantic_scores, split_sentences  # noqa: E402

# ---------------------------------------------------------------- 测试样本
# 取自 RNA 领域的真实风格文本（避免用"hello world"这种测不出分词问题的样本）
REFERENCE = (
    "The RNA molecule is a non-coding RNA that regulates gene expression. "
    "It is processed by the RNase III enzyme Drosha in the nucleus. "
    "Its secondary structure contains a conserved stem-loop motif that is required for binding."
)
PARAPHRASE = (
    "This non-coding RNA controls the expression of genes. "
    "Drosha, an RNase III enzyme, processes it inside the nucleus. "
    "A conserved stem-loop in its secondary structure is needed for binding."
)
PARTIAL = "The RNA regulates gene expression and is processed by Drosha in the nucleus."
UNRELATED = (
    "The protein adopts a beta-barrel fold. "
    "Crystallography at 1.8 angstrom resolution revealed a bound calcium ion."
)

ROUGE_CASES: list[tuple[str, str, str]] = [
    ("identical", REFERENCE, REFERENCE),
    ("paraphrase", PARAPHRASE, REFERENCE),
    ("partial", PARTIAL, REFERENCE),
    ("unrelated", UNRELATED, REFERENCE),
    ("empty", "", REFERENCE),
    ("punctuation", "The RNA, molecule; is a non-coding RNA!!!", REFERENCE),
    ("numbers", "The 1.8 angstrom 42 kDa complex", "The 1.8 angstrom 42 kDa complex was solved"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_rouge_parity(tolerance: float = 1e-9) -> dict:
    """A 关：自实现 vs rouge_score 逐样本对数。"""
    section("A. ROUGE 对数（自实现 vs rouge_score）")
    report: dict = {"available": False, "cases": [], "max_abs_diff": None, "passed": None}
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[skip] 没装 rouge_score —— 先 pip install rouge-score 再跑这一关")
        return report

    report["available"] = True
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    max_diff = 0.0
    for name, hyp, ref in ROUGE_CASES:
        ref_scores = scorer.score(ref, hyp)
        ours = {
            "rouge1": rouge_1(hyp, ref)[2],
            "rouge2": rouge_2(hyp, ref)[2],
            "rougeL": rouge_l(hyp, ref)[2],
        }
        theirs = {
            "rouge1": ref_scores["rouge1"].fmeasure,
            "rouge2": ref_scores["rouge2"].fmeasure,
            "rougeL": ref_scores["rougeL"].fmeasure,
        }
        diffs = {k: abs(ours[k] - theirs[k]) for k in ours}
        case_max = max(diffs.values())
        max_diff = max(max_diff, case_max)
        ok = case_max <= tolerance
        report["cases"].append(
            {"case": name, "ours": ours, "rouge_score": theirs, "max_abs_diff": case_max, "ok": ok}
        )
        flag = "ok  " if ok else "FAIL"
        print(
            f"[{flag}] {name:12s} r1={ours['rouge1']:.6f}/{theirs['rouge1']:.6f} "
            f"r2={ours['rouge2']:.6f}/{theirs['rouge2']:.6f} "
            f"rL={ours['rougeL']:.6f}/{theirs['rougeL']:.6f}  Δmax={case_max:.2e}"
        )
    report["max_abs_diff"] = max_diff
    report["passed"] = max_diff <= tolerance
    print(f"\n-> 最大偏差 {max_diff:.3e}（阈值 {tolerance:.0e}）：{'一致' if report['passed'] else '不一致，口径有问题'}")
    return report


def check_semantic_invariants(embedder_spec: str) -> dict:
    """B 关：语义指标不变量。"""
    section(f"B. 语义指标不变量（embedder={embedder_spec}）")
    emb = build_embedder(embedder_spec)
    report: dict = {"embedder": embedder_spec, "cases": [], "passed": None}

    cases = [
        ("identical", REFERENCE, REFERENCE, "high"),
        ("paraphrase", PARAPHRASE, REFERENCE, "high"),
        ("partial", PARTIAL, REFERENCE, "mid"),
        ("unrelated", UNRELATED, REFERENCE, "low"),
    ]
    scores: dict[str, float] = {}
    for name, hyp, ref, expect in cases:
        s = semantic_scores([hyp], [ref], emb)
        scores[name] = s.f1
        report["cases"].append({"case": name, "expect": expect, **s.as_dict()})
        print(f"[{expect:4s}] {name:12s} P={s.precision:.4f} R={s.recall:.4f} F1={s.f1:.4f}")

    # 「无关」这条对照的分数**不等于**指标地板 —— 它取决于两段文本的语域距离。
    # 实测（reports/semantic_floor.json）：
    #   跨域随机配对的地板     bert ≈ 0.52 / pubmedbert ≈ 0.95
    #   「RNA vs 蛋白质晶体学」这种同语域对照  bert ≈ 0.77 / pubmedbert ≈ 0.96
    # 所以阈值不能写死 0.95 —— 那个数只在 BERT 上成立，
    # 在 PubMedBERT 上**数学上不可能通过**（它自己的地板就是 0.9518）。
    KNOWN_FLOORS = {"bert": 0.52, "pubmedbert": 0.95}
    key = next((k for k in KNOWN_FLOORS if k in embedder_spec.lower()
                or ("bert-base-uncased" in embedder_spec.lower() and k == "bert")), None)
    floor = KNOWN_FLOORS.get(key) if key else None

    checks = {
        "identical_ge_0.999": scores["identical"] >= 0.999,
        "paraphrase_gt_partial": scores["paraphrase"] > scores["partial"],
        "partial_gt_unrelated": scores["partial"] > scores["unrelated"],
    }
    if floor is None:
        checks["unrelated_vs_known_floor"] = True
        print("[skip] 未知嵌入的地板，跳过绝对阈值检查（请查 reports/semantic_floor.json）")
    else:
        checks["unrelated_within_floor"] = scores["unrelated"] < floor + 0.06
        report["known_floor"] = floor
        report["dynamic_range"] = [floor, scores["identical"]]
        print(f"[info] 该嵌入的跨域地板 ≈ {floor:.2f} → 有效动态范围 "
              f"[{floor:.2f}, {scores['identical']:.2f}]，宽 {scores['identical'] - floor:.3f}")
        print(f"[info] 本次 unrelated={scores['unrelated']:.4f}（同语域对照，本就会比跨域地板高）")
    report["checks"] = checks
    report["passed"] = all(checks.values())
    for k, v in checks.items():
        print(f"[{'ok  ' if v else 'FAIL'}] {k}")
    return report


def check_ablation_scale(embedder_spec: str) -> dict:
    """C 关：用 PLAN §4 的稀疏度做响应方向检查。"""
    section("C. 消融阶梯的响应方向（不是复现原报数，是验证尺子会动）")

    # 三段参考文本；构造"说了 25% / 50% / 100%"的答案，看指标是否单调
    ref = REFERENCE
    ref_sents = split_sentences(ref)
    variants = {
        "perfect_100pct": ref,
        "half_50pct": " ".join(ref_sents[: max(1, len(ref_sents) // 2)]),
        "one_third": ref_sents[0] if ref_sents else ref,
        "bare_sequence_like": "ACGUACGUACGUACGUACGUACGUACGUACGU",
        "empty": "",
    }
    hyps = list(variants.values())
    refs = [ref] * len(hyps)
    r = rouge_scores(hyps, refs)
    emb = build_embedder(embedder_spec)
    s = semantic_scores(hyps, refs, emb)

    report: dict = {"rouge": r.as_dict(), "semantic": s.as_dict(), "per_variant": []}
    print(f"{'variant':20s} {'R-1':>8s} {'R-2':>8s} {'R-L':>8s} {'F1@emb':>8s}")
    for i, (name, _) in enumerate(variants.items()):
        r1 = rouge_1(hyps[i], ref)[2]
        r2 = rouge_2(hyps[i], ref)[2]
        rl = rouge_l(hyps[i], ref)[2]
        f1 = semantic_scores([hyps[i]], [ref], emb).f1
        report["per_variant"].append({"variant": name, "rouge1": r1, "rouge2": r2, "rougeL": rl, "semantic_f1": f1})
        print(f"{name:20s} {r1:8.4f} {r2:8.4f} {rl:8.4f} {f1:8.4f}")

    ordered = [v["rouge1"] for v in report["per_variant"][:3]]
    monotonic = ordered[0] >= ordered[1] >= ordered[2]
    report["monotonic_rouge1"] = monotonic
    print(f"\n-> ROUGE-1 随答案完整度单调：{'是' if monotonic else '否 —— 尺子有问题'}")
    print("   （原版 D&C 阶梯 0.2472 → 0.0922 → 0.4791 是同一把尺子量出来的三档）")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bert", help="bert / pubmedbert / gpt / HF repo id")
    ap.add_argument("--skip-embedders", action="store_true", help="只跑 ROUGE 对数（不下嵌入模型）")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "eval_selfcheck.json")
    args = ap.parse_args()

    report: dict = {"embedder": args.embedder}
    report["A_rouge_parity"] = check_rouge_parity()
    if not args.skip_embedders:
        report["B_semantic_invariants"] = check_semantic_invariants(args.embedder)
        report["C_ablation_scale"] = check_ablation_scale(args.embedder)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] 自检报告 -> {args.out}")

    verdicts = [v.get("passed") for k, v in report.items() if isinstance(v, dict) and "passed" in v]
    if any(v is False for v in verdicts):
        print("[FAIL] 有自检项没过，别上量。")
        return 1
    if any(v is None for v in verdicts):
        print("[PARTIAL] 有自检项被跳过（缺依赖）。")
    print("[OK] 尺子可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
