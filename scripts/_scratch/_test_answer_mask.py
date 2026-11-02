"""验 answer_token_mask：掩码选中的到底是不是答案 token。

为什么值得单测：这个掩码是判据的根。它错一位，`frac_neg` 就又变成噪声，
而噪声是**静默**的 —— 曲线照画、数字照出，只是判据失效。
所以不验"长度对不对"，直接**把选中的 token 解回字符串，看是不是答案**。

顺便钉住那处已知的错位：
  · emask 在 lab 的坐标上，而 logp 在 tgt 的坐标上（tgt[j] = lab[j+1]），必须错开一位；
  · BPE 会在边界上合并，所以不能用 len(enc(" A: ")) + len(enc(ans)) 去截。

跑法： python scripts/_scratch/_test_answer_mask.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402  —— 必须最早，装 torch.distributed 补丁

FAILS = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "route_c_trainer", ROOT / "scripts" / "50_routeC_e2e.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    m = load_trainer()
    tk = m.Tok(AutoTokenizer.from_pretrained(r"C:\qwen-api\models\Qwen3.6-35B-A3B"))

    print(f"  len(assistant_ids)={len(tk.assistant_ids)}  len(no_think)={len(tk.no_think)}")
    print()

    # 覆盖短答案/长答案/含空格/含大写与小写
    cases = [
        ("CUGC", "What are the first 4 nucleotides of this RNA?"),
        ("GA", "What is the second nucleotide?"),
        ("AUGCAUGCAUGC", "Give me the first 12 nucleotides."),
        ("UUUU", "How many uracils are in the first 4 positions?"),
        ("GC", "What is the last dinucleotide?"),
    ]

    print("== 1. 掩码选中的 token 解出来必须是答案本身 ==")
    for ans, q in cases:
        qa = {"answer": ans, "question": q}
        ids, lab, am = m.build_batch(tk, [({}, qa, 0, 0)])
        mask = m.answer_token_mask(tk, qa, lab)
        got = tk._tk.decode(ids[0][mask[0]].tolist()).strip()
        n_ans = len(tk.enc(ans))
        n_eos_in_mask = int(mask[0, -1])
        check(got == ans, f"答案 {ans!r:<14} 解出 {got!r:<14} "
                          f"（{int(mask.sum())} token, enc(答案)={n_ans}）")
        check(n_eos_in_mask == 0, f"          eos 未被算进答案（最后一位={n_eos_in_mask}）")

    print("\n== 2. 掩码落在标签范围内，且不是全部标签 ==")
    for ans, q in cases[:3]:
        qa = {"answer": ans, "question": q}
        ids, lab, am = m.build_batch(tk, [({}, qa, 0, 0)])
        mask = m.answer_token_mask(tk, qa, lab)
        n_lab = int((lab >= 0).sum())
        n_ans = int(mask.sum())
        # 标签从 no_think 尾巴开始，所以答案一定**少于**全部标签
        check(0 < n_ans < n_lab,
              f"答案 {ans!r:<14} 掩码 {n_ans} / 标签 {n_lab} 个 "
              f"（答案占 {100*n_ans/n_lab:.0f}%）")
        check(bool((lab[mask] >= 0).all()), f"          掩码只落在有标签的位置上")

    print("\n== 3. 错开一位：emask[:, 1:] 与 logp 的坐标对齐 ==")
    for ans, q in cases[:3]:
        qa = {"answer": ans, "question": q}
        ids, lab, am = m.build_batch(tk, [({}, qa, 0, 0)])
        mask = m.answer_token_mask(tk, qa, lab)
        tgt = lab[:, 1:]
        m_ok = tgt >= 0
        shifted = mask[:, 1:] & m_ok
        check(int(shifted.sum()) == int(mask.sum()),
              f"答案 {ans!r:<14} 移位后命中 {int(shifted.sum())} 个，与未移位一致"
              f"（{int(mask.sum())}）")
        # 解出来的仍应是答案：tgt 位置 j 的"预测目标"就是 seq[j+1]
        got = tk._tk.decode(ids[0, 1:][mask[0, 1:]].tolist()).strip()
        check(got == ans, f"          移位后解出 {got!r}")

    print("\n== 4. 标签 span 就是本脚本说的那 9.89 个（对账 n=317）==")
    spans = []
    enc_a = []
    for ans, q in cases:
        qa = {"answer": ans, "question": q}
        ids, lab, am = m.build_batch(tk, [({}, qa, 0, 0)])
        spans.append(int((lab >= 0).sum()))
        enc_a.append(len(tk.enc(f" A: {ans}{tk.eos}")))
    check(all(s == len(tk.assistant_ids) + e for s, e in zip(spans, enc_a)),
          f"标签 span = len(assistant_ids) + len(enc(a)) "
          f"= {len(tk.assistant_ids)} + {enc_a} = "
          f"{[len(tk.assistant_ids) + e for e in enc_a]}")
    print(f"    实测 span {spans}")
    print(f"    -> assistant 头那 {len(tk.assistant_ids)} 个 token **不在**标签里"
          f"（算式 L - len(assistant_ids) - len(enc(a)) 漏了 len(no_think)={len(tk.no_think)}）")

    print()
    if FAILS:
        print(f"===== {len(FAILS)} 项失败 =====")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("===== 全部通过 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
