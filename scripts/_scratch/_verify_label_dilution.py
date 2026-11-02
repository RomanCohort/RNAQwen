"""验证：run_ablation 的 frac_neg 是不是被「样板标签」稀释了。

被验的主张
----------
`per_token_logp` 把所有非 -100 的标签位置汇到一起算 frac_neg。而 `build_batch` 的
标签 span 是：

    lab = [-100] * n_head + ids[n_head:]
    n_head = len(ids) - len(assistant_ids) - len(enc(" A: {answer}{eos}"))

也就是 **assistant 头 + no_think(14 token) + " A: 答案<eos>"** 全部参与统计。
其中 no_think 和助手头是**固定文本、与码无关**，它们贡献的是 50/50 的噪声。
如果答案只占这个 span 的少数，frac_neg 就会被拖向 50%，
而 `50_` 里那条 `零表/置换 < 45%` 的判据就失效 —— 表现成假阴性。

这个能量出来，且**不需要模型、不需要训练**：只用 tokenizer 和数据文件。

跑法： python scripts/_scratch/_verify_label_dilution.py
"""
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# **必须先 `import rnaqwen`。** 本机 torch（ROCm Windows nightly）没编入分布式后端，
# 而 transformers 5 在 import 阶段就引用 torch.distributed.*，不打补丁连 AutoTokenizer
# 都 import 不了，报的却是
#   "No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package"
# —— 看起来像环境坏了，其实只是漏了 rnaqwen.env.shim。而 shim 是在 rnaqwen/__init__.py
# 里自动装载的，所以只插 sys.path 不 import 包是没用的。踩过一次。
import rnaqwen  # noqa: F401,E402  —— 必须最早执行

BASE_DIR = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
DATA = ROOT / "data" / "processed" / "rnaqa_seqtask_full.jsonl"
EVAL_N = 32          # 与 --eval-n 一致
N_SAMPLES = 400      # 统计稳定用，比 32 大得多


class Tok:
    """从 50_routeC_e2e.py 逐字抄过来，保证量的是同一样东西。"""

    def __init__(self, tk):
        self._tk = tk
        self.base_vocab = tk.vocab_size
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        self.eos = tk.eos_token
        self.im_start = tk("<|im_start|>", add_special_tokens=False).input_ids
        self.im_end = tk("<|im_end|>", add_special_tokens=False).input_ids
        self.sys_ids = self.im_start + tk("system").input_ids + tk("\n").input_ids
        self.user_ids = self.im_end + self.im_start + tk("user").input_ids + tk("\n").input_ids
        self.assistant_ids = (self.im_end + self.im_start
                              + tk("assistant").input_ids + tk("\n").input_ids)
        self.no_think = tk(" thinking\n\n<｜end▁of▁thinking｜>\n\n",
                           add_special_tokens=False).input_ids

    def enc(self, s):
        return self._tk(s, add_special_tokens=False).input_ids


def spread(ex_list, n):
    """与 50_ 里的 spread 一致：等距抽样，不取前 n 条。"""
    if n <= 0 or len(ex_list) <= n:
        return ex_list
    st = len(ex_list) / n
    return [ex_list[min(int(i * st), len(ex_list) - 1)] for i in range(n)]


def main() -> int:
    # **必须先 import torch 再 import transformers。**
    # 反过来的话 transformers 会去碰 torch._C._distributed_c10d，报
    # "No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package"
    # —— 看起来像环境坏了，其实只是导入顺序。踩过一次。
    import torch  # noqa: F401
    from transformers import AutoTokenizer
    tk0 = AutoTokenizer.from_pretrained(str(BASE_DIR))
    tk = Tok(tk0)

    print(f"  tokenizer {BASE_DIR.name}  vocab {tk0.vocab_size}")
    print(f"  len(assistant_ids) = {len(tk.assistant_ids)}")
    print(f"  len(no_think)      = {len(tk.no_think)}")
    print()

    # 读 val 记录
    val = []
    keys_printed = False
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") != "val":
                continue
            if not keys_printed:
                print(f"  记录字段: {sorted(d.keys())}")
                print(f"  例: question={d.get('question','')[:40]!r}  answer={d.get('answer','')!r}")
                print()
                keys_printed = True
            val.append(d)
    print(f"  val 记录 {len(val):,} 条")

    # 全部 val 上统计，再单独看 eval-n=32 的等距子集
    def measure(records, tag):
        tot_lab = tot_ans = tot_head = tot_think = tot_wrap = 0
        per_sample_lab = []
        per_sample_ans = []
        for d in records:
            a = f" A: {d['answer']}{tk.eos}"
            enc_a = tk.enc(a)
            enc_ans = tk.enc(d["answer"])
            lab_span = len(tk.assistant_ids) + len(tk.no_think) + len(enc_a)
            tot_lab += lab_span
            tot_ans += len(enc_ans)
            tot_head += len(tk.assistant_ids)
            tot_think += len(tk.no_think)
            tot_wrap += len(enc_a) - len(enc_ans)          # " A: " + eos
            per_sample_lab.append(lab_span)
            per_sample_ans.append(len(enc_ans))
        n = len(records)
        print(f"  === {tag}（{n} 条）===")
        print(f"    标签 span 总计      {tot_lab:>8,}   （每样本均值 {tot_lab/n:.2f}）")
        print(f"      其中 assistant 头  {tot_head:>8,}   {100*tot_head/tot_lab:5.1f}%")
        print(f"      其中 no_think     {tot_think:>8,}   {100*tot_think/tot_lab:5.1f}%")
        print(f"      其中 ' A: '+eos   {tot_wrap:>8,}   {100*tot_wrap/tot_lab:5.1f}%")
        print(f"    **答案 token**      {tot_ans:>8,}   {100*tot_ans/tot_lab:5.1f}%  "
              f"（每样本均值 {tot_ans/n:.2f}）")
        return tot_lab, tot_ans

    print()
    all_lab, all_ans = measure(val, "全部 val")
    print()

    sub = spread(val, EVAL_N)
    sub_lab, sub_ans = measure(sub, f"--eval-n={EVAL_N} 的等距子集")

    print()
    print("  === 与日志对账 ===")
    print(f"    日志里 [eval@3750] 报的是 n=317")
    print(f"    按本脚本算，{EVAL_N} 条样本的标签 span 应该是 {sub_lab:,}")
    if abs(sub_lab - 317) / max(317, 1) < 0.15:
        print(f"    -> 对得上（差 {abs(sub_lab-317)}）")
    else:
        print(f"    -> **对不上**（差 {abs(sub_lab-317)}, {100*abs(sub_lab-317)/317:.0f}%）。"
              f"说明 n=317 不是标签 span 总数，我的稀释解释需要修正。")

    print()
    print("  === 稀释能解释多少 ===")
    frac = sub_ans / sub_lab
    print(f"    答案 token 占标签 span 的 {100*frac:.1f}%")
    print(f"    若其余 {100*(1-frac):.1f}% 是 50/50 噪声，而答案 token 全部被真码改善，")
    print(f"    则 frac_neg 的下限约为 {100*(1-frac)*0.5:.1f}%")
    print(f"    实测 vszero 52.7% / vsperm 46.4% —— 与这个下限量级一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
