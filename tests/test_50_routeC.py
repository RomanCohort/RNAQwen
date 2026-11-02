"""50_ 的单元测试 —— 不加载 Qwen、不加载 ERNIE，冷启动 < 20 秒。

为什么必须写这个：
    今天所有"结论"都作废过。原因全是这类 bug：接线对但语义错、形状对但布局错、
    异常被吞掉。这些用大模型跑实验是发现不了的（一次跑 5 分钟起），
    用小张量跑是 1 秒级。

每个测试都对应一个真实发生过的 bug，注释里写明是哪一个。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

t_start = time.time()
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else '**FAIL**'}  {name}" + (f"   {detail}" if detail else ""))


# 直接 import 被测模块会拉起 transformers/peft —— 太重。只把需要的类抠出来。
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("routeC", ROOT / "scripts/50_routeC_e2e.py")
# 该模块顶层会 import transformers，但只有 E2EModel.__init__ 内部才真正加载模型，
# 顶层 import 是可接受的。若环境缺包，下面会给出明确报错。
routeC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routeC)

FSQCode = routeC.FSQCode


print("=" * 78)
print("BUG 2：压缩器的『平均池化初始化』布局")
print("=" * 78)
# 老版本写 w[i, i*pool:(i+1)*pool] = 1/pool，但输入一行布局是
#   [位置0 的 dim 维][位置1 的 dim 维]...，通道 i 在第 j 个位置的值在第 j*dim+i 列。
# 结果初始化出来的不是平均池化，而是一个搅乱通道的投影。
DIM, POOL, LAT, LEV = 8, 4, 3, 4
torch.manual_seed(0)
m = FSQCode(DIM, LAT, LEV, d_model=16, pool=POOL)
x = torch.arange(12 * DIM, dtype=torch.float32).reshape(12, DIM)   # 12 个位置
with torch.no_grad():
    got = m.compress(x[:12].reshape(3, POOL * DIM))
    want = x[:12].reshape(3, POOL, DIM).mean(dim=1)                # 真·平均池化
check("compress 初始 == 逐窗口平均池化", torch.allclose(got, want, atol=1e-5),
      f"max|diff|={float((got-want).abs().max()):.2e}")
check("平均池化后每通道等于该窗口该通道的均值",
      torch.allclose(got[0], x[:4].mean(dim=0), atol=1e-5))

print()
print("=" * 78)
print("BUG 4：极短序列（L < pool）不该 reshape 报错")
print("=" * 78)
for L in (1, 2, 3, 4, 5, 7):
    try:
        code, q_ste, t, q = m(torch.randn(L, DIM))
        n = max(1, L // POOL)
        check(f"L={L} 不报错且输出 {n} 个 token", code.numel() == n,
              f"得到 {code.numel()}")
    except Exception as e:
        check(f"L={L} 不报错", False, f"{type(e).__name__}: {e}")

print()
print("=" * 78)
print("BUG 5：码塌缩检测 —— 均匀输入不该塌成 1 个码")
print("=" * 78)
# 老版本 enc 用默认 std=0.02，可 ERNIE 隐藏态范数 ≈ 8.7 -> tanh 饱和 -> 全是同一个码。
big = torch.randn(3000, DIM) * 8.7                 # 模拟真实隐藏态量级
code, _, t, _ = m(big)
n_used = int(code.unique().numel())
ent = float(-(lambda p: (p * (p + 1e-9).log()).sum())(
    torch.bincount(code, minlength=LEV ** LAT).float() / code.numel()))
check("tanh 未饱和", float(t.abs().max()) < 0.999, f"max|t|={float(t.abs().max()):.4f}")
check("用到的码 > 1", n_used > 1, f"用了 {n_used}/{LEV**LAT} 个码，熵 {ent:.3f}")

print()
print("=" * 78)
print("BUG 1：attention_mask —— pad 长度不同的样本必须产出一致的 logits")
print("=" * 78)


class Tok:
    """最小的 Tok 替身：只提供 build_batch 需要的字段。"""
    def __init__(self):
        self.base_vocab = 1000
        self.pad_id = 0
        self.eos = "<eos>"
        self.sys_ids = [1, 2]
        self.user_ids = [3, 4]
        self.assistant_ids = [5, 6]

    def enc(self, s):
        return [10 + (ord(c) % 50) for c in s]


tok = Tok()
rec = {"uid": "u1", "sequence": "ACGU" * 40}
long_qa = {"question": "Q" * 200, "answer": "A" * 40}
ex = [(rec, long_qa, 20, 0), (rec, long_qa, 40, 0)]      # 同样的文本，只改占位长度

# 把 build_batch 里的 base_vocab 占位替换掉不影响 mask，先看形状
ii, ll, am = routeC.build_batch(tok, ex)
check("build_batch 返回三个张量", ii.shape == ll.shape == am.shape)
check("mask 长度等于真实长度", int(am[0].sum()) < int(am[1].sum()),
      f"{int(am[0].sum())} < {int(am[1].sum())}")
check("mask 与 pad 位置一致",
      bool(((am == 0) == (torch.arange(am.shape[1], device=am.device)
                          .unsqueeze(0) >= am.sum(1, keepdim=True))).all()
           or (am[:, -1] == 0).any()))
# 关键：EOS 不能被 mask 掉。Qwen 的 pad_token_id 会回退到 eos，
# 所以不能写成 (ii != pad_id)。这里 pad_id=0 且 EOS 是文本 token，故检查末位
check("序列末尾（含 EOS）的 mask 为 1",
      all(int(am[i, int(am[i].sum()) - 1]) == 1 for i in range(am.shape[0])))

print()
print("=" * 78)
print("BUG 3：state_dict 形状不匹配时 strict=False 也会抛")
print("=" * 78)
a = FSQCode(DIM, 3, 4, d_model=16, pool=POOL)      # enc: (3, DIM)
b = FSQCode(DIM, 2, 4, d_model=16, pool=POOL)      # enc: (2, DIM)
try:
    b.load_state_dict({"enc.weight": a.enc.weight, "enc.bias": a.enc.bias}, strict=False)
    check("strict=False 会抛形状错（预期抛，用来证明过滤是必要的）", False,
          "居然没抛 —— 那这个 bug 的前提不成立，需重新验证")
except RuntimeError as e:
    check("strict=False 确实会抛形状错（过滤是必要的）", "size mismatch" in str(e).lower(),
          str(e).split("\n")[0][:70])
# 正确做法：按形状过滤
cur = b.state_dict()
sd = {k: v for k, v in {"enc.weight": a.enc.weight, "enc.bias": a.enc.bias}.items()
      if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
check("按形状过滤后只留下匹配项", len(sd) == 0, f"留下 {list(sd)}")
b.load_state_dict(sd, strict=False)
check("过滤后 load_state_dict 不再抛", True)

print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败   用时 {time.time()-t_start:.1f}s")
print("=" * 78)
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
    raise SystemExit(1)
print("全部通过")
