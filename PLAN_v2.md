# PLAN v2 —— 放弃 RNA-GPT 软提示路线,转向原生文本 + 序列继续预训练

**决策日期**:2026-10-10
**决策**:彻底放弃「冻结编码器 + 投影 → 冻结 LLM」这条路线。
**性质变更**:**本项目不再是 RNA-GPT 的复现。** 它是「在 RNA-QA 这个任务上,软提示路线 vs 原生文本路线的头对头对比」。

> 原 `PLAN.md` 保留不动,作为**被证伪路线的完整记录**。它的价值在于:我们不是"做不下去才换",
> 是**逐条测出了那条路的天花板**,并且每一条都有可复现的实验支撑。

---

## 1. 为什么转向 —— 全部来自我们自己的测量

### 1.1 已证伪的四个假设

| # | 假设 | 检验方式 | 结果 |
|---|---|---|---|
| 1 | 投影在丢信息 | 32_proj_discriminability:减掉跨序列公共分量后算不同 RNA 的相似度 | ❌ **否掉**。inter = **-0.05**,信息完整且高度可分 |
| 2 | 学习式压缩器(resampler)造成稀疏/偏执依赖 | 33_token_modality_relevance 对比 resampler vs meanpool | ❌ **否掉**。两者逐 token 剖面几乎一致(负值占比 58.4% vs 63.3%) |
| 3 | val_loss 能选出更好的投影 | 两组配对对比(压缩/不压缩、resampler/meanpool) | ❌ **否掉**。两次都是 val 低 ↔ grounding 低 |
| 4 | 编码器各向异性是主因 | 34+去均值变体,500 步配对 | ❌ **否掉**。val 打平(1.4892 vs 1.4926),Δzeroed 低 8× 但消融口径不同、不可直接比 |

### 1.2 已确认的三条事实

```
① ERNIE-RNA 原始输出模长的 91.1% 落在跨序列公共方向上
   压缩器放大到 93.7%，投影放大到 97.2%
   RNA 特异能量沿链路：40.0% → 31.5% → 21.1%

② 逐 token 看，模态依赖是重尾的，且多数为负
   KL 中位 0.039（七成 token 几乎不依赖 RNA）
   KL p90 0.226 / max 10.1（少数位置极强）
   Δlogp 负值占比 58.4%（多数位置上编码器是负担）

③ Qwen3 原生 tokenizer 切 RNA 是 0.526 tokens/nt（~1.9 nt/token）
   切分**局部稳定**（单碱基扰动的中位影响跨度 0 nt，最大 4 nt）
   切分确定（三次相同），上下文依赖存在但前缀固定后无影响
```

### 1.3 四条外部证据,同一结论

| 来源 | 原话 / 结论 |
|---|---|
| **ChatMultiOmics**<br>arXiv 2412.19191 | "directly applying LoRA to train a chat model on Biology-Instructions results in suboptimal performance... **the model shows near-random performance in classification and regression tasks**... LoRA fine-tuning primarily leverages pre-trained knowledge to generate **well-formed answers based on the output format** learned from SFT data"<br>→ 三阶段:①**生物序列持续预训练** ②大规模指令微调 ③推理微调 |
| **MAPO**<br>arXiv 2605.27741 | GRPO 对所有 token 施加统一梯度 → 加剧 **late-stage modality collapse**,"models **progressively abandon the primary source signal in favor of compressed textual priors**" |
| **Moirain**<br>arXiv 2605.23961 | 条件 RNA 生成为多阶段对齐问题;DPO 的动机是"improve functional fitness **without collapsing the learned natural distribution**" |
| **RNA-GPT**<br>arXiv 2411.08900 | Stage 1 的 Q 留空是**故意**的,为的是堵死"抄 prompt 文本"的捷径 —— 这是它的解,也是它的病 |

### 1.4 一句话定性

> **我们不是在对齐一个对齐得不好的投影。我们是在让一个冻结的、对 RNA 一无所知的语言模型,
> 通过一个 3M 参数的线性层,去做它没有能力做的事。**
>
> ChatMultiOmics 的结论——"LoRA 只是按 SFT 学到的格式产出 well-formed 答案"——
> 精确命中我们的观测:val_loss 降到 1.384(格式学会了),Δzeroed 衰减到 25%(内容没进去)。

**根因是冻结的 LLM。这是软提示路线的结构性上限,换任何 Stage 1 目标都修不了。**

---

## 2. 当前构型（路线 C）

### 2.1 一句话

**RNA → 冻结的 ERNIE-RNA → 4096 个离散码 → 一张可训练的码 embedding 表 → 冻结的 Qwen3-8B。**

没有任何投影层把 768 维硬压到 4096 维。RNA 不再被"翻译"进 LLM 的向量空间，
而是**被离散化成 LLM 词表里的一批条目** —— 让 LLM 把它当普通 token 处理。

### 2.2 五个环节

```
                  冻结                      可训练                冻结 + LoRA
┌──────────┐   ┌─────────────┐   ┌────────┐  ┌──────────┐  ┌─────────────────┐
│ RNA 序列 │──▶│ ERNIE-RNA   │──▶│  FSQ   │─▶│ 码embed  │─▶│ Qwen3-8B        │
│ ≤1024 nt │   │ 768维/位点  │   │4096 码 │  │4096×4096 │  │ 文本问答        │
└──────────┘   └─────────────┘   └────────┘  └──────────┘  └─────────────────┘
                    ①                ②            ③                ④
```

| 环节 | 是什么 | 参数量 | 状态 |
|---|---|---|---|
| ① 编码器 | ERNIE-RNA，768 维，**全程冻结** | 0（冻结） | sha256 + 结构 + 序列敏感性三重验过（Δpoly 3.11 / Δshuffled 2.40 / Δrepeat 0.00） |
| ② 压缩码表 | 线性降到 12 维 → FSQ 2 级量化 → **4096 个离散码**。相当于"RNA 自己的 tokenizer" | ~16.8M | **全量训练中**（4481 条 RNA，2.4M 个位点） |
| ③ 码 → 向量 | `4096×4096` 可训练 embedding 表。RNA 位置**不过 BPE**，直接用 `inputs_embeds` 映射过去 | 16.8M | 冒烟已验通 |
| ④ 对齐 | 底座冻结 + LoRA(attn) + 码表可训练，在 92,680 条 QA 上训 | 7.67M (LoRA) | 短实验跑着 |
| ⑤ 评测 | 逐 token 消融（Δlogp），**不是 loss**；外加 ROUGE 对臂 A | — | — |

### 2.3 为什么是这些选择

**为什么 FSQ 而不是学习式码本。** 学习式码本天然塌缩（StructTokenBench 实测的 codebook
under-utilization），FSQ 从构造上不会，省掉 ProVQ 那整套量化课程。RiboSphere 和 DPLM-2
在 RNA/蛋白上也都避开了学习式码本。

**为什么 4096 而不是 65536。** 44_ 扫出来的：65536 那档只有 4.5 个位置/码、39.7% 的码
只出现一次；4096 是 31.4 个位置/码、中位重用 12 次、90.2% 利用率。正是 RiboSphere
说的"选择性稀疏"那个区间。

**为什么不做去各向异性。** 原推理（91.1% 的模长是跨序列公共分量，直接量化等于把容量
花在量化一个常数方向上）**被实测证伪** —— 对 FSQ 只改 ~0.005。因为 FSQ 前面是一个
**可学习的线性层**，它自己就把这个常数方向吸收掉了。保留为消融项，不是前置条件。

**为什么不在量化之外再叠压缩。** 实测 Qwen 的 BPE 本来就把 1022 nt 压到 522 token
且局部稳定，**长度不是瓶颈**。再叠一层压缩正是该避免的"双重抽象"。

### 2.4 判据链 —— 整套东西的核心

**每一关都是一个能证伪的数，不是 loss。loss 降下来也可能只是模型学会了忽略 RNA 直接编答案。**

| 关 | 判据 | 阈值 | 现有值 |
|---|---|---|---|
| ② | **跨序列 6-mer 迁移倍数** | ≥ **2.0x** | 1500 条上 **2.88x**；全量在跑 |
| ④ | **vs 随机表 / vs 常数码**的负 Δlogp 比例 | < **45%** | 冒烟 48.1% / 47.6%（步数不足，无判别力） |

判据 ② 为什么必须用这个：重建误差和利用率**都能骗人**。重建是循环的（我们就是照它训的）；
利用率在词表远小于样本数时必然好看（哈希也能给出 90%）。唯一非循环的是——
在**没见过的序列**上，同一个码是否指向同一类局部 6-mer。靶子来自输入序列，不是 encoder 学的。

判据 ④ 三个对照缺一不可：

```
Δlogp(t) = logp_real(t) − logp_ablated(t)
  vs 零表   低 => RNA 位置上「有东西」在起作用
  vs 随机表 低 => 起作用的是**码的内容**
  vs 常数码 低 => 起作用的是**码的区分度**
后两个 ≈50% 就是「Qwen 眼里的 RNA 是乱码」，只是换了个形式。
```

**对照臂 A 不交付，它是对照组。** 软提示臂（旧路线）已训完：早停于 15600/17760，
val_loss 1.3839，per-token 负 Δlogp **58.4%**。58.4% 和 48% 就是路线 C 要打的靶子。

### 2.5 与 RNA-GPT 的关系

**没有关系了。** 它那套是"冻结 LLM + 一个 3M 线性层学模态对齐"，正是被证伪掉的那条。
换成现在这样，是因为测出来**冻结的 LLM 在 Stage 1 里根本没获得序列理解** ——
我们是在让一个不懂 RNA 的语言模型，通过一个 3M 的线性层，去做它没有能力做的事。

### 2.6 还没定的

1. **要不要继续预训练** —— `pretrain_rna.jsonl` 14.96 万条已备好未用。实测编码 4.8 条/秒，
   全量要 8.6 小时。
2. **要不要做偏好优化** —— 按 MAPO 的思路，**不能对 RNA token 和文本 token 施加统一梯度**。
   只是设计，未动。
3. ②的迁移若不合格 → **不是加步数，是给码表加复用压力**。46_ 的 2.88x 是在**完全没有复用
   压力**的情况下拿到的（现在的 FSQ 只优化重建，没有任何机制要求码可复用）。那既是"下界"
   的解释，也是最便宜的下一个改进方向。

---

## 3. 实验设计

### 3.1 主对照(头对头)

| 臂 | 路线 | 数据 | 说明 |
|---|---|---|---|
| **A** | 软提示(旧) | 4,481 / 92,680 | 已完成 Stage 1(31 个 checkpoint)+ 待跑 Stage 2。**作为被证伪的基线** |
| **B** | 原生文本 + SFT | 同上 | 第一版新路线 |
| **C** | B + 序列继续预训练 | 同上 + RNAcentral 序列 | 若 B 不够好才做 |

**同一份数据、同一套评测、同一个底座。** 唯一变量是"RNA 怎么进模型"。

### 3.2 判据

- **主**:ROUGE(对生成质量诚实)
- **辅**:语义指标、分类指标
- **诊断**:逐 token 模态相关性(33_ 脚本改编到文本路线)——**看负值占比是否下降**
- **禁止**:只用 val_loss(§1.1 已证伪)

### 3.3 这比纯复现更有价值的地方

RNA-GPT 自己 Table 1 有一列 `RNA Sequence`(纯文本),但它**也是冻结模型、没做序列预训练**,所以那一列差不能证明文本路线不行。ChatMultiOmics 证明了该做序列预训练,**但在 RNA-QA 这个具体任务上没人做过。**

---

## 4. 保留什么

### 4.1 资产

```
data/processed/rnaqa_n6_merged.jsonl    4,481 RNA / 92,680 QA
data/evalset_merged/                    train 3552 / val 441 / test 488
data/external/rnacentral_1024.parquet   32,524,827 条序列（做①时的语料）
data/external/litscan/                  48 个文件，Lit Scan 源头
models/ernierna/                        ERNIE-RNA（保留，作为对照臂 A 的依赖）
checkpoints/stage1_ernierna_c64_full_b2/ 软提示路线的 31 个 checkpoint
```

### 4.2 代码

| 文件 | 在新计划里的角色 |
|---|---|
| `sos_lora.py` | **保留**。Stage 2 的适配器选项,有 9 组单测 + 端到端冒烟 |
| `paths.py` / `encoder.py` | **保留**。臂 A 还需用 |
| `33_token_modality_relevance.py` | **保留并改编**到文本路线 —— 这是核心诊断工具 |
| `32_proj_discriminability.py` | 归档(只对投影有意义) |
| `34/36_*.py` | 归档(证伪记录的一部分) |
| `00d_fetch_verified.py` | **保留**。通用哈希闸门取数 |

---

## 5. 方法论约定(今天用血换来的)

这几条不是流程洁癖,每一条都对应一次真实翻车:

1. **`mem_get_info()` 在 gfx1151(APU)上不可信** —— 它报 107 GB,物理内存只有 63.6 GB。
   **只看物理内存。** 上次据此说"塞得下",实际爆了。
2. **哈希闸门是唯一能挡住「大小对、内容空」的手段** —— 已两次踩到(`models/rnafm` 155/205 张量全零)。
3. **单测过了还要跑真模型冒烟** —— SOS-LoRA 单测 8/8 过,冒烟仍抓出 dtype 不匹配和跨函数作用域两个 bug。
4. **测出来的东西和以为测的东西不是一回事** —— 今天犯四次(置零补丁没生效、梯度比被 B 污染、Δzeroed 消融口径不一致、val_loss 选投影)。
   **每次都在"全零/max 也零/比值太整齐"这类信号上露馅。看到这种信号先怀疑自己。**
5. **优先用可分辨性/逐 token 这类**直接量**,而不是聚合标量** —— val_loss 已被证伪两次。
6. **换源不算办法** —— 先修校验,再等。

---

## 6. 路线 C：量化离散 RNA token（**当前主线**）

### 6.1 为什么否掉路线 B

用户的判断，我认同并补了证据：

> **Qwen 眼里的 RNA 就是乱码。**

我们实测：Qwen3 的 BPE 把 RNA 切成 `GA|UC|GA|UG|A|UG|AAAA|UA` —— 这些合并是**英文的统计**产出的，
边界与 RNA 结构无关；而它们的 embedding 是在**英文语义**下训出来的。
**所以初始状态下，Qwen 拿到的确实是一串无意义的符号。**

这和软提示路线的病**是同一个**：
- 软提示：向量在分布外，模型没有槽位接 → 实测 58.4% 的 token 上被忽略
- 原生文本：token 在词表内，但语义是英文的 → 模型没有 RNA 概念

**共同点：Qwen 没有为「RNA」准备过任何东西。**
而 LLM 真正会处理的是**「离散 token + 学出来的 embedding」**。

### 6.2 所以把编码器输出量化成离散 unit

```
RNA 序列
  → ERNIE-RNA（冻结，它懂 RNA）              ← 保留编码器的结构知识
  → 减去公共方向（ernierna_aniso_mean.pt）    ← 前置必需，见 6.4
  → FSQ 无码本量化 → 离散 unit 序列
  → 每个 unit 有**可训练 embedding**
  → 进 Qwen，当普通 token
```

### 6.3 文献依据（都已读原文）

| 论文 | 给了什么 |
|---|---|
| **RiboSphere** `2603.19636` | **RNA + FSQ 已验证**。公式 `ĉ = ⌊l/2⌋·tanh(Linear(c))`，straight-through 传梯度。**关键发现**：码本 240 → 利用率 100% 但丢局部几何；码本 4,375 → 利用率自发掉到 ~40%，**重建反而更好**。稀疏但**有选择性**的使用 "prevents posterior collapse"，不是失败 |
| **ProVQ** `2603.22304` | 命名了我们会踩的坑：**Premature Discretization** —— "enforce discretization before the encoder has captured the underlying data manifold"。修法是**把量化当课程**：manifold warmup 50k 步（全连续）→ 20k 步 cosine 退火到离散。**已在生物序列上验证** |
| **DPLM-2** `2410.13782` | 蛋白 3D 坐标 → **离散 token（LFQ）** → 语言模型按普通 token 训练。**模式在生物模态上成立** |
| **StructTokenBench** `2503.00089` | 结构 tokenizer 的评测框架。**"no single model dominates"**；学习式码本的 **codebook under-utilization** 是主要陷阱 |

**注意 RiboSphere 与 DPLM-2 都避开了学习式码本（FSQ / LFQ）。** 我们跟这个选择。

### 6.4 三条设计结论

1. **用 FSQ，不用学习式码本。** 后者天然会塌缩（StructTokenBench 的实测），FSQ 从构造上不会。省掉 ProVQ 那套课程。
2. ~~**去各向异性从"可选优化"升为"前置必需"。**~~ **【已被自己的测量证伪，见 6.6】**
   原推理：ERNIE-RNA 输出模长的 **91.1%** 是跨序列公共分量，直接量化等于把容量花在量化常数方向上。
   **实测：对 FSQ 做去各向异性，重建 err 只变 ~0.005。**
   原因很简单 —— FSQ 前面是一个**可学习的线性层**，它自己就能把这个常数方向吸收掉。
   去各向异性保留为消融项，**不是前置条件**。
3. **不要为了压缩而先池化。** 已证明 Qwen 的 BPE 能把 1022 nt 压到 522 token 且局部稳定，
   **长度不是瓶颈**。在量化之外再叠一个压缩，正是 DeCo 警告的"双重抽象"。

### 6.5 「码是词表还是哈希」—— 三个实测（44/45/46）

这条路线的成败,取决于一个**不用训 LLM 就能回答**的问题:离散码是不是**可复用**的。
不是的话,接进 LLM 只是给每个位置发了张"哪条序列"的指纹,embedding 学不到可迁移的东西。

**过程中我自己写错了两个指标,两个都必须留在记录里。**

| # | 指标 | 结果 | 为什么它不能作数 |
|---|---|---|---|
| 44 v1 | 码的重用次数 | 「52% 的码只出现一次」 | **算术必然,不是发现。** 64 条序列 29k 位置 vs 词表 65536 → 平均每码 2.6 个位置,中位重用必然是 1 |
| 44 v2 | 留出覆盖率 + iid 自助基线 | 所有配置**略低于**基线(4096: 95.8% vs 97.5%) | **指标本身没有判别力。** 词表 4096、训练位 9.3 万时,**哈希也会给出 90% 利用率和 95% 覆盖**;词表远大于样本时又样本不够。两头都不成立 |
| 45 | 序列指纹对照 (a) 同序列内留出 vs (b) 跨序列留出 | (a) -2.0pp,(b) -2.7pp,**两者一样**;burstiness 平均游程 **1.02**(iid=1.0) | 指纹解释**被推翻**。但也再次说明覆盖率测不出东西 |
| **46** | **跨序列的「码 → 局部 6-mer」迁移** | **8.48% vs 打乱对照 2.95% vs 常数 3.02% → 2.88x** | **这一个才是非循环的** |

`within`/重建 err 也**不作数**:FSQ 是确定性映射、AE 又照重构误差训,`within<1` 是构造保证的。
**只有 46 的靶子来自输入序列、不是 encoder 学的,所以不循环。**

```
训练位置 92348   测试位置 21746   (局部 6-mer, 词表 4096)
测试位置中码在训练集见过的比例 : 99.1%
按码预测 6-mer 准确率          :  8.48%
打乱 code<->位置 对照          :  2.95%   <- 回落到常数水平,对照有效
常数预测(最常见 6-mer)        :  3.02%
增益 / 倍数                    : +5.53pp / 2.88x
```

**结论:是词表,但是一个不精确的词表。** 码携带能跨序列迁移的局部结构信息,**不是序列指纹**。路线 C 没死。
两个限制要写清楚:6-mer 是**很弱**的靶子(ERNIE-RNA 在一位点上编码的远不止 6-mer),
所以 **2.88x 是下界不是上限**;反过来它也说明码与局部组成的相关性并不强。

**设计修正(来自上面的测量):**
- FSQ 量化**不做**去各向异性(证伪,见 6.4 第 2 条)
- 码本尺寸:**放弃 65536**(位置/码 只有 4.5,39.7% 的码只出现一次);**取 4096**(位置/码 31.4,
  仅 9.4% 单次,中位重用 12 次,90.2% 利用率)。这正是 RiboSphere 说的"选择性稀疏"那个区间
- **必须加复用压力。** 现在的 AE 只优化重建,没有任何机制要求码可复用 —— 46 的 2.88x 是在
  **没有**复用压力的情况下拿到的。这既是下界的解释,也是最便宜的改进方向

### 6.6 一个踩过的坑（会影响复现）

**ERNIE-RNA 的 tokenizer 每条序列加了 `[CLS]` 和 `[SEP]`** —— encoder 位置数 = 序列长度 **+ 2**。
校验方式:`115886 - 115374 = 512 = 2 × 256`,一眼可验。
不做这个偏移,位置靶子整体错位一格,**结果全是废的**。第一次跑就栽在这里。


### 6.7 尚未有先例的地方（= 贡献点，也 = 没有超参可抄）

搜到的都是：
- RiboSphere：量化 RNA **3D 结构** → 结构生成
- DPLM-2：量化蛋白 **3D 坐标** → 蛋白语言模型

**没人做「量化 RNA 序列编码器的输出 → 喂给通用 LLM 做问答」。**
所以 FSQ 的 `l`（量化级数）和 `d`（维度）**必须在自己的验证集上扫**。

---

## 7. 下一步（短实验 = 现在就跑的那个）

**唯一要回答的问题:把 RNA 位置换成可训练的离散码 embedding 之后,模型到底会不会用它。**

判据不是 loss,是 **`33_token_modality_relevance.py` 量出来的 negative-Δlogp 比例**:
软提示臂 A 的基线是 **58.4%**。**只有这个数字掉下来,才叫模态对齐。**

```
[x] 0. 路线 C 的可行性前置:码是不是可复用词表
       44/45/46 已跑 — 是词表,2.88x(下界)。码本定 4096,不做去各向异性          见 6.5
[x] 1. 软提示臂 A 基线(已完成):Stage 1 早停于 15600/17760,val_loss 1.3839
       已测 negative-Δlogp 比例 58.4%  <-- 短实验要打的就是这个数

[~] 2. 短实验 — 单脚本端到端,不碰已有 checkpoint
       a. [x] 全量 FSQ 分词器训练完 —— 4481 条 / 4096 码 / 迁移 **2.63x** 通过
          checkpoints/fsq_full/{tokenizer.pt, codes.pt}
       b. Qwen3-8B: add_tokens(码) + resize_token_embeddings,码 embedding 可训练
       c. 短 SFT ~500 步(其余参数冻结,或只挂 LoRA),数据 rnaqa_n6_merged.jsonl
       d. 判据:33_ 的 negative-Δlogp 比例 vs 58.4%
       e. 若掉下来 -> 上全量;若不动 -> 复用压力(commitment / 熵正则)再试一次

[ ] 3. 与臂 A 头对头:同数据、同步数、同评测(ROUGE + 33_ 的 per-token profile)
[ ] 4. 若需要 -> 补序列继续预训练(语料已备:pretrain_rna.jsonl,14.96 万条)
[ ] 5. 若做偏好优化 -> 按 MAPO 的方式,不要对 RNA token 和文本 token 施加统一梯度
```

**纪律:短实验只改一个变量(RNA 的表示方式)。** 别的都不动。

> **2026-10-11 更新:2.e 的两条分支都不适用了 —— 见 §9。**
> 「复用压力」这个方向的前提（码没塌）被自己的测量推翻:全量 3885/4096 在用、
> 跨序列迁移反而从 2.63x 涨到 **7.79x**。瓶颈不在 tokenizer，在 LLM 侧。

---

## 8. 数据与实验结果（截至 2026-10-11）

### 8.1 数据文件

| 文件 | 条数 | RNA 数 | 模板 | 划分 |
|---|---|---|---|---|
| `data/processed/rnaqa_seq30_split.jsonl` | 131,456 | 4,481 | 30 | train 105,144 / val 13,167 / test 13,145 |
| `data/processed/rnaqa_seq30_full.jsonl` | 4,341,380 | 145,201 | 30 | 无 |
| `data/processed/rnaqa_rfam.jsonl` | 39,452 | 19,726 | 2 | train 15,780 / test 3,946 |
| `data/processed/rnaqa_freetext.jsonl` | 13,443 | 4,481 | 3 | 无 |
| `data/external/rfam/hits32.tbl` | 442,244 命中 | 128,155 有标注 | 3,887 家族 | — |

数据生成的筛选判据（全部零成本、不用 LLM，除 freetext）：

```
序列模板 30 个     瞎猜命中率 0.0% - 17.9%（逐个过了 54_/55_ 的筛选器）
Rfam 家族 2 个     平衡采样前瞎猜 43.4% -> 每家族上限 40 条 -> 0.203%
自由文本 3 个      唯一率 0.67 - 0.96，弃答率 0.0% - 33.1%
                   提示词写死「只能用 annotation 里的事实」，弃答时输出固定句
                   抽查逐字对照 annotation，未发现编造
```

自由文本只覆盖 4,481 条带 annotation 的 RNA。其余 145,201 条只有 `seq` 和 `rna_type`，无任何描述文本，未做自由文本生成。

### 8.2 隐藏态缓存

| 缓存 | 覆盖 | 大小 | 状态 |
|---|---|---|---|
| `data/processed/_fsq_full_pos.pt` | 4,481 条 RNA | 5.9 GB | 存在 |
| `data/processed/_fsq_all_hidden.f16` | 149,598 条 RNA | 70.9 GB | **已删除**（腾磁盘） |

### 8.3 训练实际用了什么

```
文件      data/processed/rnaqa_seq30_split.jsonl
数量      训练 105,144 / 评测 26,312
覆盖      4,481 条 RNA（占预训练语料 145,201 条的 3.1%）
RNA 来源  data/processed/rnaqa_n6_merged.jsonl（4,481 条带注释 RNA）

未使用    rnaqa_seq30_full.jsonl   4,341,380 条
          rnaqa_rfam.jsonl            39,452 条
          rnaqa_freetext.jsonl        13,443 条
原因      前两者的 uid 不在 _fsq_full_pos.pt 覆盖范围内（缓存只建了 4,481 条）
          freetext 的 uid 在覆盖范围内，但未加入训练命令
```

### 8.4 阶段 0：分词器

```
4096 码（latent 12, levels 2, pool=4 在模型内部做）
  重建 mse 0.0444   归一化 err 0.4934
  用码 4092/4096 (99.9%)   位置/用码 493.4   中位重用 264   仅 1 次 0.3%
  跨序列 6-mer 迁移  8.62% vs 打乱 3.27% vs 常数 3.40%  ->  2.63x
  两次独立运行结果完全一致

256 码（latent 8, levels 2）
  重建 err 0.5777   用码 256/256   中位重用 5341
  跨序列 6-mer 迁移  ->  1.95x     （低于 2.0x 门槛）
```

结论：256 码迁移不达标，阶段 1 改用 4096 码。

### 8.5 阶段 1：端到端（route C）

**A_v2 — 12000 步（0.46 epoch），训练 prompt 无空 think 块**

```
ce 起点 11.34
[diag@12000] Δ换RNA 0.0679   Δ换问题 1.9236   Δ零表 0.1034    RNA/问题 = 0.035
             码本 371/4096 在用   熵 5.92/8.32   最大码占比 0.3%   有效秩 232.2
[最终消融]   vszero 33.0%  vsrandom 36.2%  vsconst 44.0%  vsperm 28.9%
[准确率 n=60] acc(A) 33.3%   acc(B) 28.3%   acc(C) 35.0%   **⚠️ 同样作废，见 §11**
```

**A_v3 — 从 step 12000 续训，加入空 think 块**

```
[diag@15000] Δ换RNA 0.0708   Δ换问题 1.9198   Δ零表 0.1407    RNA/问题 = 0.037
             码本 350/4096   熵 5.86/8.32   最大码占比 0.3%   有效秩 234.0
             ce 0.32 - 0.82    ent 5.4 - 6.0    |grad_fsq| 0.03 - 0.20
             码表漂移 104.6%（自内容初始化起）
[eval@15000] vszero 37.4%  vsrandom 41.6%  vsconst 24.2%  vsperm 39.5%   |Δ| 0.1321
[准确率 n=90 @step14000]  **⚠️ 已作废，见 §11**
             acc(A) 16.7%  (15/90)
             acc(B)  8.9%  ( 8/90)
             acc(C) 25.6%  (23/90)
             A vs C  z = 1.46  p = 0.14   不显著
             A vs B  z = 1.56  p = 0.12   不显著
             两轮合并（n=150）A 23.3% vs C 29.3%  z = 1.18   仍不显著

             **这组数字不能用。** `64_acc_nothink.py` 的 `gen()` 把金标准答案
             一起喂进了 prompt，实测问的是「答案已经在上下文里了，接着写」。
             详见 §11。§8.3「训练实际用了什么」那节的数据无误，只有准确率这组作废。
```

A_v3 运行中，目标 27000 步（1.03 epoch）。

生成的答案形态正确、数值错误：

```
Q What are the first 6 nucleotides of this RNA?
  预测 GGGGAG     答案 CUGGCC
Q How many G nucleotides are in this RNA?
  预测 273        答案 343
```

### 8.6 评测中修掉的两个 bug

这两个都会直接导致结论错误：

```
1. PeftModel.from_pretrained 套在已有的 PeftModel 外层
   -> adapter 权重键全部不匹配、被当 UNEXPECTED 丢弃
   -> 评测实际是「未微调的 Qwen + 训练好的码表」
   同一检查点：修前 acc 0%  ->  修后 acc 33.3%
   修法：用 set_peft_model_state_dict 往已有 PeftModel 里灌

2. 训练 prompt 没有空 think 块，评测 prompt 有
   Qwen3 是混合思考模型，默认先思考再答
   max_new_tokens 被思考文本吃光
   实测输出 'THINK THE FIRST OF THE RNASEQUENCE...' / 'OKAY LETSSEE...'
   修法：build_batch 在 assistant 头后插空 think 块（13 个 token）
        修后输出 'CUGGCC' / '273'（形态正确）
```

### 8.7 原论文的对应数字

论文训练配置：Stage 1（MA）10 epochs，Stage 2（IT）10 epochs，batch size 1，
2 × A100 40GB，两阶段合计 1 周。Stage 1 明确「freeze the sequence encoder block
and train the linear projection layer」；Stage 2 是否冻结 LLM 论文未说明。

| 指标 | RNA Sequence（不做对齐） | Modality Fusion（只跑 Stage 1） | RNA-GPT（Stage 1+2） |
|---|---|---|---|
| Rouge-1（AIS） | 0.2364 | **0.2239** | 0.5031 |
| Rouge-2（AIS） | 0.0935 | 0.1364 | 0.3667 |
| Rouge-L（AIS） | 0.2037 | 0.2091 | 0.4747 |
| F1 @ S_BERT（D&C） | 0.7625 | 0.7466 | 0.8609 |
| F1 @ S_Pub（D&C） | 0.5501 | 0.6005 | 0.7165 |
| F1 @ S_GPT（D&C） | 0.5561 | 0.6637 | 0.7560 |

论文的消融只比较了上述三列，**没有「给 RNA-GPT 但不给 RNA」这一臂**。

### 8.8 硬件与耗时记录

```
ERNIE-RNA 编码   149,598 条 / 4,950 万位置   8.2 条/秒   5.08 小时   70.88 GB
Rfam cmscan      145,201 条 / 32 片并行      约 1.5 小时  442,244 命中
阶段 0 训练      4096 码 / 6000 步           54 秒
阶段 1 训练      A_v2 12000 步               448 分钟（1.4 秒/步，batch 4）
自由文本生成     22,405 次 API 调用          68 次/秒（并发 48）
```

机器：AMD Radeon 8060S（gfx1151，APU，无独立显存），物理 RAM 63.6 GB。
物理 RAM 是唯一可信的内存读数——`mem_get_info()` 在 APU 上报告 107 GB 但实际只有 63.6 GB。

---

## 9. 2026-10-11 审计:e2e 之后的码表还是不是词表（scripts/66_code_channel_audit.py）

**判据②只在 stage-0 的冻结 tokenizer 上量过**（2.63x，逐核苷酸一个码）。
实际接进 LLM 的是**另一版** tokenizer：前面多了一个 pool=4 的学习式压缩，
`enc` 从 stage-0 载入后**继续被 CE 梯度训了 16000 步**（日志 `|grad_fsq|` 0.03~0.7）。
**这一版从来没有测过判据②。** 而它是「码的身份值不值得学」的唯一非循环证据。

三路对照（同一批隐藏态、同一套 6-mer 靶子、同一个跨序列划分 hold=0.2/seed=7）：

| 版本 | 按码预测 6-mer | 打乱对照 | 常数 | 倍数 | 全量用码 | 硬熵 |
|---|---|---|---|---|---|---|
| A 冻结 · 逐核苷酸 | 8.62% | 3.27% | 3.40% | **2.63x** | 4092/4096 | 7.72/8.32 |
| B 冻结 enc + 平均池化 4x | 5.83% | 3.17% | 3.40% | **1.84x** | 3782/4096 | 6.78/8.32 |
| C **e2e 训练后** · 池化 4x | 23.57% | 3.03% | 3.40% | **7.79x** | 3885/4096 | 7.44/8.32 |

A 行精确复现 48_ 的 2.63x —— **尺子先自检过，C 行的读数才作数**。

### 12.1 三个假设被自己的测量否掉

**① 「码本塌缩到 350/4096」不成立 —— 又一次采样量幻觉。**
那个读数来自训练日志的 `[diag] 码本: 用到 350/4096`，它只覆盖 **40 条序列 / 约 4.5k 个窗口**。
全量 4481 条 / **50.3 万窗口**上是 **3885/4096 在用、硬熵 7.44/8.32、最大码占比 0.43%**。
没有塌缩。44_ v1 的教训（「52% 的码只出现一次」是算术必然、不是发现）**今天原样又犯了一次**，
而且是犯在**训练日志里已经写进结论**的那一行上。

**② e2e 训练没有破坏 tokenizer，反而把它变强了（2.63x → 7.79x）。**
原假设「CE 梯度把码改成了每条序列各自不同的片段」被否掉。附带读数：

```
码 ID 与冻结版一致率（窗口首位）  0.1%      <- 键全换了，内容映射还在
enc 相对漂移                      31.6%
compress 距平均池化               36.5   （平均池化自身范数 13.9，即已远离 3 倍）
```

**留一个口子（必须写清楚）**：7.79x 只在**窗口首位**那个 6-mer 上量过。
若 `compress` 学到的是「只编码窗口首位」，这个数会虚高，而窗口内另外三位的
信息可能已经丢了。要坐实，得补「窗口内四位各自的可预测性」对照。**未做。**

**③ 熵正则是惰性的 —— `--ent-coef 1.0` 这 16000 步几乎什么都没干。**
软使用率饱和在 **8.31/8.32**（上限 8.32），梯度 ≈ 0。原因是构造性的：
真码在 `tau=6`、`d2∈[0,48]` 的 softmax 里只分到约 6% 权重，其余摊给 12 个邻居，
平均下来必然接近均匀。所以**「复用压力」从来没真正施加过**——
这既解释了 §6.5 里「2.88x 是在完全没有复用压力下拿到的」，也说明不能拿
`--ent-coef 1.0` 当作「加过压力」的证据。它不需要干（码没塌），但方向因此失去依据。

### 12.2 对 §7.2.e 的影响

```
原判据：判据④若不动 -> 加复用压力（commitment / 熵正则）再试一次
现证据：码没塌（3885/4096）、迁移反而变强（7.79x）、熵正则已饱和
=> 复用压力不是瓶颈。
```

**结论：瓶颈在 LLM 侧 —— 码里有东西，模型不读。** 下一步不再是重训 tokenizer，
而是量清楚 LLM 侧到底怎么失败的（见 §10）。

---

## 10. 2026-10-11 码信道的天花板（scripts/69_code_answer_probe.py）

§9 只回答「码还是不是词表」。这一节回答「码里有没有这批问答的答案」——
它决定下一步是动 LLM 侧还是动 tokenizer。
方法：**不需要 LLM**，在训练序列上拟合「码直方图 → 答案」，在没见过的序列上考
（划分与 LLM 评测同一份 `rnaqa_seq30_split.jsonl`）。

### 10.1 先补掉 §9 留的口子

```
窗口内第 0 位 92.1%   第 1 位 92.9%   第 2 位 92.4%   第 3 位 89.8%   （随机 25.0%）
```

**四位都还在。** 池化没有把窗口内容压掉，7.79x 不是「compress 只编码窗口首位」的虚高。口子关上。

### 10.2 探针（只看码）

| 模板 | 探针准确率 |
|---|---|
| dinucleotide at positions 1 and 2 | **81.0%** |
| first 4 nt（含它的反向互补） | **67.4%** |
| first 6 nt / positions 1-6 / 反向互补 first 6 | **55.3%** |
| What type of RNA is this? | **62.8%**（众数地板 34.7%） |
| first 8 nt | 44.6% |
| last 4 / last 6 / last 8 nt | 14.9% / 11.3% / 8.9% |
| **10 个计数类模板** | **0.1% ~ 3.7%** |
| **6 个百分比类模板** | **0.0%** |
| G in first 50 | 3.2% |

总体 18.3%（4807/26312）。旧口径的 LLM acc(A) 16.7%、众数地板 3.3%（见 §11，旧口径已作废）。

### 10.3 计数天花板：不训练任何东西，看码里有多少计数信息

精确匹配被 0.1% 的精度要求卡住，分不清「码里没有」和「线性探针做不到」。
所以另做**无状态最优估计**：每个码在训练窗口里的平均碱基个数，逐窗口求和。

| 目标 | MAE | 中位绝对误差 | \|误差\|<0.5 的比例 |
|---|---|---|---|
| A 计数 | 8.41 个 | 4.79 | 10.5% |
| C 计数 | 8.09 个 | 4.50 | 11.4% |
| G 计数 | 7.90 个 | 4.06 | 11.2% |
| U 计数 | 8.22 个 | 4.80 | 8.7% |
| GC 含量 | 4.21 个百分点 | — | 5.8% |

（众数地板在这些模板上是 1.6% ~ 2.6%）

### 10.4 判读：码信道**局部强、全局弱**

- **局部强**：前几位序列、位置 1-2 的二核苷酸读得出来（55% ~ 81%，远离概率水平）。
  这是 LLM **至少**该达到的水平。
- **全局弱**：精确计数读不出来 —— 最好的无状态估计也只有 ~11% 精确、MAE ≈ 8 个碱基。
  而 30 个模板里 **16 个是计数/百分比类**。

两条限制必须一起写：
1. 「无状态」是**下界**。码是上下文化的，序列感知的估计器（含 LLM）可以更好；
   MAE 8 个碱基 **不等于**「不可能」。
2. 探针的拟合只用**训练序列**，测试序列全部没见过；划分与 LLM 评测一致。

---

## 11. 2026-10-11 仪器翻车：评测 prompt 里带着金标准答案（scripts/_probe_eval_prompt.py）

**`64_acc_nothink.py` / `67_acc_dump.py` 的准确率全部作废 —— PLAN §8.5 那三个数
（acc A 16.7% / B 8.9% / C 25.6%）也是。**

`gen()` 把 `build()` 返回的 ids 整个喂给 `generate(inputs_embeds=...)`，而那个 ids 是

```
head + tk.enc(f" A: {r['answer']}{tk.eos}")
                 ^^^^^^^^^^ 金标准答案
```

`lab` 才是用来算 loss 的。而 transformers 在**只给 `inputs_embeds`** 时把 `input_ids`
初始化成 `(B, 0)`（`generation/utils.py:694`，读源码核实），**返回序列只含新生成的 token**。
于是这个评测实际问的是：

> 「金标准答案和 `<|im_end|>` 都已经在上下文里了，接着写。」

实测逐字（`reports/probe_eval_prompt.log`，同一个样本）：

```
leak  prompt 结尾 : '... A: CUGGCC'
      续写(原样)  : '<|im_start|>assistant\n A: GGGGAG<|im_end|>'    <- 另起一轮，再答一遍
clean prompt 结尾 : '...<｜end▁of▁thinking｜>\n\n'（只到 assistant 头 + 空 think 块）
      续写(原样)  : ' A: GGGGAG<|im_end|>'
```

三件事：

1. 旧口径**不是**「第一次作答的准确率」，而是「第二轮作答」；
2. 金答案就摆在眼前，模型**也没有抄**（leak 与 clean 给出同一个 'GGGGAG'）——
   这本身说明它连"复制上下文里的数字"都没做；
3. 只要有样本会去抄，旧口径就被**系统性抬高**，方向不明。**所以旧数字一个都不能用。**

**已修的两处（`67_acc_dump.py`）**：
- ids 只到 `assistant 头 + 空 think 块`，答案段不进 prompt；
- split 有三个值（train / val / test），旧写法「非 train 即 test」把 **val 的 13,167 条
  混进了测试池**（64_ 只用 test，两边取到的样本本就不同）。

**补进 §5 方法论的两条**：

7. **`generate(inputs_embeds=...)` 的返回值只含新 token。** 这个语义与「把 prompt ids
   和它一起传进去」挨在同一行，是最容易读错的组合。凡是新写的评测口径，
   先拿**一个样本把 prompt 与输出逐字打印出来**，再跑全量。
8. **报告里凡是"当前模型准确率"的数，都要标注口径版本。** 今天同一批权重、
   同一批数据、同一段代码家族，三个口径给出 33.3% / 16.7% / 6.7% 三种结果。

---

## 12. 底座模型决策与量化可行性（2026-10-11）

### 12.1 决策

**改用 `Qwen/Qwen3.6-35B-A3B`（MoE，35B 总参 / 3B 活跃）作为底座，4-bit QLoRA 训练。**

理由：
- MoE 只有 3B 活跃，forward/backward 算力接近 3B 稠密模型，**比现在的 Qwen3-8B 还快**
- 同样 31M 可训参数，搭在一个大 4 倍的底座上（0.4% -> 0.09%）
- 这正是「PEFT 才是核心」这个问题该问的：可训参数放在哪、配多大的底座

### 12.2 量化可行性 —— 已验证

```
bitsandbytes 0.50.2 在 gfx1151 上可用
  Linear4bit 前向        OK
  Linear4bit 反向        OK      x.grad 形状正确
  PagedAdamW8bit         OK
  4bit 底座 + LoRA 前反向 OK     可训参数 1024（测试小模型）
```

启动时会打印 `Could not detect ROCm GPU architecture` 和 `ROCm GPU architecture detection failed`，
**是虚警** —— 它退回默认架构，但算子确实跑通了。

RAM 换算：
```
bf16    约 70 GB   ->  装不下（物理 63.6 GB）
8-bit   约 35 GB   ->  能装
4-bit   约 18 GB   ->  宽裕
```

### 12.3 本地模型清单（`C:\qwen-api\models\`）

| 目录 | 大小 | 格式 | 是否能训 |
|---|---|---|---|
| `JoyFox-35B-safetensors` | 67.0 GB | HF `qwen3_5_moe` | 是，但 bf16 装不下 |
| `JoyFox-Qwen3.8-27B-Aggressive` | 27.3 GB | HF `qwen3_5` | 是（8-bit 量化版） |
| `DavidAU-...27B-...MAX` | 23.2 GB | GGUF | **否** |
| `JoyFox-Qwen3.6-35B-A3B-RP-Aggressive` | 23.6 GB | GGUF | **否** |
| `Qwen3.6-35B-A3B-msc` | 20.6 GB | GGUF | **否** |
| `Qwen3.6-40B-Deckard-MTP` | 23.7 GB | GGUF | **否** |
| `Qwen3.5-9B-NSFW-RP` | 6.9 GB | GGUF | **否** |

七个里只有两个是 HF 格式。其余是 GGUF（llama.cpp 格式，transformers/peft 直接训不了）。

**注意：除正在下载的官方版外，本地的都是角色扮演 / 越狱微调版**（RP-Aggressive / Uncensored / NSFW）。
对 RNA 精确值问答是负资产 —— 输出先验偏向叙事文本。已实测过一次同类问题：
Qwen3-8B 因思考先验输出 `THINK THE FIRST OF...` 导致准确率归零。

### 12.4 下载状态 —— 卡住

```
仓库      Qwen/Qwen3.6-35B-A3B  （ModelScope，608,702 次下载）
文件      40 个 / 67.0 GB     26 个 safetensors 分片 + config/tokenizer
命令      python scripts/00b_download_modelscope.py --repo Qwen/Qwen3.6-35B-A3B \
            --dest C:\qwen-api\models\Qwen3.6-35B-A3B --workers 8
已完成    前 8 个配置/tokenizer 文件（小文件，秒下）
卡在哪    26 个分片的 .part 文件全部 0.0 MB，60 秒采样增长 0.0 MB —— **不是慢，是不动**
日志      reports/dl_35b.log（注：日志里的「累计 xGB/67GB」计数器是假的，
          每个文件固定加 1.7 GB，和真实大小无关，看目标目录文件实际大小才准）
磁盘      C: 可用 171 GB
```

**下一步**：改用 `snapshot_download` 或 `ms.exe download`，或查 `00b_download_modelscope.py` 的并行逻辑。

### 12.5 下载完成后必做

```
1. sha256 验证（scripts/00c_verify_models.py）
   26 个分片，任何一个坏掉都会在加载时报奇怪的错。
   今天已经在两个文件上栽过「大小对、内容坏」：
     id_mapping.tsv.gz      2.5 GB 大小正确，20M 行后 zlib 报 invalid block type
     models/rnafm/model.safetensors  398,117,588 B 正确，155/205 个张量全零
2. 试 AutoConfig / AutoModel 能否加载 qwen3_5_moe
   transformers 5.9.0 未必内置支持新架构 —— 这是最后一个未知数
3. 改 scripts/50_routeC_e2e.py 支持 load_in_4bit
   peft 的 prepare_model_for_kbit_training + LoraConfig
```

### 12.6 环境坑（新对话必须知道）

```
peft 的导入顺序
  直接 `import peft` 会炸：
    peft -> transformers.generation -> torch.distributed -> torch._C._distributed_c10d
    ModuleNotFoundError: No module named 'torch._C._distributed_c10d'
  **必须先 `import rnaqwen`**（或任何先建立正确导入顺序的东西），训练脚本就是这么干的。
  同理，裸跑 `python -c "from transformers import AutoTokenizer"` 也会炸。

AMD / ROCm 上的库支持经常是「文档说支持，实际报错」
  torch.distributed.is_available() == False
  MIOpen 在 gfx1151 上不支持 Conv1d（报 miopenStatusInternalError）-> 已改用 unfold+Linear
  bitsandbytes 可用（已验证）

内存
  mem_get_info() 在 APU 上报告 107 GB 但物理只有 63.6 GB
  **只有 Win32_OperatingSystem 的物理内存读数可信**
```

### 12.7 四个待判假设的现状

| 假设 | 证据 | 状态 |
|---|---|---|
| 1 数据不足 | 训练只用了 4,481 条 RNA = 可用语料的 3.1%；rfam/freetext/seq30_full 均未加入 | 强 |
| 2 还没收敛 | 0.72 epoch vs 论文 20 epoch；RNA/问题 0.035 -> 0.037 -> 0.081，**还在涨** | 强 |
| 3 结构性做不到 | `\|grad_fsq\|` 0.03-0.2 而码表漂移 114% —— 编码器几乎不动，码表动很多；鸡生蛋 | 中 |
| 4 模型能力太差 | 只有 0.4% 参数可训，瓶颈不可能是底座不够大 | 弱 |
| 5 脚本有 bug | 评测侧今天抓到两个（PeftModel 嵌套、think 块不一致）；训练侧接线有硬断言通过 | 未排除 |

**假设 2 和 3 共用同一个实验**：把 27000 步跑完，看 `RNA/问题` 是继续涨还是走平。

---

## 13. 数据情况与实验结果（2026-10-11 夜）

### 13.1 数据文件与覆盖

```
data/external/rfam/
  all.fa          145,201 条   47,547,438 位置   最长 1022  最短 20  均值 327
                  uid = md5(sequence)[:16] —— 序列相同必然合并，无重复序列
  hits32.tbl      442,244 hits / 128,155 唯一序列（cmscan 输出）

data/processed/（隐藏态缓存）
  _fsq_full_pos.pt       4,481 条   2,018,834 位置   5.8 GiB   （seq8 / freetext 用）
  _rfam_hidden.f16      19,726 条   6,680,378 位置   9.6 GiB   （rfam 用）
  _rfa_full_hidden.f16  预分配 47,547,438 x 768 fp16 = 68.02 GiB （升级 2，编码中）

磁盘：编码前空闲 104 GiB -> 预分配后 36 GiB
```

`19,726` 不是随机抽样。`61_rfam_qa.py` 按家族平衡采样（每家族 <= 40），
因为原始语料 rRNA 占 51%、RF00177 一个家族就 43.4%，"永远答最常见家族"能拿 43.4%。
**但新数据集的答案全部从序列复算，标签不参与 —— 家族平衡这个约束自动失效。**

### 13.2 三处仪器修正（都会改数字，必须先记下来）

**(a) 码信道在尾部是缺的（新发现，推翻 §10 的一条读法）**

```python
# scripts/50_routeC_e2e.py:125-126  FSQCode.forward
n = L // self.pool
x = x[: n * self.pool]        # 尾部截断
```

24,207 条 RNA 里 **18,313 条（75.6%）位置数不是 4 的倍数**：

```
位置数 %4 = 0: 5,894   =1: 5,644   =2: 6,923   =3: 5,746
```

于是 §10 那三个数

```
last-4  0.149      last-6  0.113      last-8  0.089
```

**被污染了**：答案是按**原始**序列算的，模型看的是**截断后**的。缺 1 个碱基就要求模型猜中 1/4，
缺 3 个是 1/64。哪怕信道完美，last-4 的天花板也被压到 0.25 附近。
所以它们**不能**单独用作"码信道读不到尾部"的证据。

修法：`effective(s) = s[:len(s)//4*4]`，所有答案按截断后的序列算。

**(b) rfam 的划分是纯随机的，`61_` 的注释和代码不一致**

```python
# 这里的泄漏风险是"同一序列"或"近亲"…用 57_ 的聚类结果过滤。
rng2 = random.Random(SEED + 1)
order = list(picked); rng2.shuffle(order)     # <- 注释说要用聚类，代码没用
```

实测 20-mer 重叠（`57_` 的连边阈值 0.5）：

```
rfam 现有划分   中位 0.000  均值 0.103  >=0.5: 6.67%  >=0.9: 0.87%
seq8 对照       中位 0.000  均值 0.061  >=0.5: 4.71%  >=0.9: 4.04%
```

**两批都有近亲，seq8 也没干净**，形状还不同（seq8 是 4% 紧密近乎同一对，rfam 是 6.7% 弥散同源）。

**(c) 首/末 4~6 nt 的"100% 重叠"是组合数学，不是泄漏**

```
首 4 nt 在 train 出现过  3946/3946 = 100.0%     ← 4^4 = 256 种，必然全覆盖
首 6 nt                  95.5%                  ← 4^6 = 4096 种，均匀期望 97.9%
首 8 nt                  39.5%
```

**同时决定：不做簇划分。** 4^8 = 65,536 种 8-mer 对 145,201 条序列，
鸽笼原理下不存在"首 8 未共享"的划分；而 route C 不是泛化任务（答案按定义在输入里），
经典泄漏不适用。替代仪器是 §13.5 的指纹子集。

### 13.3 C_merged 的按来源结构（合并决策的证据）

```
rnaqa_merged.jsonl   88,743 条   RNA 24,207 条
  rfam     39,452   44.5%    19,726 RNA   2.0 QA/RNA
  seq8     35,848   40.4%     4,481 RNA   8.0 QA/RNA
  freetext 13,443   15.1%     4,481 RNA   3.0 QA/RNA

freetext ∩ seq8 = 4,481  （freetext 独有 0 —— 一条新 RNA 都没有）
rfam     ∩ seq8 = 0
```

**freetext 的增量只有 3 个新问法，RNA 多样性 = 0。**

```
freetext 退化为拒答串 'The provided annotation does not state this.'
    1,957 / 13,443 = 14.56%（疾病模板 32.3%，功能模板 10.5%）
    占合并全集 2.21%
答案平均 228 字符，最长 465，训练集 8,496 个互不相同的长描述
    -> exact match 恒为 0，只能看 char-F1
众数地板 14.26%（seq8 4.55%，rfam 0.12%）
```

**rfam 探针（scripts/75_rfam_probe.py）**

```
序列 19,726   train 15,780 / test 3,946   家族 3,182 类
众数地板            0.05%
探针准确率（未见表） 14.95%      训练集 83.26%  <- 严重过拟合（每家族约 5 条）
判据               >= 0.40（和 seq8 同门槛）   判定 drop
```

有信号（地板的 300 倍），但比 seq8 弱一个量级。

### 13.4 合并方向错了 —— 已验证模板，不是 RNA

`code_answer_probe`（`scripts/69_`）完整表：

```
0.810  dinucleotide at positions 1 and 2          ┐
0.674  first 4 nucleotides                        │
0.674  reverse complement of the first 4          │ 读序列开头 —— 能读
0.628  What type of RNA is this?  (cat)           │
0.553  first 6 / revcomp 6 / positions 1-6        │
0.446  first 8 nucleotides                        ┘
---------------------------------------------------------------
0.149  last 4 / 0.113 last 6 / 0.089 last 8       ┐ 读序列结尾 —— 读不到
---------------------------------------------------------------
0.037  count of CG / 0.032 G in first 50          ┐
0.00x  count A/C/G/U、GC content、各百分比        ┘ 需要全局聚合 —— 全死
```

**码信道是前向局部的信道。** 这是 pool=4 加位置序编码的性质。

`77_verify_seq_align.py` 逐条验过：**seq8 的 8 个模板里有 7 个能从序列精确复算**
（`4481/4481` x 7），第 8 个 `What type of RNA is this?` 需要标签。

所以正确形式是 **新 RNA x 已验证模板**，而不是原来的 **旧 RNA x 未验证模板**。

### 13.5 新数据集 rnaqa_seqtask.jsonl

```
242,070 条   24,207 RNA（4,481 seq8 + 19,726 rfam，交集 0）   10 个模板
  head 7 个（探针 0.446~0.810，读开头）   169,449 条
  tail 3 个（探针 0.089~0.149，读结尾）    72,621 条
划分  train 193,660 / test 43,920 / val 4,490
缺格 0（24,207 x 10 精确等于 242,070）
uid 在两个缓存里 100% 命中

tail 三组进训练集是为了做**因果**检验：同一批 RNA、同一个信道、同一把尺子，
只换读的位置。head 学得会而 tail 学不会 -> "前向局部"从相关变因果。

global（计数/百分比）默认不进 —— 它们的失败有可解释的原因，
塞进去只会重复 freetext 的错（用不可解的目标稀释梯度）。
```

**指纹子集（替代簇划分）** `data/processed/fingerprint.json`

```
训练集 128,721 条 -> 首 8 唯一值 34,642
测试集  16,480 条：首 8 在训练集见过 83.5%   未见过 16.5%（2,715 条）
                    末 8 在训练集见过 83.5%   未见过 16.5%（2,720 条）
```

在"未见过"子集上单独报准确率，一致就排除记忆；不一致则说明模型在背先验。
配合 C 臂（码向量置零）本来就能量先验本身。

### 13.6b 一处静默 bug：诊断套件读错了隐藏态（2026-10-12 发现，已修）

`scripts/50_routeC_e2e.py` 里 `_logp_of()`（诊断套件的底层）漏传了 uid：

```python
def _logp_of(model, tk, ex_list):
    for r, qa, clen, off in ex_list:
        x = hid_slice(off, clen)        # <- 缺 uid=r.get("uid")
```

`off` 是**该记录在自己那份缓存内部**的偏移。rfam/rfa 的 off 指向副缓存
（最大 47,547,438），而 `hid_slice` 不传 uid 时退回**主缓存**（2,018,834 位置）——
读的是完全无关的位置，**而且不会报错**。

**为什么一直没暴露**：`diagnose()` 只取 `ev_ex[:40]`，而
`rnaqa_merged.jsonl` 的写出顺序是 seq8 -> freetext -> rfam，
前 40 条恰好全是 seq8（走主缓存）。如果数据文件的顺序换一下，
或者 `--data` 直接给 rfam 的记录，就会静默读到错误的隐藏态。

**对已记录结论的影响**：

```
C_merged 的 [diag@3000..15000] 那一串 RNA/问题 0.048 -> 0.228
  不是错的，但它**只测了 seq8 那一片**（ev_ex[:40] 全是 seq8）
  和 B_seq8 的 0.469 确实可比（两边都是 seq8）—— 这条比较成立
  但它对 rfam / freetext **什么都没说**
```

修法两处：
1. `_logp_of` 传 uid
2. 新增 `spread(ex_list, n)`：**等距抽样**而不是取前 n 条。
   数据文件按来源顺序写（seq8 在前），取前 n 条 = 只在 seq8 上监视，
   而训练数据 85% 是 rfa。`diagnose()` 和 `val_ce()` 都改用 spread。

### 13.6 步数才是约束，不是数据量
`batch` 由 `token_budget=600` 个**码 token** 截断（`50_routeC_e2e.py:748-752`），实测约 4 条/步。

```
C_merged    train 71,006 QA     18,000 步 x 4 = 72,000 QA  = 1.01 epoch
新数据集    train 193,660 QA    18,000 步 x 4 = 72,000 QA  = 0.37 epoch
                                48,000 步                  = 0.99 epoch
升级 2 后   train ~1,306,000 QA 一个 epoch 要 326,000 步 ≈ 5.6 天
```

**更多 RNA != 更多训练。** 固定步数下样本被看的次数直接下降。
升级 2 的真正价值是"同样 80,000 次抽样，从 6 倍大的池子里抽，覆盖更杂"——
随机梯度多样性上去了，消耗的数据量没变。所以**步数不需要跟着放大 6 倍**。

### 13.8 合并为什么错：loss 的 token 质量配比（2026-10-12）

**一条一般结论，不只是这一次的数据清洗。**

**测出来的配比**（rnaqa_merged 的训练集，答案 token 用 Qwen3-8B tokenizer 实算）：

```
来源        样本    占比     答案token   占比    每样本token
freetext  10,758  15.2%    619,577   58.8%    57.6
rfam      31,560  44.4%    266,571   25.3%     8.4
seq8      28,688  40.4%    168,236   16.0%     5.9
合计      71,006           1,054,384            14.8
```

**freetext 只占 15.2% 的样本，却占 58.8% 的答案 token —— 也就是 58.8% 的 loss 梯度。**
而 seq8 占 40.4% 的样本，只拿到 16.0% 的梯度：
**唯一一个「答案可证明在信道里」的来源，分到的信号最少。**

**机制：这不是稀释，是反向目标。**

1. freetext 的答案是 (问题, 注释库) 的函数，与 RNA 信道基本无关
2. CE 是极大似然。在这类目标上降低 loss 的**最省力办法是建模 P(答案 | 问题)**
   —— 而那要求**忽略信道**
3. 当这个来源占 58.8% 的梯度，梯度的主方向就是「忽略信道」

对应 PLAN 里已有的三条文献，指向同一件事：
`2308.12553`（CE 的最大间隔偏置**即使数据充分**也会自己造出捷径，不是数据不够）、
`2506.11550`（模态懒惰 / 数据配比）、`2206.04496`（冲突梯度导致模态塌缩）。
这次是我们自己撞上它的一个实例。

**C_merged 的数字因此有了机制解释**：

```
B_seq8    100% seq8，每样本 5.9 个答案 token，全部可算    RNA/问题 0.469
C_merged  58.8% 的 loss 落在信道无关的散文上             RNA/问题 0.228
```

RNA 那一支被压掉一半，是这个配比下**应该**出现的结果，不是意外。
之前只能说"看起来被稀释了"，现在有机制和数字。

**由此得到一条方法论条件：模态对齐阶段的 loss 必须由被对齐的那个映射主导。**

```
条件 1  答案必须能从新信道推出来
        否则最优解就是忽略它
        freetext 违反（答案来自注释库，按构造不在序列里）
        rfam 家族模板也违反（探针 14.95%）

条件 2  token 质量必须落在那类答案上
        否则梯度被别的东西统治
        freetext 违反（15.2% 样本 / 58.8% token）
```

校正后的 `rnaqa_seqtask` 两条都满足：10 个模板答案全是 4~8 字符、
每样本 2~6 个 token、全部可复算、token 质量大致均摊。
**它和 C_merged 的本质差别不是「去掉了坏模板」，是把梯度的主方向从
「忽略信道」翻回「读信道」。**

#### 13.8.1 freetext 到底有没有信号 —— 一个被自己推翻的读法

76_ 在 step_015000 上跑到 freetext 时，A 臂和 C 臂（码向量置零）**逐点完全相等**
（前 35 条）。当时我读成"信道贡献为零"。**那个读法过了，后面被自己的数据推翻：**

```
[freetext] 35/40   A 2.9%   C 2.9%
[freetext] 36/40   A 5.6%   C 2.8%     <- 分开了
[freetext] 37/40   A 5.4%   C 2.7%
```

37 个点里有 2 个 A 命中而 C 没命中 —— **信道在 freetext 上有影响，
只是弱到 37 次里只翻动 2 次输出。**

精确的说法应该是：

```
站得住   码向量置零不改变绝大多数贪心输出（35/37）
站得住   A 臂准确率远低于该来源众数地板 14.26%
站不住   "贡献为零"
         -> 贪心取 argmax，微弱到不翻转 argmax 的信息在 exact match 下看不见。
            "零信息"和"信息很弱"在这把尺子下长得一样。
```

量它的正确仪器是 `scripts/87_channel_by_source.py`：四臂 teacher-forced CE
（A 真实码 / B 换码 / C 置零 / MODE 只答模板众数）。
MODE 那一臂是关键 —— 它是「只读问题」能拿到的 CE，exact match 看不到这一层：

```
CE_A < CE_C   码向量置零会变差   -> 信道在用
CE_A < MODE   比只读问题答众数还好 -> 确实从序列读到了东西
CE_A ≈ MODE   输出基本由问题决定  -> 序列没贡献
```

一次前向 vs 一次生成（35B 上生成实测 40 秒一条），逐 token 比较，弱信号也看得见。

**自由文本的结论**：它不该进模态对齐阶段，但也不该被永久排除。
正确形态是独立阶段，二选一：
- 对齐完成、参数冻结之后只训解码器
- 或把 loss 按 token 显式加权，让对齐那部分占绝对多数

代价要认：去掉 freetext 后 10 个模板的答案全是 4~8 字符的闭式答案，
模型再也不会被要求产出长文本。若最终目标包含"能对 RNA 说人话"，
这个能力就没有训练信号了 —— 但那应该作为独立阶段、配独立度量、单独报数。

### 13.7 本轮新增/改动的脚本

**第二批（§13.8 / §14 对应）**

```
81_probe_qwen36.py     底座可行性（类解析/模块形状/参数账/tokenizer）
82_qwen36_keymap.py    key 比对（决定怎么装权重）
83_verify_full_cache.py 新全量缓存与旧缓存逐位对拍
84_qwen36_quant.py     Qwen3.6 流式装载 + 专家量化（薄 CLI，逻辑在 src/rnaqwen/qwen36.py）
85_moe_forward_equiv.py 专家 Linear 化与官方逐位对拍（含反向）
86_lora_targets_qwen36.py LoRA target 四套预设实测
87_channel_by_source.py  按来源量信道信息量（teacher-forced CE，含 MODE 基准）

src/rnaqwen/qwen36.py  装载逻辑（从 84_ 抽出，50_ 也用它，避免两份实现漂移）
50_routeC_e2e.py       加 --base / --qwen36-mode / --lora-preset；早停；val/test 分离
51_content_init_codebook.py  加 --d-model / --tgt-norm / --out
```

**第一批（§13.1–13.6 对应）**

```
75_rfam_probe.py        rfam 家族探针（14.95%）
76_acc_by_source.py     按来源拆三臂（训练在跑时不要并跑，会 0xC0000005）
77_verify_seq_align.py  序列与缓存逐位对齐 + 答案可复算性
78_build_seq_task_dataset.py  模板 x 新 RNA 数据集（--full 走 split_full.json）
79_encode_rfam_full.py  全量 145,201 编码（偏移量按实际写入条数算，修掉 73_ 的错位隐患）
80_split_full.py        全量划分 + 指纹（split_full.json / fingerprint.json）
```

---

## 13.10 合并的最终判决（2026-10-12，script 76_ + 87_ 的思路）

`scripts/76_acc_by_source.py --per-source 40` 跑在 `C_merged/step_015000`（10.4 h，已训 83%）上，
三臂 exact match 按来源拆开：

```
来源         n    accA   accB   accC      A-C    F1A   F1B   F1C     地板      判定
freetext   40    5.0%   5.0%   2.5%    +2.5pp  0.57  0.56  0.56  14.26%   没学
rfam       40    0.0%   0.0%   0.0%    +0.0pp  0.53  0.38  0.51   0.12%   没学
seq8       40   45.0%   7.5%  17.5%   +27.5pp  0.88  0.45  0.65   4.55%   学码信道
加权总     120   16.7%   4.2%   6.7%
```

**对照 `B_seq8`（纯 seq8、14,000 步）：`A 80.0% / B 7.0% / C 10.5%`。**

**同一个来源、同一批模板、同一把尺子，合并把 A 臂从 80.0% 砍到 45.0%。**
稀释在**生成侧**被独立证实，不再只是 `Δ换RNA/Δ换问题` 那个诊断比值。

三支的失败方式各不相同，都能解释：

| 来源 | 占训练数据 | 占 loss token | accA | 地板 | 怎么回事 |
|---|---|---|---|---|---|
| seq8 | 40.4% | 16.0% | 45.0%（单独跑 80.0%） | 4.55% | 被 59% 的反向梯度拖掉一半 |
| rfam | 44.4% | 25.3% | **0.0%** | 0.12% | 家族模板探针仅 14.95%，实际零信号 |
| freetext | 15.2% | **58.8%** | 5.0% | 14.26% | 答案不在序列里；连地板都够不到 |

freetext 的样例输出全是记忆里的模板：问 RASSF1-AS1 的描述，吐的是粟酒裂殖酵母线粒体大亚基的句子。
rfam 猜 `RF00001`（答案 `RF00177`）。**两支都在输出训练集里的高频串，不是在读序列。**

**结论：合并不是"不够好"，是把主方向从「读信道」翻成了「忽略信道」。
校订后的 `rnaqa_seqtask`（10 个可复算模板、答案 4~8 字符、token 质量均摊）两条条件都满足。**

---

## 14. 换底座：Qwen3.6-35B-A3B 全量训练 + 早停（2026-10-11 起）

### 14.1 底座事实（全部实测，不是文档抄来的）

```
路径      C:\qwen-api\models\Qwen3.6-35B-A3B     66.99 GB   26 分片
顶层      model_type qwen3_5_moe    arch Qwen3_5MoeForConditionalGeneration（VLM）
文本      text_config.model_type qwen3_5_moe_text   hidden 2048   40 层   vocab 248320
层型      30 x linear_attention + 10 x full_attention（full_attention_interval = 4）
          linear_attn 是 Mamba 式 SSM：A_log / dt_bias / conv1d(kernel 4) / in_proj_qkv
          attn_output_gate = true（q_proj 输出 8192 = 16 head x 256 x 2）
MoE       256 experts / top-8   moe_intermediate 512   shared_expert 512
其他      head_dim 256   16 attn heads   2 kv heads   mrope   mtp_num_hidden_layers 1
          linear_conv_kernel_dim = 4（和我们的 pool=4 无关，是巧合）

参数量分布
  text      692 张量   34.152 B   94.99%
  mtp        19 张量    0.845 B    2.35%
  lm_head     1 张量    0.509 B    1.41%
  vision    333 张量    0.447 B    1.24%
  合计                35.952 B

  bf16 文本+lm_head  64.6 GiB  > 63.6 GB 物理内存（§12 的结论对上了）
  8bit               32.3 GiB
  4bit               16.1 GiB
  视觉塔 bf16         0.83 GiB（不值得做 stub，装了再 del）
```

### 14.2 权重装载：认前缀的是 VLM 类，而且**不要改前缀**

```
直接比对               交集 1       只在 ckpt 1,044   只在模型 692
去掉 model.language_model. 前缀   交集 693     模型里缺 0
残余                    333 视觉塔 + 19 mtp  -> 丢弃
```

transformers 5.9.0 源码：

```python
# modeling_qwen3_5_moe.py:1837  Qwen3_5MoeForConditionalGeneration
_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
# modeling_qwen3_5_moe.py:903   Qwen3_5MoeForCausalLM
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

**上面那个"去掉前缀 693/693 命中"是相对文本-only 类说的。**
文本-only 类没有 `language_model` 这一层，所以它的 key 是 `model.layers.*`。
但从同一个 config 也可以用 `AutoModelForCausalLM.from_config` 建出
`Qwen3_5MoeForCausalLM`（34.66 B，顶层 `['model','lm_head']`）——
**那是个陷阱**，因为 `from_pretrained` 会用它，然后 690 个文本 key 全部 missing，
静默装出一堆随机权重。

**实测结论（84_ 里已按这个写，并踩过一次反例）：**

```python
cfg = AutoConfig.from_pretrained(MDL)                       # Qwen3_5MoeConfig
m   = Qwen3_5MoeForConditionalGeneration(cfg)               # VLM 类
# checkpoint 的 key 与 VLM 类**逐字相同**，包括 model.language_model.layers.*
# 所以装载时**一个字都不要改**，只丢弃 mtp.* 和 model.visual.*
text    = m.model.language_model
lm_head = m.lm_head
del m.model.visual                                          # 回收 0.83 GiB
```

反例（实测报错）：如果按"去掉 `model.language_model.` 前缀"去构造 state_dict，
`load_state_dict` 会把 `model.layers.0.input_layernorm.weight` 全部报成
**unexpected**，那一层 528 个参数原地不动留在 meta 上，
直到 `.to(DEV)` 才炸 `Cannot copy out of meta tensor`。

**还有一个不会自己报错的坑：**

```
rotary_emb.inv_freq / original_inv_freq 是 __init__ 里算出来的，不在权重文件里
-> strict=False 的 load_state_dict 不会碰它 -> 永远停在 meta 上
-> 只在第一次前向或 .to() 时炸
修法：整个模块在真实设备上重建（它只有 buffer，没有参数）
      text.rotary_emb = type(text.rotary_emb)(tcfg, device=DEV)
```

### 14.2b 流式装载（scripts/84_qwen36_quant.py）

```
模型先在 meta 上建（不占内存，3s）
逐层：读盘 -> load_state_dict(frag, strict=False, assign=True) -> .to(DEV)
      -> 专家 Linear 就地换成 bnb int8 -> 释放 bf16 副本
```

关键是 `assign=True`：它**替换** Parameter 对象而不是 `copy_`，
所以能写进 meta 参数。峰值内存压在一层上，而不是整个模型的 67 GiB。

实测（smoke 3 层）：

```
layer  0   528 张量  量化 512 个 Linear   9.4s
layer  1   528 张量  量化 512 个 Linear   7.9s
layer  2   528 张量  量化 512 个 Linear   8.1s
无残留 meta 参数、无 unexpected key
40 层预计 5~6 min
```

**不要走 `transformers.integrations.bitsandbytes`。** 5.9.0 的
`BitsAndBytesConfig` 只剩 `quant_method` 一个 dataclass 字段，
`replace_with_bnb_linear(model, modules_to_not_convert, ...)` 不再返回替换后的模块。
直接用 `bitsandbytes.nn.Linear8bitLt` + `Int8Params`，
量化发生在 `Int8Params.cuda()` 里（所以先造对象再 `.to(DEV)`）。

### 14.3 内存：bnb 量化不到专家 —— 这是启动前的主工程量

```python
# modeling_qwen3_5_moe.py:736
class Qwen3_5MoeExperts(nn.Module):
    self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*intermediate, hidden))
    self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden, intermediate))
    ...
    gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, -1)
```

专家是**裸的 3D `nn.Parameter`**，不是 `nn.Linear`。bnb 的 `replace_with_bnb_linear`
只替换 `nn.Linear`，**量化不到它**。

```
256 x 1024 x 2048 = 0.537 B/层   (gate_up)
256 x 2048 x 512  = 0.268 B/层   (down)
          合计     0.805 B/层 x 41 层 = 33.01 B
text 总 34.152 B  ->  专家占 96.7%

专家 bf16            33.0 x 2 = 61.5 GiB   <- 光这一项顶满 63.6 GB
embed + lm_head bf16 0.51+0.51 =  1.9 GiB
其余 ~1.1 B 量化后              ~2~6 GiB
4bit 也要 ~66 GiB  ->  装不下
```

**§12.2 的"量化可行性已验证"是建立在 bnb 能覆盖全部权重的前提上的，对这个模型不成立。**
（那次验证是在 8B 的 `nn.Linear` 上做的。）

两条可行路线：

```
路线 1  把每个专家拆成独立的 nn.Linear
        256 x 41 x 2 = 20,992 个 Linear，改 MoE 的 forward 按 active expert 派发
        然后 bnb 的 int8 / int4 内核照常可用
        int8 专家 30.7 GiB / int4 专家 15.4 GiB
        改动集中在 Qwen3_5MoeExperts 一个类里，复用 bnb，风险低

路线 2  自己写 int8 专家 + 按需反量化
        只反量化 top-8 命中的专家
        带宽代价：每 token 每层读 8 x 3.1 M x 41 = 1.03 GB int8 —— 慢
        不划算，作为备选
```

**选路线 1。** 目标内存账：

```
int8 专家 30.7 + 其余 bf16/8bit ~7 + LoRA/优化器/激活 ~6 = ~44 GiB  < 63.6 GB  ✅
int4 专家 15.4 + 其余 ~7 + ~6                            = ~28 GiB  ✅ 更宽裕但代码更多
```

先做 int8。跑通再考虑 int4。

### 14.4 LoRA target 必须重选（现在的配置会漏掉 30/40 层）

```python
target_modules=["q_proj","k_proj","v_proj","o_proj"]   # 只命中 10/40 层
```

真实模块名与形状：

```
full_attention（10 层）  self_attn.q_proj [8192,2048]  k_proj [512,2048]
                        v_proj [512,2048]  o_proj [2048,4096]
linear_attention（30 层）linear_attn.in_proj_qkv [8192,2048]  in_proj_z [4096,2048]
                        in_proj_a [32,2048]  in_proj_b [32,2048]  out_proj [2048,4096]
                        conv1d [8192,1,4]
MoE 共享专家（40 层）    shared_expert.gate_proj/up_proj [512,2048]  down_proj [2048,512]
```

LoRA 参数量（r=8, alpha=16）:

```
A  只在 full 层的 q/k/v/o                        1.72 M
B  A + 线性层的 in_proj_qkv / in_proj_z / out_proj  7.13 M
C  B + 线性层的 in_proj_a / in_proj_b              8.13 M
D  C + 共享专家的 gate/up/down                    10.59 M
```

`mlp.experts.gate_up_proj / down_proj` 无 `.weight` 后缀、是堆叠 3D Parameter，**不列入候选**。

**选 C。** 30/40 层是线性注意力，只打 full 层的 q/k/v/o 等于让 3/4 的网络混合器完全不动，
注进去的 RNA 向量出不来。加上 a/b 是因为它们是 SSM 的门控参数，直接决定状态更新。
10.59 M（D）也可以接受，但先跟 C 对齐最小改动。

### 14.5 内容初始化要重建（d = 4096 -> 2048）

`checkpoints/fsq_full/code_emb_contentinit.pt` 是 (4096, 4096)。
新底座 `hidden_size = 2048`，码表是 (4096, 2048)。
`51_content_init_codebook.py` 要能以 35B 的 `embed_tokens` 为目标重跑一遍。

### 14.6 早停（代替固定步数）

**为什么必须换掉固定步数**：§13.6 的账。数据涨了 6 倍而步数不变等于欠训；
步数按 6 倍放大等于 5.6 天。固定步数在两个方向上都是错的。

**监视什么**

不用总 val loss —— 它被简单模板主导，和我们要的东西（三臂对比、head/tail 对照）不对齐。

```
主指标   val 上 **A 臂（真实 RNA）的答案 CE**，用 §11 修正后的无泄漏 prompt
        注意：绝不能再用 gen() 那种把答案喂进去的构造（§11）
副指标   val 上 C 臂（码向量置零）的 CE
        健康判据是 gap = CE_C - CE_A 在扩大。CE_A 降但 gap 不涨 = 在学先验不是学信道
加速     只算 CE，不生成。生成要解码，35B 上代价高得多
```

**早停参数**

```
eval-every      500 步（A/C 两臂各一次前向，很便宜）
min-delta       相对改善 < 0.5% 视为无改善
patience        6 个 eval 点（= 3,000 步）
min-steps       4,000（防止早期噪声触发）
max-steps       上限，比如 60,000
保存            按主指标保存 best，不是 latest
```

**必须一起改的**：现在 `50_` 只在 `step % save_every == 0` 存盘（续训用），
早停需要另存一个 `best.pt`，且 best 的判据、耐心计数、`stop_reason`
都要写进日志和 checkpoint。

`--eval-every 3000` 现在太稀（3,000 步一个点，patience 6 就是 18,000 步），
要降到 500。

### 14.7 LoRA target 实测确认（scripts/86_lora_targets_qwen36.py）

PLAN 里那四套 target 原来是**按模块名推的**，现在实测过了。先看真实的 Linear 分布：

```
40/40  mlp.shared_expert.gate_proj / up_proj / down_proj
40/40  mlp.shared_expert_gate               <- Linear(2048,1)，任何一套都不碰它
30/40  linear_attn.out_proj / in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
10/40  self_attn.q_proj / k_proj / v_proj / o_proj
```

peft 的实际挂载结果（meta 上建模型，不耗内存）：

```
A 只 full 层 q/k/v/o        命中  40 模块   LoRA  1.72 M
B + 线性层 qkv/z/out        命中 130 模块   LoRA  7.13 M
C + 线性层 a/b              命中 190 模块   LoRA  8.13 M
D + 共享专家 gate/up/down   命中 310 模块   LoRA 10.58 M
对照 Qwen3-8B                命中 144/144     LoRA  7.67 M
```

**四套的参数量和推算完全一致**，对照组的 144 == 36 层 x 4 说明方法本身没写错。
`o_proj` 确实不匹配 `out_proj`，`q_proj` 也不匹配 `in_proj_qkv` ——
原来 8B 那套确实只覆盖 10/40 层的注意力。

**§14.4 选 C 不变**，且已实测可用。

### 14.8 route C 的注入路径可以直接用 VLM 类

```
Qwen3_5MoeForConditionalGeneration.forward(
    input_ids, attention_mask, position_ids, past_key_values,
    inputs_embeds, labels, pixel_values, ..., **kwargs)
```

**接受 `inputs_embeds`。** 所以不需要绕到 `model.language_model` 再手动接 `lm_head`——
直接用 VLM 类做前向即可，peft 也照常包在外面。
（一度担心 VLM 的 forward 只认 pixel_values 那条路，实测不是。）

### 14.9 tokenizer 实测（换底座必须重新验，这里踩过坑）

```
                      Qwen3-8B          Qwen3.6-35B-A3B
class                 Qwen2Tokenizer    Qwen2Tokenizer
vocab                 151643            248044      （模型 embedding 248320）
eos                   151645  248046    <|im_end|>
<|im_start|>          151644            248045
pad                   <|endoftext|>     <|endoftext|>  248044
sys_ids                [151644,8948,198]        [248045,8678,198]
user_ids       [151645,151644,872,198]   [248046,248045,846,198]
asst_ids       [151645,151644,77091,198] [248046,248045,74455,198]
no_think 长度          13                14          <- 变了
no_think decode        一致，都是 ' thinking\n\n<｜end▁of▁thinking｜>\n\n'
```

两件要注意、但**都不用改代码**：

1. `no_think` 从 13 token 变 14 token。`Tok.__init__` 是**从 tokenizer 现算的**
   （`tk.no_think = tk(" thinking\n\n<｜end▁of▁thinking｜>\n\n", add_special_tokens=False).input_ids`），
   不是硬编码。这正好避开了「THINK THE FIRST OF」那类坑的根源——
   当初就是因为 token 序列和 Qwen3 的约定不一致。
2. `text_config.eos_token_id` = 248044（`<|endoftext|>`），
   而 tokenizer 的 eos 是 248046（`<|im_end|>`）。`Tok` 取后者：
   训练目标要求模型学会吐 `<|im_end|>` 才停下来（`build_batch` 里
   labels 是从 assistant 头开始算的），和 8B 同构。

占位 token 用 `tk.base_vocab = model.base_vocab`：8B 是 151936，35B 是 248320，
都越过 embedding 行数，作哨兵，由 `model.embed` 替换。

### 14.10 MoE 改写正确性：与官方逐位等价（scripts/85_moe_forward_equiv.py）

84_ 证明了权重**装得进去**，85_ 证明**算得对**：

```
[2] T=1/4/17/33/64/128，K=1/2/8      forward |Δ|max 0.000e+00   grad |Δ|max 0.000e+00
[3] silu / gelu / relu               0.000e+00
[4] float32 / bfloat16               0.000e+00
[5] 512 token 全走专家 0（index_add_ 累加路径）  0.000e+00
[6] 真实维度 E=256 H=2048 I=512      0.000e+00
```

**不是"在容差内"，是字面完全相等**，forward 和反向都是。

配套的坑：transformers 5.9 用 `integrations/moe.py` 把 `Qwen3_5MoeExperts.forward`
包了一层，包装器读 `self.config._experts_implementation`。
真实 config（PretrainedConfig 子类）有这条属性，**裸测试类没有**，
不带就 `AttributeError`。测试里得自己补 `_experts_implementation = "eager"`。

### 14.11 换底座前的实测结果（截至 2026-10-12 早）

| 项 | 结论 | 脚本 |
|---|---|---|
| 类可解析 | `Qwen3_5MoeForCausalLM` / `ForConditionalGeneration` / `TextConfig` 都在 | 81_ |
| 权重装载 | 走 VLM 类，key **逐字相同**，不要改前缀 | 82_/84_ |
| meta buffer | `rotary_emb.inv_freq` 必须重建，否则 `.to()` 时炸 | 84_ |
| 专家量化 | 3D `nn.Parameter`，bnb 碰不到 -> 拆成 20,992 个 Linear | 84_ |
| 流式装载 | 3 层实测 8~9s/层，无残留 meta、无 unexpected | 84_ |
| MoE 正确性 | 与官方**逐位等价**（含反向） | 85_ |
| LoRA target | 四套全部实测挂上，C = 190 模块 / 8.13 M | 86_ |
| inputs_embeds | VLM 类接受，route C 不用绕路 | 86_ 附带 |
| tokenizer | `no_think` 13 -> 14 token，无硬编码，无需改 | 附带 |
| 线性注意力 | `flash-linear-attention`/`causal-conv1d` 未装，走 torch 回退 | 全程可见 |

**仍未做的一件事**：全量 40 层装载 + 真实前向。需要约 34.9 GiB，
C_merged 还在跑的时候不并跑（§14.3 的内存账）。

### 14.12 启动前检查清单（其中打勾项见 §14.11）

```
[x] 专家 Linear 化 + int8 量化跑通（84_ smoke 3 层：8~9s/层，无残留 meta、无 unexpected）
[x] 从 VLM 目录装权重（key 逐字相同，不要改前缀；§14.2）
[x] MoE forward/backward 与官方逐位等价（85_）
[x] LoRA target 实测挂上（86_，四套全过，C = 190 模块 / 8.13 M）
[x] Tok.no_think 在新 tokenizer 上重新验（13 -> 14 token，现算不硬编码，无需改）
[x] tokenizer vocab 248320 vs 8B 的 151936；占位 token 用 base_vocab 作哨兵
[ ] 全量 40 层装载 + 真实前向（需要约 34.9 GiB，等 C_merged 结束）
[ ] 抽样核对若干层权重和 ckpt 一致（不是只看 key 对上了）
[ ] code_emb_contentinit 重建到 d=2048（51_）
[ ] 早停的 A/C 两臂 prompt 过一遍 §11 的静态 leak 断言
[ ] flash-linear-attention / causal-conv1d 都没装，线性注意力走 torch 回退
    —— 先量一下每步耗时，慢到不可接受再考虑装
[ ] 50_ 加 --base 分支：8B 走原路，35B 走 84_ 的流式装载 + 8.13 M LoRA
[ ] 早停落地：best.pt、min-delta、patience、stop_reason 进日志和 checkpoint
[ ] batch/token_budget 与 8B 保持一致，便于对照
```

### 14.14 硬阻塞：MIOpen 不支持 Conv1d —— 30/40 层跑不起来（已修）

**在极小 MoE 上排练时崩出来的**（scripts/89_tiny_moe_vlm.py）：

```
RuntimeError: miopenStatusInternalError
  modeling_qwen3_5_moe.py:493  F.silu(self.conv1d(mixed_qkv)[:, :, :L])
  torch/nn/modules/conv.py:380 -> F.conv1d
```

这正是 PLAN §12.6 早就记过的那个坑（本项目的 `FSQCode` 压缩层当初就是因此
从 Conv1d 改成 unfold+Linear 的）。而 35B 的 **30/40 层**是 `linear_attention`，
每层都有：

```python
self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=False,
                        kernel_size=4, groups=conv_dim, padding=3)
```

**不修这一条，35B 一步都跑不了。**

**修法**：深度可分离 + kernel=4 + padding=3 的因果卷积 = 4 个移位乘加：

```
out[c, t] = Σ_{k=0..3} w[c, 0, k] · x[c, t + k − 3]
```

纯逐元素与广播，不经过任何 conv 算子，不需要 unfold+Linear（那要 8192×32768 的矩阵）。
**保持参数名 `conv1d.weight` 不变**，checkpoint 的 key 才对得上。
实测数值等价：`最大差 1.192e-07`（float32 eps 量级）。

代码在 `src/rnaqwen/qwen36.py: patch_conv1d()`。

**遗留风险**：替换实现的反向会保留 4 个乘积项 × 30 层，粗估 8.6 GiB 额外激活
（B=4, C=8192, L≈1100, bf16）。若显存吃紧，下一步是给 decoder 层开
gradient checkpointing。

### 14.15 另一个只在非 meta 路径上暴露的 bug（已修）

模型的 `_init_weights` 是**按属性名**访问专家的：

```python
elif isinstance(module, Qwen3_5MoeExperts):
    init.normal_(module.gate_up_proj, mean=0.0, std=...)
    init.normal_(module.down_proj, ...)
```

而专家 Linear 化之后只有 `gate_up` / `down` 两个 ModuleList ——
于是非 meta 设备上建模型直接 `AttributeError`。

**一直没撞上，只是因为 `load_qwen36_text` 总在 `torch.device("meta")` 下建模型**
（meta 上跳过权重初始化）。这个前提一旦不成立就崩。
修法：`patch_moe_experts()` 里连 `Qwen3_5MoePreTrainedModel._init_weights` 一起包掉，
逐个子 Linear 初始化。

### 14.16 peft × bnb × inputs_embeds 集成验证（scripts/88_）

86_ 测的是 meta 上的普通 `nn.Linear`，而真跑时那些 Linear 是 **bnb 量化过的**，
peft 走的是另一条路径。88_ 补上：

```
quant=none prep=0   挂载 8 个 LoRA   save过滤✓  输入梯度✓  LoRA梯度✓
quant=int8 prep=0   量化 15 个 Linear  同上全 ✓
quant=int8 prep=1   同上全 ✓
quant=int4 prep=0   同上全 ✓      <- 35B 实际配置
quant=int4 prep=1   同上全 ✓
```

三条命门都通：
1. peft 能挂到 bnb 量化过的 Linear 上（int8 / int4 都行）
2. `save_ckpt` 的 `"lora_" in k` 过滤能命中
3. **梯度能回到 `inputs_embeds`** —— route C 的码表和 FSQ 全靠这条回传
4. 顺带：`prepare_model_for_kbit_training` **不是必需的**（prep=0 也行），少一个偏离

### 14.17 整条 35B 接线的极小规模排练（scripts/89_）

用**同一个 VLM 类、同一套专家改写、同一套 LoRA target**，
把 config 缩到 hidden 64 / 4 层 / 8 专家，在真实设备（非 meta）上整条走一遍：

```
[conv]    替换 vs nn.Conv1d  最大差 1.192e-07      ✅ 数值等价
[build]   Qwen3_5MoeForConditionalGeneration  0.52 M
[experts] 4 层换成 per-expert Linear
[quant]   量化 64 个专家 Linear (int4)
[lora]    target 9 项  挂载 18 个  可训 18.6 K
[fwd ]    logits 有限
[bwd ]    输入梯度 ✓ |g_v|max 1.385e-04   LoRA 梯度 ✓（36 张量）
[save]    含 'lora_' 的 key 36 个  过滤 ✓
```

实际 key 形态：
`base_model.model.model.language_model.layers.0.linear_attn.out_proj.lora_A.default.weight`
—— 命中的是 `linear_attn`，说明 C 预设按设计生效（不是只打到 10/40 层的 full attention）。

### 14.19 真正的内存杀手：mmap 页常驻工作集（2026-10-12 定位并修复）

> **读这节之前先分清两个账（2026-10-12 下午补，见 §15.2.1）。**
> 只读文件映射的页**计入 WorkingSet，但不计入提交量**。本节讲的是前者 ——
> `trim_working_set()` 确实有效，RSS 被压到 0.01 GiB。
> 但**不要因此以为它在吃提交量**：实测 `np.memmap(mode='r')` 读遍 2 GB，
> 提交量只涨 0.01 GiB（对照组匿名分配 2 GB 涨 2.00 GiB）。
> 这台机器上提交量的账在 GPU 侧，不在映射上 —— 我因为混了这两个账，
> 把一次 OOM 误归因给副缓存的 73 GB，绕了一个多小时。

**症状**：全量 40 层 int4 装载在 20~31 层之间硬崩，**没有异常栈**。

**为什么难查**（先后排除四个假设，全部证伪）：

```
假设                          实验                                    结果
page cache 被分片读撑满        顺序读 18 GiB 分片                      空闲一点没掉
safetensors mmap 不释放        读 14.3 GiB、26 个句柄全留着            空闲一点没掉
GPU 分配对系统内存计价         纯分配 14 GiB 显存                      空闲一点没掉
bnb 量化泄漏                   单个专家量化 ×3、堆叠张量 256 个视图量化   都不涨
```

（另外我自己的 RSS 读数还是坏的：`powershell -Command` 里 `$PID` 是
**PowerShell 自己的进程号**，不是 Python 的，量出来永远是 0.07 GiB。
必须用 `os.getpid()` —— 这个假读数把排查带偏了一轮。）

**逐个时间点切开一层，才定位到：**

```
c. del frag + gc        RSS 0.77
d. swap_linears         RSS 2.74   <- +1.97 GiB，就在这一步
e. to(dev)              RSS 2.59
```

**机制**：safetensors 是 mmap 的，量化一个专家要 `.half()` 它的权重，
这会**读遍每一页**；那些页 fault 进进程工作集，而 `WorkingSet64`
**计入常驻的文件映射页**。

```
实测：摸 8.0 GiB 分片  ->  RSS +8.1 GiB，系统空闲同步掉 8.1 GiB（1:1）
一层专家 bf16 = 1.6 GiB，40 层 = 64 GiB
-> 直接压死这台 63.6 GB 的机器
```

**修法**：每层量化后修剪工作集。两个公开 API，不需要提权：

```python
psapi.EmptyWorkingSet(h)
kernel32.SetProcessWorkingSetSize(h, (SIZE_T)-1, (SIZE_T)-1)   # -1,-1 = 立即修剪
```

**argtypes 必须声明**，否则 HANDLE 被当 int，报 `OverflowError`。

效果：

```
修剪前：摸 11.2 GiB -> RSS 8.51 GiB，空闲 43.4
修剪后：摸 11.2 GiB -> RSS 0.07 GiB，空闲 51.0
```

**全量装载结果（scripts/90_full_load_forward.py，实测）：**

```
40 层   量化 20,480 个 Linear   用时 480s   显存 18.18 GiB
每层 RSS 稳定 0.01~0.02 GiB   系统空闲稳定 51.0~51.3 GiB（不再逐层下滑）

(a) input_ids 前向      logits (1,7,248320)  有限    4.26s（含首次预热）
(b) inputs_embeds 前向  logits (1,7,248320)  有限    1.93s
(c) 反向                g_v 非零 |g_v|max 6.348e-02  3.26s
前向+反向峰值显存 23.77 GiB
```

**顺带修掉的两个收尾 bug**（都会在第一次前向才炸）：

1. **漏了 `model.language_model.norm.weight`**。硬编码的收尾清单只有
   embed_tokens + lm_head，而最后那个 RMSNorm 既不在 `layers.*` 下、
   也不是那两个。它小到在"还有 X B 在 meta 上"里显示成 0.000 B，看不出是谁。
   改成**自动补载所有仍在 meta 上的参数**。
2. **非层内模块没搬到设备**。层是逐个 `.to(dev)` 搬的，
   `embed_tokens` / `norm` / `lm_head` 不在层里 → 前向时报
   `index is on cuda:0, different from other tensors on cpu`。
   现在收尾时显式搬，并断言没有参数留在 CPU 上。

### 14.20 实测步速与可行的预算（2026-10-12 11:11）

**D_qwen36 第一次启动（`--steps 150000`）跑出的真实步速：**

```
=== 训练 150000 步（从 1）===
  step     1  ce 4.3702  ent 3.109  |grad_fsq 1.005e+01|  码表漂移 1.6%  576s   （含首次预热）
  step    20  ce 0.8504  ent 3.567  |grad_fsq 6.739e-01|  码表漂移 8.4%  876s
-> 19 步 / 300 秒 = **15.8 秒/步**
```

**结论：150,000 步 = 27.4 天，不可行。** 已中止重配。

几个必须一起看的数：

```
装载 476~480 s（40 层 / 20,480 个量化 Linear / 18.18 GiB 显存）
前向+反向峰值显存 23.77 GiB
CE 19 步内 4.37 -> 0.85，|grad_fsq| 非零、码表漂移 8.4%
-> 链路是通的，只是慢
```

慢在哪：MoE 专家被拆成 per-expert Linear 后，每层要跑
`expert_hit` 循环（top-8 命中 -> 8 次 `torch.where` + 2 次小 matmul + `index_add_`），
40 层 = 约 320 次 Python 级小算子调度/前向，**开销由调度主导而不是算力**。
再加上 conv 替换的 4 项乘加 × 30 层。

**重配后的预算**（已启动）：

```
--steps 30000            最坏 5.5 天，早停会提前切
--eval-every 750         第一次评估在 ~3.3 h
--min-steps 1500         早停从 ~6.6 h 起算
--patience 4             连续 4 个评估点（3,000 步）无改善就停
--eval-n 40              每次 2 臂 x 40 次前向，约 4~5 min
--save-every 1500
```

**第一次的配置错在**：`eval-every 3000` + `min-steps 6000` 意味着**第一次早停检查在 13 小时之后**，
来不及救。早停的粒度必须和步速匹配 —— 这是 §14.6 里那条警告
（"eval-every 太稀，patience N 等于 N x eval_every 步"）的实例，只是我把它设得比那条警告还稀。

### 14.21 本节一路修掉的东西（都是静默失效类）

```
1  早停没有真的停          缺 break，会打印"早停触发"然后跑满 steps
2  tokenizer 写死 8B       跑 35B 时用 8B 的特殊 token，不报错
3  早停判据被噪声主导       eval-n=12 -> 标准误 0.22 vs min-delta 0.005（差 40 倍）
                           -> 判据加 se 项：need = max(min_delta*CE, se)
4  final.pt 记错 step       写 args.steps，即使早停在 12,000 也记成 150,000
5  val_ce 跑在 train 模式   监视量带 dropout
6  内存守卫阈值过激         系统"空闲"含可回收 page cache，误报中止
7  _logp_of 漏传 uid        诊断套件读到错的缓存（不崩，因为前 40 条恰好是 seq8）
8  spread() 缺失            监视只取前 n 条 = 只在 seq8 上监视，而训练 85% 是 rfa
9  MIOpen 不支持 Conv1d     30/40 层跑不起来 -> 4 项移位乘加等价替换（1.19e-07）
10 _init_weights 属性名访问 专家 Linear 化后没有 gate_up_proj，非 meta 路径崩
11 漏了 language_model.norm.weight  尺寸小到显示成 0.000 B，看不出是谁
12 非层内模块没搬设备       embed_tokens/norm/lm_head 留在 CPU，前向才炸
13 mmap 页常驻工作集        每层 +1.6 GiB，40 层压死机器（硬崩、无栈）
                           -> 每层 trim_working_set()，RSS 稳定 0.01 GiB
```

### 14.18 顺序

C_merged（18,000 步）和升级 2 编码（145,201 条）同时在跑，约 5.5 小时后落地。
到点先做两件不依赖新底座的事：

```
1. scripts/76_acc_by_source.py --per-source 60
   -> C_merged 里 seq8 那一支的准确率，和 B_seq8 的 A 80.0% / B 7.0% / C 10.5% 比
   -> 这一段是"合并到底有没有稀释 seq8"的答案，对照臂是现成的，不需要重训
2. 用 split_full.json 把 rnaqa_seqtask 扩到 145,201 RNA（升级 2 之后）
```

然后再动 35B。**不要在没有按来源结果的情况下先动底座** ——
那次实验是免费的，而换底座会把所有变量一起换掉。

---

## 15. 2026-10-12 下午：SOS 适配器、GPU 内存账、checkpoint 机制

### 15.1 SOS-LoRA 在 35B 上挂得上（我先前判断错了两次）

论文 arXiv 2607.16252v1 的正交多专家 LoRA。**结论：35B 可用，而且参数量与 peft LoRA 严格相等。**

```
[sos] Qwen3.6-35B  替换 190 个 nn.Linear  K=4 r_tot=8 γ_max=2.5  层数核对 190
[sos] 可训 8.13 M
[lora] 可训参数 8.13 M
```

190 = 10 层全注意力 x4（q/k/v/o）+ 30 层线性注意力 x5（in_proj_qkv/z/a/b + out_proj），
与直接读 `model.safetensors.index.json` 数出来的完全一致。权重表里 `q_proj` 有 11 个，
多出的那 1 个不在文本模型里（视觉塔之类），所以是 10 不是 11。

**为什么 8B 不行而 35B 行**：`50_routeC_e2e.py:253` 那个 `nn.Linear` 守卫写在 `else`（8B）分支里。
8B 那条路是**整模型** bnb 量化，全是 `Linear8bitLt`/`Linear4bit`，SOS 挂不上。
35B 那条路只量化 MoE 专家（40 层 x 512 = 20,480 个），**注意力层保持 nn.Linear** ——
这是 `qwen36.py` 刻意的设计，日志里"量化 20,480 个 Linear"正好等于专家数。

三个只在真机上才暴露的问题（单测照不出来）：

```
1  A/B 建在 CPU         __init__ 处理了 dtype、漏了 device。
                        前向第 1 步直接炸：
                        `mat2 is on cpu, different from other tensors on cuda:0`
                        （sos_lora.py:110 的 self.B[k]）。
                        两个坑里 dtype 那个是已知的，模块注释还专门警告过
                        "单测用默认 dtype 测不出来" —— device 这个同样漏在注释之外。
                        scripts/_scratch/_test_sos_lora.py 现在专门加了
                        "底座放 cuda" 那条断言；基准必须取底座自己的 device
                        （torch.device("cuda") == torch.device("cuda:0") 是 False，
                          拿 "cuda" 去比会误报）。
2  正交初始化的 G 也在 CPU  逃过一劫纯粹因为 copy_ 允许跨设备。
                            但每层 4 次没必要的跨设备同步 x 190 层 = 760 次。
3  单测的期望值写错         mock 里每层同时挂了 self_attn(4) 和 linear_attn(5)，
                            目标是 45 不是 23；另外"8 个参数只有 4 个有梯度"
                            是**标准 LoRA 初始化**（A 随机、B 置零 -> grad_A = Bᵀ·上游 = 0），
                            不是 bug。把 B 推离零后 A 才有梯度。
```

### 15.2 内存账 —— 真正的账本在 GPU 侧，不在 Python 堆

这一节花了一个多小时，中途**两次归因错误**，都记下来，因为错的方向比结论更有参考价值。

#### 15.2.1 先把两个错误假设用实验排掉

| 假设 | 验证方式 | 结果 |
|---|---|---|
| 73 GB 副缓存（`_rfa_full_hidden.f16`）的只读 mmap 占提交量 | 建 2 GB 文件，`np.memmap(mode='r')` 读遍每一页，`GlobalMemoryStatusEx` 测提交量增量 | **+0.01 GiB → 推翻**（对照组：匿名分配 2 GB → +2.00 GiB） |
| 训练进程私有堆在膨胀 | 100 秒 x 5 次采样 | 36.80 → 36.80 GB，**平 → 推翻** |

第一条特别值得记住：**只读文件映射的页计入 WorkingSet（§14.19 说的那个杀手），但不计入提交量。**
两者是不同的账。用 `trim_working_set()` 压 RSS 是有效的，但**不要因此以为它在吃提交量** ——
这台机器上提交量的账在别处。

#### 15.2.2 看不见的那部分

```
所有进程私有合计        47.2 GB
系统已提交            122.5 GB   (122,453 MB / 上限 152,578 MB)
内核池（非分页+分页）    4.2 GB
差额                   ~71 GB   <- 不属于任何进程的 Private Bytes
```

WDDM 的 GPU 内存**不进进程 Private Bytes**，所以前面所有按进程求和的做法全都看不见它。
直接查每进程 GPU 计数器：

```
pid_5296（训练）  Shared 28.04 GB   Local 88.53 GB   Committed 88.53 GB
适配器合计         dedicated 61.94 GB   shared 28.24 GB   total 90.44 GB
```

**训练进程的 GPU 提交量 88.53 GB，而 int4 权重只有 18.18 GiB。** 这才是"膨胀"。

它是**真实工作内存**，不是可以回收的分配器缓存，粗算能对上：

```
权重                                18.18 GiB
存给反向用的激活 约 15 GB    40 层 x 8 batch x 1200 token x 2048 hidden x bf16，每层若干张量
MoE 专家分发中间量 约 12 GB  256 专家 / top-8 的 gather + mask
全词表 logits 及梯度 约 10 GB  vocab 248,320 x 8 x 1200 x bf16 ≈ 4.8 GB，再加 softmax 中间量
碎片 + 20,480 个 Linear4bit 的块开销 约 10 GB
```

所以 `torch.cuda.empty_cache()` 救不了，它只能回收未使用的块。
要真降下来只能动 batch / token-budget，或者让 loss 只算答案位置的 logits。

#### 15.2.3 这台机器的账

```
物理内存        63.6 GB 可见（其余被 BIOS 划给核显，这是一台 APU：GPU 显存就是系统内存）
提交上限        ≈ 149 GB = 63.6 物理 + 85.4 页面文件
```

关于这项有一条容易搞错的：`torch.cuda.mem_get_info()` 在这台机器上**撒谎** ——
训练进程持有 18.18 GiB 权重时它报 "空闲 99.58 / 总 99.74，已用 0.15 GiB"。
`qwen36.py` 里 `free_gb()` 的注释早就写过这一点，这次又验证了一遍。只有
`GlobalMemoryStatusEx` / `Win32_OperatingSystem` 那套关于主机内存的数可信。

#### 15.2.4 「之前还好好的」不是真的 —— 一直贴着上限

2026-10-12 17:09 实测提交可用 **0.56 GB**，按当时速率**约 23 秒归零**。
归零就是 `0x800705AF`（"页面文件太小，无法完成操作"），连 `echo` 都跑不了。

但这不是这次才涨出来的：当时系统已提交约 **148 / 149 GB = 99% 满**，
其中 llama-server（站点本地的 Qwen3.6-35B 助手）额外占 30 GB。
**任何抖动**（它的页面文件拆解、一个浏览器标签、另一个 agent）都能把它推过去。

处置：停掉 llama-server，提交可用 0.56 → 30.19 GB，训练全程未受影响。
现在稳定在 119.8 / 149 GB，**余 30 GB，且 100 秒采样完全平**（GPU 提交 +0.00、系统提交 +0.0）。

**结论：没有泄漏，是一次性的稳态占用。**余量只有 30 GB，而它恰好等于 llama-server 原来占的量 ——
所以**训练期间 llama-server 必须保持关闭**（这几天站点改走 DeepSeek API）。

唯一能真正加余量的办法是加大页面文件（磁盘剩 89 GB）。PLAN 里从 §14.19 起就把这条记为
三次硬崩的根因，到现在仍未做。

### 15.3 checkpoint：原子落盘 + 动态时机 + 轮转

原来的三处问题，第一处是**会造成实际损失**的：

```
1  落盘不原子          torch.save 直写目标文件。写到一半被 OOM 杀掉会留下截断的 .pt，
                       而它**看起来合法**：文件在、后缀对、体积也不离谱。
                       实测踩过：step_000600.pt 254.9 MB（正常 322 MB），
                       torch.load 报 "failed reading zip archive:
                       failed finding central directory"。
                       更要命的是 resume 用 `cands[-1]` 按名字排序，
                       **正好会挑中这个坏文件** —— "续训"直接从崩溃点再炸一次，
                       人还以为是新问题。
2  时机只看步数        --save-every 300 x 18.5 s/步 = 92 分钟一个档，
                       崩一次最多丢 1.5 小时，而且步速越慢丢得越多。
3  只留 4 个           够用但偏紧。
```

改法：

```
原子   写 *.pt.tmp -> flush + os.fsync -> 用 zipfile 列一遍中央目录 -> os.replace
       列中央目录比 testzip() 便宜得多（不动数据段），且恰好覆盖"被截断"这个失败模式。
       正式名**永远**指向一个完整文件：要么是新的，要么是上一个。
动态   --save-every 300（步数下限）+ --save-minutes 30（墙上时钟兜底），谁先到算谁。
       丢失窗口从此与步速无关。首次触发实测就是时间兜底：
       `[ckpt] step_000101.pt (171.3 MB, 保留最近 10 个, 时间兜底)`
轮转   --keep-ckpt 10，按 step 排序留最新的 10 个，顺带清 *.pt.tmp。
       **best.pt / final.pt 不参与轮转、不被清理** —— 它们是唯一不可再生的两个档
       （best 是早停要载回的，final 是最终产物）。
```

单例实测 171.3 MB（不是之前记的 322 MB，那是 8B 的码表尺寸）：6104 个 zip 条目、
1520 个适配器张量（= 190 层 x K=4 x (A,B)）、1524 组优化器状态、`rna_emb` (4096, 2048)。
`args` 烘在每个档里，所以任何一份都能自证来源（`adapter=sos keep_ckpt=10 save_minutes=30`）。

单测 `scripts/_scratch/_test_ckpt_rotate.py` 专门钉**静默失效**那几种：
轮转选错方向（留下最旧的 10 个、清掉最新的）、best.pt 被顺手清掉、截断 zip 被放过。

### 15.4 启动脚本自己的三个坑（都不是训练的错）

```
1  管道把 python 带死    $ErrorActionPreference='Stop' 配 `2>&1 | Tee-Object` 时，
                        原生程序写往 stderr 的**第一行**会被当成终止性错误
                        （NativeCommandError），管道当场被拆掉、python 被带死。
                        而 torch 一启动就往 stderr 写 UserWarning（tensor_numpy.cpp:219），
                        于是表现为「装载到一半无声退出，exit 1，日志停在某个中途行」——
                        看起来像 OOM，其实是自己掐的。
                        修：调 python 之前把 ErrorActionPreference 放回 Continue。
2  前置检查假阳性        检查日志文件是否被占用时没先判存在：
                        [IO.File]::Open 在文件不存在时抛 FileNotFoundException，
                        和"被占用"进同一个 catch，再 Remove-Item 又抛一次 ->
                        「日志还不存在」被误报成「日志被占用且删不掉」。
3  ROCm 崩后不退出        python 崩在第 1 步之后**不会真的退出** —— 挂在上下文销毁上。
                        进程一直留着，继续占着 18 GiB 显存和日志句柄，
                        而 Stop-Process -Force **杀不掉**（内核态线程被驱动攥着），
                        只能 taskkill /F /T。
                        后果有两层：下一轮启动的 Tee-Object 直接打不开日志；
                        以及"空闲显存"凭空少 18 GiB —— 看着像 OOM，其实是上一个进程没死。
                        治本：50_routeC_e2e.py 末尾用 _hard_exit()（os._exit）跳过解释器收尾。
                        代价是 atexit 不跑、python 层缓冲不自动刷（所以先 flush）；
                        checkpoint 全是周期落盘，没有"退出时再存一次"的逻辑，所以不丢东西。
                        单测 scripts/_scratch/_test_hard_exit.py 用一个 atexit(睡 600 秒)
                        模拟"收尾会挂"，验 os._exit 确实跳过它、且退出码语义正确。
```

### 15.5 看门狗

`scripts/95_mem_watchdog.ps1`，脱离会话常驻，训练进程消失时自己退出：

```
每 60 秒采一行：commit / phys / step / 训练私有，滚存到 reports/watchdog.log
跌破 15 GB 记 WARN，跌破 8 GB 记 ALARM（记起止时刻）
不自动杀训练 —— checkpoint 是原子落盘 + 每 30 分钟必存一个，
最坏只丢 ≤30 分钟，没必要为了保档去动一个正在收敛的训练
```

### 15.6 本轮的实测数字（供后面估算用）

```
35B int4 / batch 8 / token-budget 1200       19.6 s/步
装载 40 层（20,480 个专家 Linear 量化）       556 ~ 597 s
早停配置  min-steps 1500 / eval-every 750 / patience 4 / min-delta 0.005
          -> 从 1500 步起每 750 步查一次，最迟 4500 步收敛
          -> 跑满 30,000 步约 6.4 天；早停生效则约 7.7 ~ 23 小时
GPU 提交 88.5 GB   进程私有 36.8 GB   提交余量 30 GB
```

---

## 16. 35B / SOS 这一轮的现状（2026-10-13）

### 16.1 配置

```
底座      Qwen3.6-35B-A3B，int4（只量化 20,480 个 MoE 专家 Linear；注意力与 shared_expert 保持 bf16）
适配器    SOS-LoRA  r_tot=8  K=4  alpha=16  dropout=0.05  gamma_max=2.5  ortho_coef=1.0
target    preset C = 9 个注意力投影 -> 190 个 nn.Linear / 8.13 M 可训参数
数据      rnaqa_seqtask_full.jsonl（train 1,297,910 / val 29,500 / test 169,410）
          主缓存 _fsq_full_pos.pt（4,481 条 seq8）+ 副缓存 _rfa_full_hidden.f16（145,201 条，mmap）
训练      batch 8 / token-budget 1200 / 20,480 个专家 Linear 装载约 520~600 s
          稳态 19.6 ~ 21.4 s/步
```

### 16.2 判据链：gap 单调放大（这是本轮目前最硬的正面结果）

早停监视 val 上 A 臂（真码）CE，C 臂（码置零）给出先验基线，**健康信号是 gap = CE_C − CE_A 在扩大**：

```
step     CE_A            CE_C      gap
  750    0.6740±0.0440   0.7279   +0.0538
 1500    0.5131±0.0498   0.7568   +0.2437
 2250    0.3970±0.0591   0.7420   +0.3450
 3000    0.3046±0.0556   0.8599   +0.5554
 3750    0.2178±0.0530   0.8863   +0.6685
```

**五次 eval 全部刷新最优，gap 从 0.054 单调涨到 0.669（约 12 个标准误）。**
喂真码时 loss 持续下掉、喂零码时不下掉 —— 码信道在工作，而且越来越起作用。

注意早期那个"CE 卡在 0.80"的担忧是**采样窗口造成的误判**：前 400 步确实平，
但 3000 步时已到 0.30、4500 步时到 0.05。

### 16.3 step 3750 的 5 臂消融：一个自相矛盾，以及怎么读

```
[eval@3750] n=317  vszero 52.7%(m+0.558)  vsrandom 48.9%(m+0.586)
                   vsconst 47.9%(m+0.832)  vsperm 46.4%(m+0.811)
|Δ| 均值 0.8320
```

（`mean` = 真码 logp − 该臂 logp，正 = 真码更好；`frac_neg` = 该臂反而更好的 token 占比）

**两个仪器打架**：CE 那一侧说码极重要（gap +0.669），而 `frac_neg` 四个臂全在 46–53%，
接近抛硬币。按 `50_` 里写死的规则（`零表/置换 < 45%`）会判 ❌，**但那是假阴性**。

**真正的证据是均值的排序**：

```
零表   +0.558     <- 把码全置零，伤害最小
随机表 +0.586
置换   +0.811     <- 打乱「码 <-> 内容」的对应，伤害大
常数码 +0.832     <- 所有位置同一个码，伤害最大
```

若模型只用"存在感"（只感知 RNA 位置有个向量、不看内容），排序应该**反过来**：
置零抹掉存在感，应该伤得最重；置换和常数码保留存在感，应该伤得轻。
实测正好相反。**只有"模型在读码的身份"能解释** —— 而且"错误但看似合理的身份"
（置换/常数）比"没有身份"（置零）**更有害**，因为它让模型自信地读出错的答案，
而置零只让它退回语言先验、靠模板化的短答案有时还能懵对。**误导 > 无信号。**

均值不被稀释，所以它是比 `frac_neg` 更可靠的证据。

### 16.4 三个仪器修正（都是这轮踩出来的）

**1）`frac_neg` 被样板 token 稀释 —— 已修，改用「只看答案」统计**

实测 `build_batch` 的标签 span 平均只有 **9.89 个 token**：

```
no_think 尾巴（固定样板）   4      40.4%
" A: " + eos（固定样板）   ~3.08   31.1%
答案                      2.81    28.4%
```

样板与码无关、天然 50/50，把 `frac_neg` 拖向 50%。稀释下限约 36%，
所以 `45%` 那条线在 seqtask 上**永远通不过**。

新增 `answer_token_mask(tk, qa, lab)`：**用 offset mapping 精确定位答案 token**，
不能靠长度相减（BPE 在边界会合并，实测 `'AUGCAUGCAUGC'` 掩码 5 个而 `enc(答案)` 是 6 个、
`'UUUU'` 掩码 3 个而 `enc(答案)` 是 2 个）。`per_token_logp` 返回值从 2 个变 3 个，
`run_ablation` 每臂同时给「全部标签」和「只看答案」两套，**两行都打印**。
判据改用 `frac_neg_ans`。单测 `_scratch/_test_answer_mask.py`（把掩码选中的 token 解回字符串比对）。

**2）loss 等值性验证的判据：单步 → 累积中位数**

`answer_only_loss`（只在有标签位置算 logits）与原 HF loss 数学等价
（配置 `output_router_logits=False`，没有 MoE aux loss；`fixed_cross_entropy` 不带
`num_items_in_batch` 时就是 `reduction="mean"`，和手写一致）。

但两条路的**浮点路径**不同：走 `body(...)` 和走 `model.llm(...)` 的注意力掩码构造与算子选择不同。
**实测这台机器上同输入两次前向的 loss 能差到 3.87e-02**（step 4202 实测），
所以单步判据必然误杀。改成累积 `--verify-loss` 步后按**中位数**判，
阈值 `max(6 x 中位数抖动, 2e-3)` —— 错位是确定性的、每步都偏，中位数会很大；算子抖动是偶发的。

实测汇总：`|手工-HF| 中位数 1.72e-03` 对 `|HF重复差| 中位数 3.80e-03` → 等价。
**两个中位数同量级，这就是"算子差异"而非"接错了"的直接证据。**

**3）`@torch.no_grad()` 被字符串替换带走（2026-10-13，死在第 4500 步的消融上）**

用 `def per_token_logp(` 当锚点插新函数，把上面那行装饰器留给了新函数。
于是 `model.rna_emb.weight.copy_(real_table)` 在需要梯度的 leaf 上原地操作，
报 `a leaf Variable that requires grad is being used in an in-place operation`。
**训练路径完全正常，只在跑到消融那一步才炸** —— 又一次延迟失效。

教训：改完只跑 `py_compile` 不够，还要**确认被装饰的函数装饰器还在**。
本会话共两次字符串替换事故（另一次是把两行接在一起造成语法错误）。

### 16.5 内存账的更新（修正 §15.2 的两处）

**有效的那一条：只在答案位置算 logits。** 实测：

```
                   改之前        改之后
GPU shared        28.04 GB  ->   0.12 GB
GPU local         88.53 GB  ->  59.62 GB
训练进程私有       36.79 GB  ->  11.39 GB
物理空闲           ~9.5 GB  ->  46.0 GB
提交可用           29.1 GB  ->  40.2 GB
```

**步速没有变快**（21.4 对 20.9 s/步，在噪声里）。事后算也对得上：省掉的 logits matmul
约 9.8 TFLOP（约 1 s）+ `log_softmax` 2.4e9 元素（约 0.2 s）≈ 3%。
**这笔交易是拿 3% 的速度换 27 GB 显存。**

**无效的那一条：`expandable_segments`。** ROCm 上不支持：

```
UserWarning: expandable_segments not supported on this platform (HIPAllocatorConfig.h:40)
```

设了被静默忽略。**§15.2 里把它算成一项节省是错的** —— 那 27 GB 全部来自答案位置 logits
这一条。"降碎片"这个杠杆在这台机器上暂时没有。

**commit 那个数会骗人。** 页面文件是动态的（实测 85.4 GB → 65.7 GB 自己缩），
所以"提交上限"是会移动的天花板，"提交可用"量的是到那面墙的距离、不是剩余容量。
看门狗按它报 ALARM 叫了一整天，是假阳性。**可信的是页面文件当前用量**（实测只有 3.6 GB）。

### 16.6 运维（这一轮建起来的）

```
checkpoint   原子落盘（写 *.pt.tmp -> 验 zip 中央目录 -> os.replace）+ 动态时机
             （--save-every 300 或 --save-minutes 30，谁先到算谁）+ 只留最近 10 个
             单例 171.3 MB（6104 zip 条目 / 1520 适配器张量 = 190 x K=4 x (A,B) / 1524 组优化器状态）
             best.pt 与 final.pt 不参与轮转
看门狗       scripts/95_mem_watchdog.ps1，60 秒一次，WARN 15 GB / ALARM 8 GB，
             带 15 次采样的重启宽限（否则一重启就自杀）
             不自动杀训练 —— checkpoint 是原子的且每 30 分钟一个，最坏只丢 ≤30 分钟
启动脚本     scripts/run_D_qwen36.ps1，-Preset / -AblationEvery / -Resume / -SaveMinutes / -VerifyLoss
             续训时会从 checkpoint 的 args 里读 preset 并比对，不一致就大声报警
             （中途换 preset 是静默变味：适配器接得上、新增的从零开始、优化器被重置）
```

### 16.7 还没拿到的

- **step 4750 的「只看答案」消融**（`--ablation-every` 临时调到 250 以尽早拿到）。
  置换臂在答案 token 上的 `frac_neg` 才是判据。
- **三臂精确匹配**（`76_acc_by_source.py` / `87_channel_by_source.py`）——「答对了多少」。
  Δlogp 的均值只说明"码被用了"，不说明"答对了"。
- **SOS vs LoRA 的对照**。8B 上那个 A/B 从来没跑过，35B 上也没有对照臂，
  所以**目前还没有任何证据表明 SOS 比同 rank 的普通 LoRA 好**。
  唯一确定的是参数量严格相等（8.13 M vs 8.13 M），所以对照是干净的。

---

## 17. 部署形态：RNAcentral（后续计划，先记下来）

初定目标是 RNAcentral 规模的库。要在库环境里落地，设计会整体变一次 —— 而且方向是**变简单**。

### 17.1 核心结论：码可以离线预计算，在线路径不需要 ERNIE-RNA

昂贵的部分按 RNA 算，不按提问算；而 RNA 是静态数据。

```
离线（每个 RNA 一次，永久有效）
  RNA -> ERNIE-RNA -> 4x 压缩 -> FSQ -> 码 id 序列        存进库

在线（每次提问）
  按 rna_id 取码 id -> 查码表 -> 替换 <RNA> 占位的 embedding -> 生成
```

**在线路径里没有 ERNIE-RNA**，也没有 FSQ 编码器。这消掉一整个模型、一套依赖、
以及每次请求的前处理延迟。对数据库部署这是决定性的。

### 17.2 体量：该存的产物是码，不是隐藏态

```
一个码 = 12 bit（latent 12, levels 2 -> 4096）
pool=4 -> 1 码 / 4 nt
100 nt 的 RNA -> 25 码 -> 300 bit ≈ 38 字节
码表本体 4096 x 2048 float16 = 16.8 MB
```

对照：**副缓存的隐藏态是 145,201 条占 73 GB（约 500 KB/条）。差三个数量级。**
全库按三千万条算，码的总量约 1.5 GB，可以整个放进内存。

所以预计算作业的产物格式应该是 `rna_id -> int16 码序列` + 一个版本戳。

### 17.3 版本耦合（最容易出人命的地方）

预计算的前提是**码的语义不随模型变**，而 FSQ 编码器**是可训练的**。
实测它几乎不动（`|grad_fsq|` 0.03~0.3，而码表漂移 110%+），
所以实际上已经接近一个确定性函数 —— 但**只要它动一次，全库预计算的码全部作废**，
而重算要拿 ERNIE-RNA 跑满三千万条序列，是小时到天级的作业。

```
(encoder_version, code_table_version, adapter_version, base_version) 必须原子
```

任何不匹配都会**静默降级** —— 不报错、不崩溃，只是答得变差。

**结论：跑全库预计算之前必须把编码器冻住。** 数据说它本来就几乎不动，所以冻住代价极小；
如果 freetext 那轮需要它动，就在那之后再冻一次、再重算一次。

### 17.4 路由：查表 vs 模型（这条会影响怎么定位产品）

**在数据库环境里，seqtask 那 10 个模板的答案查表就能得到，而且精确、可验证、可追溯。**
所以**seqtask 上再高的分数也不能证明模型在库环境里有价值** —— 那些问题本就不该问模型。

模型真正有用的地方，是**答案不在库里、需要把序列和语言先验结合起来推**的问题。所以 harness 要有路由：

```
答案在库里 / 可由序列直接算  ->  服务器端算，精确返回（不经过模型）
需要跨信息推理              ->  走模型
```

这也直接指出下一轮该造什么训练数据：**不是 seqtask 的模板，而是"查表查不到"的那类问题**。
和 freetext 是同一个方向。

### 17.5 把三臂对照做成每次回答的「信道归因分」

我们本来就在算「真码 vs 零码」的差。在库规模下这个差可以直接当产品指标：

```
差大   -> 答案确实在读这个 RNA
差≈0  -> 答案来自语言先验，跟这条 RNA 没关系，应该打标记
```

代价是每次多一次前向（零码臂），可以做成抽样 QC（比如 1% 流量）或按需开启的 verified 模式。

价值在于**把可证伪性从实验属性变成运行时属性**。这台机器上同输入两次前向能差 3.9e-02，
所以"随手测一下"根本不可靠，必须有内建对照。而对数据库产品来说，
"这个答案是读出来的还是编出来的"是用户最该知道、又最难知道的事。

### 17.6 模板必须单一来源（现在就该做，不依赖判决）

训练侧现在是在 `build_batch` 里**内联**拼提示词，harness 一定会再实现一遍。
只要有一处不同（`no_think` 加不加、空格落点、`<RNA>` 前后标点、assistant 头的处理），
产品就会**静默降级** —— 推理时根本没有 loss 能暴露它。

而**这件事已经踩过一次**：`build_batch` 的注释写"labels 从 assistant 头开始
（含 `<|im_end|><|im_start|>assistant\n`）"，实测标签 span 却是
`len(assistant_ids) + len(enc(a))`，**漏了 `len(no_think)`**，所以 assistant 头那 4 个 token
根本不在标签里、标签从 `no_think` 的后半截开始。（功能上不一定要改，eos 在 `enc(a)` 里，
"学会停"这条没丢；但注释与实现不符，而且它决定了 `frac_neg` 池子里的成分。）

**动作：把提示词构造抽成一个函数，训练和 harness 都调它**，再加一个
"训练构造 == 服务构造"的逐 token 断言测试。这是唯一能让"训出来的东西"和
"部署出去的东西"是同一个东西的机制。

### 17.7 前置条件（顺序不能反）

```
1  拿到 step 4750 的「只看答案」消融  ->  码到底有没有被读
2  拿到三臂精确匹配                  ->  答对了多少
3  确认 SOS 相比同 rank LoRA 有没有收益（对照臂）
4  把提示词构造抽成单一来源 + 断言测试   <- 这条可以先做，不依赖前面
5  冻编码器 -> 全库预计算 -> 服务 harness + 归因分
```

**在第 1、2 步有结果之前不造 harness。** 如果置换臂最终在 50% 附近，
那时候造出来的会是一个"会流畅地编答案"的服务 —— 工程押在一个还没验证的信道上。

---

## 18. 2026-10-14 装载崩在 commit 额度上（根因 + 修法）

**症状**：装载 35B 时**没有 Python 栈**的崩溃，两种面貌交替出现：

```
76d~76g      停在 layer 35 附近   Exception Code: 0xC0000005
                （栈底是 torch_cpu 的 _local_scalar_dense_cpu，看着像张量悬空）
76_ 冒烟      停在 layer 11        OSError 1455 页面文件太小
84_probe      停在 layer 26/28     被 min_free 守卫主动中止
```

一天里换了四种姿势崩，所以先怀疑过"某个张量坏了"。**不是。**

### 18.1 根因一：26 个分片句柄 = 67 GB 的 commit（已修）

`scripts/_scratch/_probe_commit.py` 实测（这台机器 commit 额度 = 物理 63.6 + 固定页面文件 32 = 95.6 GB）：

```
0 基线                          commit 已用 56.8 / 95.6 GB
1 mmap 68 GB 副缓存（numpy）       57.8  (+1.0)   <- 只算 1 GB
3 开  5/26 个 safetensors 句柄     72.4  (+14.3)  <- 按文件大小算
3 开 10/26 个句柄                  85.5  (+13.1)
3 开 15/26 个句柄                  92.6  (+7.1)
-> 第 16 个句柄就 OSError 1455（与装载时同一个错）
```

**safetensors 的 `safe_open` 在这台 Windows 上按私有映射计 commit，值约等于文件大小。**
而装载器（`ShardReader`）为了 `assign=True` 的惰性视图，把**所有**用过的分片句柄都留着 ——
26 个分片 67 GB，正好把额度吃光。`scripts/_scratch/_probe_shard_view.py` 证实
`get_tensor` 给的是视图（同一 key 取两次 `data_ptr` 相同、commit 增量 0），
所以句柄确实"不能随手关"，但 `_probe_shard_close.py` 证实**丢引用 + gc 能还回来**
（5 个分片 71.5 -> 57.1 GB）。

**修法**（`src/rnaqwen/qwen36.py`，训练与评测共用这一份）：

1. `ShardReader` 记住每个分片覆盖的层范围，`trim(done_layer)` 在
   `text.layers[i].to(dev)` **之后**关掉层范围已走完的分片；含非层内张量
   （embed / norm / lm_head）的分片钉住不动。
2. `get()` 一律返回 `clone()` 的私有副本。多花一次 1.7 GB/layer 的拷贝，
   换掉"视图活过 trim = use-after-free"这条只在几十层后炸、没有任何栈的隐患。
3. 每层的日志行加 `commit 已用/额度` 和 `句柄 N` 两列；`stats` 里加 `shard_peak`。

实测效果（`reports/76_smoke3.log`）：**句柄峰值 26 -> 4**，装载从 layer 11 一路走到 layer 38。

### 18.2 根因二：权重进显存那 18 GB 也算 commit（待重启生效）

修完句柄之后 commit 不再被浪费，但**每层仍然涨 0.45 GB**，涨的就是 int4 权重搬进
APU 统一显存那部分：

```
layer 20   commit 86.4/95.6
layer 30   commit 91.0/95.6
layer 38   commit 92.1/95.6   -> 第 39 层 0xC0000005
```

装载自己需要约 38 GB，机器剩下约 39 GB，差的就是收尾那 2 GB（embed + lm_head）和余量。
**这是硬账，没有代码可省。** 已把 C: 页面文件从 32 GB 改成 64 GB
（`scripts/_scratch/_fix_pagefile.ps1`，需要管理员；设置已写入，等重启生效，
额度会到约 128 GB）。

### 18.3 教训（补进方法论）

- **这台机器上要盯的是 commit，不是物理空闲。** 物理报"空闲 36 GiB"的时候
  commit 已经到 92/95.6 了。`free_gb()` 量不到这一层，两个都要看。
- **原生崩溃（0xC0000005）没有 Python 栈，别去猜是哪一行。** 本次靠
  `RNAQWEN_TRACE_GET=1`（`ShardReader.get` 逐张量打印）拿到"最后打印的是谁"，
  再靠三个探针把"是句柄还是张量"分开。带着 34 ~ 500 个 torch frame 的栈
  **只说明崩在 native 里，不说明崩在谁头上**（本次栈底指 safetensors，
  实际是内存额度）。
- **PS 1.5 的 `.ps1` 必须是 ASCII。** Windows PowerShell 5.1 把无 BOM 的脚本
  按 GBK 读，中文注释会让解析器报 `ExpectedExpression` —— 这次白折腾了一轮提权。



---

## 19. D 臂收官与解码路径的真 bug（2026-10-15）

### 19.1 训练结束：早停，不是跑满

`D_qwen36_sos` 在 05:17 早停结束，用时 935 min。判据是
「CE_A 连续 4 个 eval 点（3000 步）改善 < max(0.5%×CE, 1 se) = 0.0459」。

```
step   6750  CE_A 0.1601  CE_C 1.2179  gap +1.0577
step   7500  CE_A 0.0891  CE_C 1.1881  gap +1.0990   <- best
step   8250  CE_A 0.1762  CE_C 0.9924  gap +0.8162
step   9000  CE_A 0.0823  CE_C 0.8937  gap +0.8114
step   9750  CE_A 0.1064  CE_C 0.8708  gap +0.7643
step  10500  CE_A 0.0589  CE_C 0.9261  gap +0.8672
```

**一个要重看的地方**：step 10500 的原始 CE_A（0.0589）是全程最低，但它相对
0.0891 只改善 0.0302，没过 1 se = 0.0459 那一关，被当成噪声拒掉。
「哪个点最好」取决于你信原始数值还是信带 se 的判据。**落盘的 `final.pt` 是 step 7500。**

终局 5 臂消融（只看答案 token 的 frac_neg）：零表 8.3% / 随机表 3.3% /
常数码 8.3% / 置换 3.3%，全部远低于 45% 那条线 —— 码确实被读了。

### 19.2 解码路径的 Conv1d 从来没打补丁（这是 10-14 那次 0.0% 的真因）

`patch_conv1d` 只换了预填充用的 `self.conv1d`。生成走的是**另一条路**：

```
Qwen3_5MoeGatedDeltaNet.forward
  -> self.causal_conv1d_update(...)      # modeling_qwen3_5_moe.py:467
  -> torch_causal_conv1d_update(...)     # modeling_qwen3_5_moe.py:230
  -> F.conv1d  ->  MIOpen: miopenStatusInternalError
```

`self.causal_conv1d_update` 是在 `__init__` 里按**模块名**取的
（`causal_conv1d_update or torch_causal_conv1d_update`），FLA 没装、走 torch 回退。
而**训练只做预填充**，这条路径一次都没跑到 —— 直到 10-14 那次评测：
120 条生成全变成 `<ERR RuntimeError>`，准确率 0.0%，看着像仪器坏了。

修法：在建模型之前把 `M.torch_causal_conv1d_update` 换成同一套移位乘加（fp32 累加），
与 `F.conv1d(padding=0, groups=C)` 对齐。对拍脚本
`scripts/_scratch/_test_conv1d_update.py`：`max|d_out| 9.5e-07`，conv_state 更新逐位相同。

**教训：训练不跑的代码路径，评测第一次跑才会暴露。** 生成之前没有任何 loss 能提示
这件事 —— 与 §17.6 的「模板必须单一来源」是同一类失效：静默，只在部署侧炸。

### 19.3 内存：页面文件生效后的实测

页面文件 64 GB 生效，额度 127.6 GB。当天三次装载的 commit 峰值：

| 场景 | 峰值 commit | 结果 |
|---|---|---|
| 训练续跑（50_ `-Resume`） | 89.8 / 127.6 | 通过 |
| 全量评测（76_ `--per-source 30`） | 97.3 / 127.6 | 通过 |

句柄峰值 4 个（trim 生效）。旧额度 95.6 GB 正好卡在这两次峰值之下，32 -> 64 GB 这一改是决定性的。

### 19.4 新增：RNAQwen-preview 打包路径

`src/rnaqwen/preview.py`（量化权重的存取）+ `scripts/97_export_preview.py`（合并 + 打包 + 回装校验）。
机制全部实测过，不是推理：

- **bnb 0.50.2 的量化状态是张量。** `Linear4bit.state_dict()` 出 6 个键
  （`weight` / `absmax` / `quant_map` / `nested_absmax` / `nested_quant_map` /
  `quant_state.bitsandbytes__nf4`），所以 safetensors 能完整存取。
- **回装必须走 `Params4bit.from_prequantized`**；`Linear4bit.load_state_dict` 会直接报
  "Unexpected key(s)"。256x256 探针：`max|diff| == 0.0`。
- **一个模块的 6 个键必须落在同一个分片**，由 `group_keys()` 保证。
- SOS 的目标（注意力 + shared_expert）在 35B 上是 **bf16 nn.Linear**（只有专家被量化），
  所以 `merge_sos_layers` 是纯 `copy_`、无损。脚本里显式断言这一点 —— 否则合并会静默写坏 Linear4bit。
- 导出脚本自带四道闸：适配器覆盖集合、合并安全性、合并前后 logits 一致、**回装后 logits 逐位相同**。

**为什么值得做**：preview 包存的是已经量化好的权重（约 23 GB），装载一次读完，没有 26 个
bf16 分片的映射、也没有逐层现量化 —— 卸载峰值 commit 从 97 GB 降到几十 GB。
§18 那堵墙在部署侧就此消失。

**状态**：代码与单元级检查（分组、conv1d 对拍）通过，**35B 上的整包导出尚未跑**
（要先等评测把内存让出来）。

## 20. GGUF 分发路径（2026-10-15）

§19.4 的 preview 包是给本仓库 HF 装载路径用的；GGUF 是给 llama.cpp 生态用的：
单文件、CPU 可跑、端侧能落地。`scripts/98_export_gguf.py`，六个阶段
`patch / convert / assets / quantize / verify / smoke`。

### 20.1 三个卡点

1. **适配器没有槽位。** GGUF 里塞不进 LoRA，必须先把 SOS 合并进底座。
   合并漂移是可信数（`runs/RNAQwen-preview/merge_drift.json`）：
   `max|dlogit| 2.32`，相对量级 `9.05e-2`，128 个探针里 top-1 翻了 4 位。
2. **码信道必须变成真 token。** 训练时 `E2EModel.embed()` 直接把 `rna_emb[c]`
   写进输入序列，token id 248320 只是个占位符、词表里没有信息。GGUF 只能从
   词表取向量，所以给每个码补一个真 token：id = `248320 + c`，文本形式
   `<R{c}>`，共 4096 个，把 `rna_emb[c]` 烤进该 token 的输入嵌入行
   （bf16 取整，和训练时 `rna_vecs.to(e.dtype)` 一致）。
3. **248070..248319 是空档。** 转换器给这些 id 填 `[PADi]`。最终词表
   `248320 + 4096 = 252416`。

**输出行必须是常数 0，不能是 -100。** 码 token 的输出行设成常数 c 时，
logit = c * sum(h)。`sum(h) < 0` 时 -100 会翻成一个极大的正 logit，模型对着
纯文本提示会吐一串 `<R...>`（smoke 里真发生过）。改成 0 之后 logit 恒为 0，
并且 verify 里加了闸门守着这一条。

### 20.2 前端进不了 GGUF

ERNIE-RNA 编码器 + FSQ + 码表留在仓库，和 GGUF 一起分发：
`gguf/fsq.pt`、`code_emb.pt`、`tokenizer.pt`、`code_tokens.json`、
`rna_prompt.py`。后者负责 序列 -> 码 -> token id，也是「GGUF 看到的就是训练喂的
向量」这句话的唯一保证。序列先过大模型那条路（§17.1）仍然要求前端在场。

### 20.3 产物

| 文件 | 体积 | 用途 |
|---|---|---|
| `gguf/RNAQwen-preview-Q8_0.gguf` | 35.22 GiB | 近无损对照 |
| `gguf/RNAQwen-preview-Q4_K_M.gguf` | 20.23 GiB | 可分发成品 |

`--phase verify` 四道闸全绿：词表 252416 == 嵌入行数；码 token 输入嵌入
`max|d| 1.043e-02`（码表自身离散度 1.647e-02）；码 token 输出行 `max|v| 0`；
`blk.3/7.attn_q` 的 `均值|gguf-合并| < 均值|gguf-底座|`（合并确实进了盘）。

`--phase smoke` / E2E：Q4_K_M + `llama-server -ngl 0 -t 12`（纯 CPU），
`scripts/_scratch/_e2e_gguf.py` 走 `gguf/rna_prompt.py` 造 prompt，
first-4 / first-6 各两条，**4/4 命中**。

### 20.4 词表被转换器重新编号（2026-10-15 复盘，这一条最值得记）

第一版 GGUF 的 E2E 是 4/4 通过的 —— token **id** 是对的（`248320 + c` 那 4096 行
确实装着 `rna_emb[c]`）。但**token 的文本和 id 差了 243 位**：

```
AutoTokenizer.from_pretrained(底座).vocab   -> 只命名 248077 行（嵌入矩阵 248320 行）
我们把 <R0> 声明在 248320
实测 detokenize(248320) == <R243>   detokenize(248077) == <R0>
```

**根因**：`conversion/base.py: get_vocab_base()` 是 `for i in range(vocab_size)` +
`reverse_vocab[i]`，而 `reverse_vocab` 来自 `AutoTokenizer.from_pretrained(dir).vocab`。
**声明的 id 只要和"最后一个有名有姓的 id"不连续，就会被重排成 `max(id)+1`。**
底座自己只命名 248077 行，中间空着 248077..248319 这 243 行，于是我们的 4096 个码
被整体前移 243 位。

后果分两半，一半是坏的：

- 走 id（`rna_prompt.py`）**完全正常** —— 所以 E2E 一次就 4/4，看不出问题。
- 走文本（`<R{c}>` 当普通 token 打进去）**错 243 位**，拿到的是底座那 243 行死权重。
- 顺带一个泄漏：id 248077..248319 那时挂着 `<R0>..<R242>` 的名字，而它们的**输出行
  没被置成 sentinel**，理论上可以被采样成怪 token。

**修法（已实测）**：底座那 243 行谁也命不了名（tokenizer 根本产生不了这些 id），所以
把它们**输入置 0、输出也置 0**，再补 243 个 `<|pad{i}|>` filler token 占住 id，
让码稳稳落在 `248320 + c`。`build_src()` 里加了一道闸门，直接问转换器要问的那个
tokenizer：

```python
av = AutoTokenizer.from_pretrained(str(src)).vocab
got = {c: av.get(f"<R{c}>") for c in (0, 1, man["n_codes"] - 1)}
assert got == {c: man["base_vocab"] + c for c in got}     # 动了就不转换
```

干跑脚本 `scripts/_scratch/_tokid_dryrun.py` 把两种排布都验过：
带 filler 的 `<R0>` -> 248320（取这个），不带的 -> 248077。

**修完的四道闸（Q4_K_M，753 张量）**

```
词表 252416 == 248320 + 4096
码 token 输入嵌入 max|d| 1.043e-02  均值 1.205e-03（码表自身离散度 1.647e-02）
码 + filler 输出行 max|v| 0.000e+00
blk.3/7.attn_q  均值|gguf-合并| 1.00e-03 / 9.42e-04  <  均值|gguf-底座| 2.64e-03 / 2.31e-03
```

E2E 除了 4/4 命中，还多了一条**文本形式**对齐检查：

```
[tok] 文本形式 <R3591><R3798>... -> 23 id，与码信道路径一致
      头 [251911, 252118, 251830, 252383]      (251911 - 248320 == 3591)
```

**教训**：这次是"id 对、名字错"，所以**端到端能跑通**是最没有说服力的证据 ——
它只验证了其中一条路。名字和 id 对不上的时候，必须**两个方向都测**。

### 20.5 零码臂到底是不是「问题先验」？（2026-10-15）

论文草稿里原来写着：零掉码信道之后，模型给出的那个固定答案**就是问题先验**。
这句话必须验，因为它是「C 臂也不是噪声」那条论证的支点。三件事全都能离线查，
不用再跑 35B：

```powershell
& C:\ana\envs\comfyui\python.exe scripts/_scratch/_prior_forensics.py
# -> reports/prior_forensics.json, reports/prior_forensics.log
```

**先验地板**（训练集里每个模板的众数答案，拿去打该语料自己的 test split）：

```
10 个模板合计 4.26%  (7,225 / 169,410)
```

**C 臂的常数是不是众数**（每模板 30 条，n=30）：

```
模板                          C 臂常数     占比   语料众数      是众数吗
first 4                       UUAU        77%    UACG          NO
first 6                       UUUU        43%    UACGUA        NO
first 8                       UUU         53%    UACGUAGG      NO
last 4                        UUUU        77%    ACAG          NO
last 6                        CUUUUU      77%    AAACAG        NO
last 8                        UUUUUUAU    57%    GCAAACAG      NO
positions 1-6                 UUUUUA      57%    UACGUA        NO
dinucleotide 1-2              UU          83%    UA            NO
RC first 4                    AUUA        70%    CGUA          NO
RC first 6                    CUUUUA      63%    UACGUA        NO
```

10 个模板**一个都不是众数**，全是 U 串（最多夹一个 A/C）。

**A 臂里有多少是先验分**（同一 300 条生成结果，按 gold 是否等于众数切开）：

```
 arm   slot=众数   gold=众数   那 26 条 prefix   gold!=众数   其中 exact  prefix   还在吐众数
   A       23          26           88.5%            274        12.0%    68.2%        0
   B       23          26            0.0%            274         0.0%     2.2%       23
   C        0          26            0.0%            274         0.4%     0.4%        0
```

**结论（三句话）**

1. 零码**不是**回到问题先验，是塌进一个比先验**更差**的常数：先验值 4.26%
   （这条 300 采样上是 8.7%，因为 sampler 把众数答案抽多了：26 条 vs 期望 13 条），
   C 臂只有 0.3%（1/300）。
2. 先验这条线真正卡住的是 A 臂：exact 11.0% 只比同采样的先验 8.7% 高 2.3pp，
   n=300 下**区分不开** —— 差的那部分全被「不停机」吃掉了。
3. 换个尺子就分开了：在**众数根本答不对**的那 274 条上，A 臂 prefix 68.2%、
   exact 12.0%，而且从不吐众数（0/274）；同样 274 条，B 臂 2.2%、C 臂 0.4%。
   也就是说 A 臂在难子集上拿到的分**全部来自码信道**。

**待办**：decoder 加 stop 条件后重生成（现在 35B 一条约 40 s），把 exact 那把尺子
救回来，否则论文里 A-vs-先验 的对比在 exact 口径下站不住。

### 20.6 前端把序列截到 512 nt（2026-10-15，改在 1024）

`gguf/rna_prompt.py` 的 `max_nt` 默认是 **512**，而训练（47_/48_/50_）用的是
`MAX_NT = 1024`，语料本身是 ≤1022 采的。curated 那 4,481 条里 **1,845 条（41%）
超过 512**。

**这个 bug 是「测不出来」的那种**：`_e2e_gguf.py` 按长度排序取了**最短的 4 条**
（都 ≤120 nt），而且判命中的是**子串**（`gold in txt.upper()`），所以模型输出
` A: CUGG` 也算命中。测试自己挑了最有利的输入。

**码本身没问题**：`scripts/_scratch/_codes_equiv.py` 拿缓存隐藏态（fp32，
`_fsq_full_pos.pt`）算的码和前端用 ERNIE 重算的码对比，600 条序列 / 71,111 个码位
**0 翻转**。所以问题只在长度，不在数值。

**配对复跑（同一批 120 条 prompt，只换 cap）**：

```
                                  条数   512cap   1024cap
总体                              120    54.2%    55.8%
  nt < 200                         60    65.0%    65.0%
  200-512                          10    80.0%    80.0%
  512-800                          20    30.0%    45.0%
  >800                             30    40.0%    36.7%
真截断组 (codes==128, nt>512)      50    36.0%    40.0%
未截断组                           70    67.1%    67.1%
```

**确定性对照**：70 条 prompt 完全没变的条目里，69 条（98.6%）两轮输出逐字相同，
exact 也是 67.1% → 67.1%。decoder 基本确定，两轮可比。

**踩过的坑（必须记）**：第一次分析用 `codes*4 < nt` 当截断标志，这个条件在
`nt % 4 != 0` 时也成立（FSQ 每 4 个位置出一个码，余数丢掉），于是分组退化成
「3 条 RNA vs 9 条 RNA」的假对比，还给出 83.3% vs 44.4% 这种漂亮数。真实原因是
**每个模板的抽样位置相同 → 10 个模板共用同样 12 条 RNA**：条目数 120 看着多，
RNA 层面的 n 只有 12。以后按 RNA 抽。

改法：`rna_prompt.py` 的 `MAX_NT = 1024`（加注释说明必须跟训练对齐），
`serve/server.py` 的 `MAX_NT` 一起改成 1024。**cap 该改，但只值 +1.6pp**；
长序列（>512 nt）本身难得多，那是能力问题，不是这条 bug。

### 20.7 网关服务（2026-10-15）

分发出去的 GGUF 不能自己算码，所以前面加了一层网关：`serve/server.py`（FastAPI）
+ `serve/static/index.html`（单文件工具页）。

```
启动  & C:\ana\envs\comfyui\python.exe -u serve/server.py --port 8100
页面  http://127.0.0.1:8100/
后端  --backend 默认 http://127.0.0.1:8099（llama-server + RNAQwen-preview-Q4_K_M）
```

| 端点 | 作用 |
|---|---|
| `GET /health` | 本服务 RNA 侧 + 后端状态（RNA 侧 22 s 载入，页面每 5 s 轮询） |
| `GET /templates` | 训练用的那 10 个模板 |
| `POST /rna/ask` | `{sequence, question}` -> `{answer, codes, tokens, ms, stopped}` |
| `POST /v1/chat/completions` | OpenAI 形状；只有最后一个 user 消息里的 `<RNA>...</RNA>` 块会被翻译成码，其它原样转发 |

要点：

- 提示词拼法与 `76_acc_by_source.py` 的 `gen()` 逐 token 一致（`Q: <RNA>` + 码 token
  + `</RNA>{question}` + assistant + no_think），码 token 用 `code_tokens.json` 的
  `base_vocab + c`。
- 答案按评测口径剥掉 `A:` 前缀；请求带 `stop=["\n", "<|im_end|>", "<|im_start|>"]`。
- 实测延迟：短序列（40 nt / 10 码）约 4.8 s，1022 nt / 255 码约 11 s。
- 工具页截图：`reports/ui/desktop.png`、`reports/ui/mobile.png`
  （无头 Chrome 渲染，420 px 与 1400 px 两档都查过不会溢出）。

---

## 21. cont_rff 收官（2026-10-17）

### 21.1 训练与产物

```
起始  step_007913.pt（D 臂）
预算  -ExtraSteps 6000，绝对终点 13913，--min-steps 钉死 → 没有早停
结局  step 13913，python exited with code 0
产物  runs/e2e_cont_rff/ckpt/final.pt          186.5 MB
      runs/cont_rff/artifacts/sos_adapter.pt     21 MB
      runs/cont_rff/artifacts/{code_emb,fsq}.pt + meta.json
导出  gguf/RNAQwen-preview-1018-Q4_K_M.gguf   20.23 GB（Q8 已删腾盘）
```

### 21.2 终评（只看答案那一行）

| 臂 | 平均 logp 差 | 负值占比 |
|---|---|---|
| vsperm | 4.5780 | **3.3%** |
| vszero | 3.2177 | 3.3% |
| vsrandom | 3.7240 | 1.7% |
| vsconst | 6.0745 | 0.8% |

判据成立：码身份被读。**但这一版用的是 `--token-budget 512`**，和 8250–10500 那几档
（1024 记账）不可直接比，跨档比较必须在正文里标出来。

### 21.3 为什么降到 512

1024 时死在 step 11060 的 `total.backward()`，报 `bad allocation`，自报 `[mem]` 只剩
7.0 GiB 空闲物理内存。根因是 `--token-budget` 只累计压缩码长（clen），忽略问答文本
token：seqtask 答案 6–8 字符时无害，freetext 答案中位 278 字符、最大 551，把真实
B·T 顶到 2392（D 臂可存活上限 1656）。**改为硬上限 512 之后没有再崩。**

---

## 22. 免疫垂直方向：数据盘点（2026-10-18）

### 22.1 三个数据源

| 来源 | 内容 | 可用量 |
|---|---|---|
| GSE174374 | NSUN2-KO vs WT 的 dsRNA-seq + m5C 位点 | 11,550 个基因；541 个 m5C 位点（529 可用） |
| GSE247936 | MeRIP-seq peak | **39,662 m6A + 67,620 m5C = 107,282 个位点** |
| PDB | 免疫相关 RNA-蛋白复合物 | 116 个 mmCIF（38.7 MB）→ 104 条结构，62 双链 / 42 单链 |

**m5C 命名基因只有 27 个，和 dsRNA 数据重叠 12 个。** 这是免疫表型标签的真实瓶颈，
不是工程问题。

### 22.2 结构评测的基线（PDB，scripts/97_ + 98_）

对 42 条实测单链结构，ViennaRNA 预测：

```
488 个实测配对   precision 0.7982   recall 0.9242   F1 0.8566
最差 2B3J（TLR7 配体）F1 = 0
```

自由溶液结构与受体诱导结构可以完全不同 —— 这条要写进任何「结构预测」主张的限制里。

### 22.3 已完成的数据资产

```
data/processed/rfam_seqs.jsonl      145,201 条，116,070 条有 family 注释
data/processed/rfam_struct.jsonl     92,685 条，带 ViennaRNA 七项特征
data/processed/pdb_rna_struct.jsonl 104 条
```

---

## 23. 位点级任务：判据链的第三次翻车与修正（2026-10-18 → 10-19）

这一节是**方法论最重要的一节**。我们连续四次改任务定义，每次都是被自己的对照否掉。

### 23.1 A2：同染色体邻近负样本（`scripts/A2_site_task.py`）

```
immune_site_rna_mod.jsonl   104,662 行 = 52,331 对（m6A 52,426 / m5C 52,236）
负样本：同染色体，离 peak 200–2000 nt，窗口不压任何 peak
切分：染色体留出，test = chr20/21/22/X
```

| 设置 | m6A | m5C |
|---|---|---|
| 码通道 | 0.654 | 0.635 |
| + ViennaRNA | 0.654 | 0.633 |
| GC 单特征 | 0.582 | 0.591 |
| 4-mer 组成 | 0.593 | 0.601 |
| ViennaRNA 单独 | 0.608 | 0.605 |

**同 2 万条子集上的打乱对照**：

| | m6A | m5C |
|---|---|---|
| 真实序列 | 0.678 | 0.635 |
| 单核苷酸打乱 | 0.665 | 0.564 |
| Δ | **+0.013** | **+0.071** |

**结论一：m6A 的 0.68 是假的。** 打乱序列只掉 0.013（不到一个标准误），m6A 在这个
数据上基本是单核苷酸组成，任何「学到 m6A motif」的说法都不成立。

### 23.2 仪器翻车：vsperm 对袋装码向量失效

`--tag perm`（4096 码的随机双射）在 m6A 上给出 0.645，对比真实 0.652 —— **几乎不掉**。

原因：码向量取均值后，随机双射等价于对 4096 维码直方图做一次随机投影，2048 维下
信息基本保住。**vsperm 是生成式 LM 的仪器，不是袋装向量头的仪器。**

以后这一层一律用**序列打乱**做对照，`--tag shuffle`。

### 23.3 A3：同基因内负样本（`scripts/A3_site_within_gene.py`）

```
immune_site_m5c_within_gene.jsonl   47,143 对 / 26,965 基因
负样本：同基因、相对位置相近、不压 m5C peak
test 6,562
```

| 信号 | AUC |
|---|---|
| 基因内相对位置 | 0.506 |
| GC | 0.564 |
| 4-mer | 0.585 |
| 码通道 · 打乱 | 0.576 |
| **码通道 · 真实** | **0.605** |

配对 bootstrap（`scripts/A5_paired_bootstrap.py`，2000 次重采样，同一批行）：

```
real 0.602  shuffle 0.580  Δ +0.022  CI [+0.009, +0.036]  P=0.000
```

### 23.4 A4：同基因 + TSS 距离匹配（`scripts/A4_site_tss_matched.py`）

第一版有偏：`|dts|` 单独就有 0.566 的 AUC，因为靠近 TSS 的正样本找不到同侧负样本。
改成 `--min-dts 250` + 对称 `|dts|` 匹配后，偏倚降到 **0.4987**。

```
15,630 对，test 2,182
码通道 0.628  打乱 0.609  GC 0.566  4-mer 0.586
bootstrap: Δ +0.008  CI [-0.015, +0.030]  P=0.23
```

**不是「没效应」，是样本量不够**：test 只有 2,182，A3 是 6,562。

### 23.5 A6：跨基因、|dts| × GC 分箱匹配（`scripts/A6_site_binmatched.py`）

同基因负样本的天花板约 2 万对（m5C peak 成簇，簇内没有同位置的非 peak 位点）。
要功率就得放弃「同基因」，但保住另外两个混淆：**|dts| 对数分箱 + 101 nt 窗口 GC 分箱**。

```
immune_site_m5c_binmatched.jsonl  52,460 对 = 104,920 行，test 7,556
400k 候选池按 (|dts| bin, GC bin) 分箱，1:1 不复用
平衡检查：GC 均值 0.5217 vs 0.5215；同基因对 = 0
```

| 信号 | test AUC |
|---|---|
| GC 单特征 | 0.5008 |
| 4-mer 组成 | 0.5527 |
| 码通道 · 打乱 | 0.5103 |
| **码通道 · 真实** | **0.5685** |

```
bootstrap: Δ +0.058  CI [+0.041, +0.074]  P=0.000
```

**组成捷径被彻底堵死后，打乱序列掉回随机（0.510），真实序列还有 0.5685。**
这是整条免疫线上唯一一个功率充足、对照干净的正面结果。

### 23.6 A7：同容量 k-mer 基线 —— 把上面那条打回去（`scripts/A7_kmer_features.py`）

A6 里那个「4-mer 0.5527」是用逻辑回归跑的，和 2048 维线性头容量不对等。
换成**同一个头、同一批切分**重跑：

| 特征 | 维度 | test AUC |
|---|---|---|
| SOS 码通道 | 2048 | 0.5685 |
| **4-mer 计数** | **256** | **0.5663** |
| 5-mer 计数 | 1024 | 0.5534 |
| 6-mer 计数 | 4096 | 0.5494 |

```
配对 bootstrap（码通道 − 4-mer）: Δ +0.0057  CI [-0.009, +0.020]  P=0.23
```

**结论二：码通道没有超过 4-mer 词袋。**

5-mer/6-mer 反而更差（0.553 / 0.549）是过拟合，不是更多生物学。

### 23.7 这一节的净结论

能说的：**m5C 位点确实由序列排布决定，不只是 GC、也不只是邻域**（打乱掉 0.058）。
不能说的：**我们的码通道比一个 4-mer 词袋更强**（打平）。

「同容量基线」从此是硬规则：**任何「我们的表示更好」的主张，基线必须用同一个头、
同一批切分、同一套超参。** 容量不对等会让结论反向（+0.016 → +0.006）。

---

## 24. 基因级任务失败（2026-10-18）

`immune_gene_dsrna.jsonl`：3,932 行 = 1,966 对（GSE174374，rel > 0.20 为正，
|rel| ≤ 0.05 为负，严格 1:1 长度匹配，同染色体优先）。

混淆检查：

```
初始最近长度匹配   长度 AUC 0.676
严格长度匹配后     长度 0.525   位置 0.498   WT dsRNA 0.580
```

头的结果（test 564 个基因）：

| 读法 | gene AUC |
|---|---|
| SOS-LoRA 最后一层隐状态 | 0.536 |
| + 逐维标准化 | 0.536 |
| 只走码表向量（不用 35B） | **0.545** |
| GC 组成 | 0.542 |
| 基因长度 | 0.500（本来就按长度配对） |
| WT 基线 dsRNA | 0.418（反读 **0.582**） |

**两个结论**：

1. **35B 没有带来超过码表本身的信息**（0.536 vs 0.545），最后一层隐状态在基因间
   相关系数 0.997，几乎是个常数分量。
2. **最强单预测因子是 WT 基线 dsRNA（反读 0.582）** —— 这个标签很大程度上是
   「基础表达 / 动态范围」效应，不是局部序列决定的。

基因级这个定义（基因体 101 nt → 细胞级 dsRNA 表型）信号不够，转位点级是对的。

---

## 25. 结构探针的判据校准（2026-10-19）

在跑结构判别头之前，先把**组成基线**算出来（`scripts/63_kmer_struct_baseline.py`），
否则又会犯 A6 那种错。

| 目标 | GC | 长度 | GC+长度 | 4-mer | 4-mer+GC+长度 |
|---|---|---|---|---|---|
| `mfe_per_nt` | 0.641 | 0.259 | 0.670 | 0.687 | **0.719** |
| `paired_frac` | 0.128 | −0.004 | 0.129 | 0.237 | **0.232** |
| `longest_stem` | 0.023 | 0.534 | 0.536 | 0.156 | **0.551** |
| `acc_mean` | 0.227 | 0.348 | 0.363 | 0.469 | **0.527** |
| `mfe`（原始） | 0.323 | **0.929** | 0.955 | 0.340 | **0.955** |

三条判读：

1. **`mfe` 原始值 = 长度 + GC 的线性组合（0.955）**，以后只能作对照，不能当主指标。
2. **`mfe_per_nt` 的门槛是 0.719**，「比 0 高」不算数。
3. **`paired_frac` 是唯一干净的战场**：长度 −0.004、GC 0.128、4-mer 0.237。

判据随之改成：**每个目标各自对自己的最好组成基线，报 Δρ。**

结构判别头的设计（`scripts/61_struct_head.py`）：五个目标 × 四组特征
（ERNIE mean / 码表 mean / mean+std / mean+max+min）+ 四组基线，按 Rfam family 切分。

---

## 26. RNAcentral 全量 + 码表流式扩容（2026-10-19 → 10-31）

### 26.1 下载

```
文件  data/raw/rnacentral/rnacentral_active.fasta.gz
大小  11,858,076,614 字节（11.04 GiB），与服务器 Content-Length 一致
内容  48,053,536 条序列 / 41.34 Gnt
```

长度分布：

```
≤256 nt   17,132,913 条（2.14 Gnt）
≤512 nt   30,666,682 条（7.56 Gnt）
≤1024 nt  36,906,805 条（12.08 Gnt）
```

精确去重后仍是 **100.0%** —— 一个 URS 对应一个唯一序列，去重省不了东西。

**下载本身的坑（会重复踩）**：

- 多连接会被 EBI 掐死：16 路并行测出 57.7 KB/s 但随后全部挂成 0 字节；4 路、8 路同样
- 只能单连接断点续传，速度 38–400 KB/s 之间跳
- 共 7 轮 `curl -C -` 才拉完，耗时约 5 天
- 已注册 `HKCU\...\Run\RNAcentralDownload`，重启自动续

### 26.2 转 jsonl

`scripts/50_make_rnacentral_jsonl.py`：**只做 T→U + 大写**，不做二次过滤
（ERNIE-RNA 吃 N/R/Y 不报错，走 unknown token）。

```
data/processed/rnacentral_2m.jsonl   2,187,209 条 / 751 MB
```

### 26.3 流式码表训练（`scripts/49_stream_tokenizer.py`）

原 `48_stage1_full_tokenizer.py` 把全部位置 hidden 缓存到磁盘：4,481 条 → 2M 位置
→ 5.9 GB。218 万条会变成 **1.5 TB**，不可能。

流式版：编码一批 → 训几步 → 丢掉；每 8,000 行落一个分片（`<tag>_<dataset>_parts/partNNN.pt`），
重启时按已落盘行数跳过。

```
1063 个 chunk × 12 步，chunk = 2048 条，窗口 = 101 nt
```

**pilot（6,000 条、12 步）就已经把跨序列迁移 lift 打到 3.25x**，现有码表基线是 2.63x。

### 26.4 码表形状不变

```
latent 12 × levels 2 = 4096
enc.weight  (12, 768)      9,216 参数
rna_emb     (4096, 2048)   8.39M 参数
```

`fsq_full/tokenizer.pt`、`e2e_cont_rff/final.pt`、`fsq_stream/state_latest.pt` 三者形状一致，
`50_routeC_e2e.py` 的「只加载形状匹配的键」不会跳过任何东西。

**但码 ID 的语义变了**：旧 `c=1234` 和新 `c=1234` 指向不同内容，所以 `rna_emb` 不能沿用，
必须重跑 `--init-code-emb`（内容初始化，同 6-mer 余弦）。

---

## 27. 工程事实：并行、线程、崩溃（2026-10-31）

### 27.1 32 线程没有用（实测）

| | 8 线程 | 32 线程 |
|---|---|---|
| 速度 | 30.0 s/chunk | **30.0 s/chunk** |
| CPU 占用 | 1.23 核 | 1.96 核 |

**32 核里只用 1.2 核 → 瓶颈在 GPU 上的 ERNIE 前向，不在 CPU 线程。**
加到 32 只会让 CPU 空转，已改回 8。

### 27.2 崩溃根因：辅助进程抢 HIP 上下文

两次 0xC0000005（`libomp140.x86_64.dll`）都发生在**训练进行中又启动了另一个
`import torch` 的进程**时。venv 里的 torch 是 ROCm 版，`import torch` 会初始化 HIP
上下文，AMD 驱动会让先持有上下文的训练进程崩掉。

**规则：训练期间任何辅助 Python 进程都要带 `CUDA_VISIBLE_DEVICES=""`，让它 CPU-only。
监控一律用 PowerShell 读文件，不碰 torch。**

### 27.3 两个被实测否掉的优化假设

| 假设 | 结果 |
|---|---|
| 按长度分桶能省一半 padding | ❌ 脚本把每条序列都裁成 101 nt，所有 batch 本来就等长（日志 `first=median=last=101`） |
| 每块 200 MB 临时分配导致越来越慢 | ❌ 改成预分配 + `index_add_` 后，速度仍在 15–30 s/chunk 之间波动，与改动无关 |

### 27.4 一个会误导人的读数

`[train] chunk N ... X s/chunk` 里的 X 是 `elapsed / (chunk_index + 1)`。**续训时
chunk_index 从 501 开始，但实际只跑了 10 块**，于是打印出 0.8 s/chunk 这种假数。
以后看速率必须用「两次采样之间的 chunk 增量 / 墙钟时间」。

---

## 28. RNAQA v2：从「机械模板」换成「语义任务」（2026-10-31）

### 28.1 旧数据的真实多样性

```
rnaqa_seq30_full.jsonl   4,341,380 行 / 145,201 个不同 uid = 每条序列 29.9 个问题
30 个模板，全是字符串操作（first 6 nt / reverse complement / how many G / positions 1-6）
答案中位 5 字符，p95 = 8 字符
单 epoch 约 412.7 M token，其中答案只占 1.13%
```

**「434 万条」的真实多样性是 14.5 万序列 × 30 个字符串操作。** 这也解释了为什么
码通道在 A6 上打平 4-mer —— 任务本身没要求更深的表示。

### 28.2 v2 第一阶段（L2 注释 + L3 结构）

`scripts/60_build_rnaqa_v2.py` → `data/processed/rnaqa_v2_l23.jsonl`

```
777,212 行 / 145,201 序列 / 7 个模板
每条序列最多 8 个问题，模板上限 20%
切分按 Rfam family，按序列数均衡
```

| 项 | 数量 |
|---|---|
| `family_name` | 116,070 |
| `family_acc` | 145,201 |
| `rna_type` | 145,201 |
| `struct`（配对比例 / MFE / 最长茎 / 可及性） | 370,740 |

```
family 切分：train 1,198 / val 267 / test 483，跨 split 泄漏 = 0
行切分：train 614,203 / val 47,298 / test 115,711
```

对比旧数据：

| | 旧 seq30_full | 新 v2_l23 |
|---|---|---|
| 行数 | 4,341,380 | 777,212 |
| 每条序列问题数 | 29.9 | 5.4 |
| 模板数 | 30（字符串操作） | 7（语义） |
| 单 epoch token | 412.7 M | **59.9 M** |
| 答案 token 占比 | 1.13% | 2.21% |
| 有 split 字段 | ❌ 无 | ✅ family 级 |

**同样的序列池，用 1/7 的算力换掉 30 个字符串操作，还补上了 family 级切分。**

已知短板：答案中位仍只有 5 字符，**答案 token 只占 2.21%**。这套数据训练的是判别
（读码 → 选对类别），教不会模型解释。L4（definition/annotation 长答案）和 L6（自由
提问）要后续补，才能把长答案能力拉回来。

### 28.3 下一步的优先级

1. **等结构探针**（`scripts/61_struct_head.py`，排队中）——决定「码编码结构」这条腿成不成立
2. **位置敏感任务**——4-mer 词袋对置换不变，码信道有位置，这是唯一能把两者分开的地方
3. **因果接地四条**（置零 / 换码 / 打乱 / 只给问题）——纯推理，几小时
4. 扩数据与部署表格

**「打到 B 的水平」的定义**：在某一个下游任务上，码通道显著超过同容量基线。
现在 A7 是 Δ +0.006、CI 跨 0，还没到。

---

## 29. 2026-10-31：读出方式才是那个洞（进行中）

### 29.1 诊断：我们一直在拿「口袋」比「口袋」

回头看所有判别头的读出，全都是对码做**置换不变**的统计：

```
scripts/A1_gene_cls.py:298     hs = v.reshape(n, ncode, -1).mean(1)
scripts/61_struct_head.py:195  mean_c / std_c / max_c / min_c over codes
```

把 25 个码任意打乱，这五个量一个都不变。**A7 那个 Δ +0.006 打平，不是码表不够好，
是两个词袋容量本来就差不多。** 位置信息在读出这一步就被丢掉了，从来没进过头。

### 29.2 新码表 + 内容初始化

流式训练收尾时 `49_stream_tokenizer.py` 挂在 `val_wins` 为空（loader 会丢掉含 N/R/Y
的窗口，`kept` 达不到 `--n-seq - --val`）。权重没丢，用 `scripts/65_finalize_stream_tokenizer.py`
从 `state_latest.pt` 导出：

```
tokenizer.pt        chunk=1060  step=12732  4096 码
20k 窗口评估        mse 0.0936  used 3502/4096 (85.5%)
                    once 6.14%  median_reuse 109
transfer            acc 8.04%  ctrl 0.55%  lift 14.71x  ← IN-SAMPLE，不能当判据
```

`51_content_init_codebook.py` 加了 `--fsq` 参数，给新码表生成内容初始化：

```
同 6-mer 两处的余弦   +0.4010   （旧码表 +0.3130）
随机的余弦            +0.1131   （旧码表 +0.0170）
```

### 29.3 位置分箱读出（`scripts/64_position_readout.py`）

把码序列按**相对位置**切成 B 段，每段取均值。B=1 就是旧的口袋读出，所以两者来自
同一个文件：`X` 是分箱（B×2048），`Xmean` 是全局均值（2048）。

### 29.4 20k 试点结果（同一批行、同一个头）

| 特征 | test AUC |
|---|---|
| 4-mer 计数（同容量头） | 0.5611 |
| **码通道 · 口袋读出（新码表）** | **0.5848** |
| 码通道 · 位置分箱 B=5 | 0.5744 |
| 码通道 · 打乱序列 | 0.5134 |

配对 bootstrap（码通道口袋 vs 4-mer，2000 次）：

```
real 0.5935  kmer4 0.5567  Δ +0.0369  CI [+0.0062, +0.0682]  P = 0.0105
```

**这是第一次在同容量基线上赢**。但 n_test 只有 1,636，区间下界刚过 0，只能算「有信号」。

两个附带观察：

1. **位置分箱（B=5）反而更差**（0.5744 vs 0.5848）。5 个码取均值比 25 个码取均值噪声大，
   而头的参数量翻五倍、训练行只有 1.5 万，容易过拟合。B=2/3 还没试。
2. 打乱序列掉到 0.5134，和 A6 的 +0.058 一致 —— **序列排布确实携带信息**，
   只是口袋读出抓不到「怎么排」，只抓得到「排了些什么」。

### 29.5 待确认

全量 104,920 条（test 7,556）的抽取正在跑。只有全量上也显著，才能写进论文。
如果全量回落，说明 20k 那个 +0.037 是小样本波动，仍然要回到「改码表目标函数」那条路。

---

## 30. 2026-11-01/02：v3 训练吞吐、MoE 加速、任务形态重构

这一节记录三件事：把 35B 训练从「GPU 空转」救回来、找出「训不动」的真正原因、
把任务从「生成家族名」改成「候选判别」。

### 30.1 MoE 专家执行：从逐专家到 grouped

**问题**：Qwen3.6-35B 每层 256 个专家、每个专家两个 int4 Linear。原来的
`_forward_per_expert` 是逐专家循环，每层数百次小 kernel（反量化 + 小 GEMM），
GPU 大量时间在两次 launch 之间空转。A100 上 nvidia-smi 常见 0%，偶发尖峰。

**为什么不能用 `torch._grouped_mm`**：torch 2.11.0+cu130 上前向可用，但
**反向直接抛** `RuntimeError: Invalid strides/sizes, got [0,0] for strides`。
训练必须支持对 `hidden_states` 的梯度，所以走 autograd 稳定的 `bmm`。
bnb 的 `matmul_4bit` 又强制 `quant_state` 为 2D，没法直接做批量 NV4 GEMM。

**做法**（`src/rnaqwen/qwen36.py`，`--experts grouped`）：
1. 按 `(expert, slot, token)` 稳定排序路由，`bincount` 得每专家计数
2. 命中专家的 int4 权重**批量反量化**到 `[E_hit, out, in]` 一块缓存
3. 路由 token 散射进 `[E_hit, max_c, H]`，两次 `bmm` 完成门控/下投影
4. 权重冻结，autograd 只回传到 `hidden_states`

**关键坑：nested double-quant 的 absmax 还原**。逐专家调
`bnb.dequantize_blockwise` 每层 512 次，profiler 显示这组调用 ~6 ms 且 CPU
间隙更大。改成 `code[uint8码] * state2_scale[block] + offset` 一次算完整层。
**注意 offset 是每个专家各自不同的标量**，一开始拿 expert0 的套给所有专家，
引入 1 个 bf16 ULP 误差（测试当场 FAIL），按专家广播后恢复 `0.000e+00`。

**内存**：整层 256 专家的 bf16 权重约 2.1 GiB，加 padded 激活在 40GB 上 OOM。
按 32 个专家一块处理，峰值临时显存从 ~3.6 降到 ~0.5 GiB。

**但真正的 OOM 根因不是这个**：分块 forward 的 8 个块的激活仍全部挂在
autograd 图上等 backward，路由倾斜时整层瞬时 ~12 GiB。修法是自定义
`_GroupedMoEFn(torch.autograd.Function)`：forward 全程 `no_grad`（只保存路由
索引），backward 按相同块边界**逐块重算**、算完即释放。任一时刻只有 1 个块的
临时显存。附带好处：MoE 部分不再吃 gradient checkpointing 的 3 倍前向。

**实测**（A100 40GB，单层真实 dims：E=256, H=2048, I=512, T=8192, top-8）：

| | per_expert | grouped（初版） | grouped（批量反量化后） |
|---|---|---|---|
| 单层 fwd+bwd | 0.405 s | 0.081 s | **0.027 s** |
| 提速 | 1× | 5.0× | **14.3×** |
| 反量化 vs 逐专家 | — | 逐位一致 | **逐位一致 0.000** |

端到端（A100，token-budget 8192 / batch 64）：step 2/3 稳态 **9.1–9.6 s/步**
（基线 step 1 约 201 s / 20 样本），GPU 利用率 **87–88%**，显存 26.7 GiB。

**注意**：step 1 的 178 s 是 FLA 线性注意力 backward kernel 首次编译，一次性。
别拿 step 1 判断稳态。

**一个被否掉的优化**：把 `gather` 回填改成 `tok_padded + index_add` 是**负优化**
（0.027 → 0.035 s）。padded 行数（89,600）比路由数（65,536）多 37%，得不偿失，已 revert。

### 30.2 SOS 正交正则：从「接不上」到「接得上但用不着」

`sos_lora_penalty` 原来是 `total += p` 对 190 层求和再乘 `coef=1.0`。实测：
单层 penalty 值中位 **1.99e-08**，全模型 loss 项 **2e-5**（占 ce=0.3 的 0.007%），
对 A 的推动只有 **4e-8/步**（相对 Adam 步长 2e-4 是 0.02%）。

原因有两层衰减：
- `dirs = a / ||a||` —— 归一化让梯度除以 `||a||² ≈ 36`
- 再除 `k(k-1) = 12`

**修法**：不再归一化，直接罚未归一化的内积平方；并把跨层求和改成按层平均
（这样 coef 的含义与 preset 层数无关）。修复后梯度范数 2.0e-4 → 1.9e-3（**9×**），
功能自检：完全塌缩的层 penalty 是正交层的 **14 倍**。

**但要说清楚**：这个修复**不会让当前训练变快或变准**。实测 A 的余弦相似度
≈ 0.0002（初始化就正交），没有塌缩需要纠正。它是保险丝，不是引擎。

### 30.3 「训不动」的真正原因：任务定义锁死了上限

原 v3 任务：给序列，**生成 1,947 个自由文本家族名中的一个**。

**诊断链**：
1. val CE 在 step 1000 达到最优 1.1709，之后 4,500 步在 1.17–1.47 震荡，不再改善
2. train CE 0.2 vs val CE 1.2 —— 6 倍差距，典型过拟合早期
3. 永远猜众数（RF01960，占 41.4%）的 CE = 0.883 —— 模型比无脑猜还差
4. **决定性证据**：数据是**按家族切分**的，train 1,198 个家族与 test 483 个家族
   **交集为 0**

所以 exact match 在 test 上**结构性地只能是 0**，训多久都一样。这不是训练量问题，
也不是参数问题。顺带排除了另外三个嫌疑：

| 嫌疑 | 实测 | 判定 |
|---|---|---|
| 码信息量不足 | 按模板分开看，同一码 → 唯一答案 **100.0%** | ✅ 够 |
| 可训参数太小 | 已学会输出合法 `RFxxxxx` 格式 | ✅ 够 |
| 学习率/优化器 | Adam step 正确、exp_avg 1521/1521 非零、wd/eps 标准 | ✅ 健康 |
| SOS 正交约束 | 实际 loss 项 2e-5，占 0.007% | ✅ 无干扰 |

（补记：我一开始算「同码对应 2 个答案占 75.6%」，是**跨三个模板混算**造成的假象 ——
同一 RNA 在 acc 模板答 `RF01960`、在 name 模板答 `SSU_rRNA_eukarya`，指的是同一个东西。）

### 30.4 任务重构：候选判别（`scripts/build_candidate_dataset.py`）

**新形态**：
```
序列 + "Which of these Rfam families does this RNA belong to?
A) RF01788
B) RF00651
C) RF00005
D) RF00053
Answer with the letter only."
-> "C"
```

- 家族名**直接出现在 prompt 里**，模型不需要背；对见过的家族也有效
- **随机划分**（不用原始的按家族切分），test 家族 100% 在 train 出现过
- 4 选 1，A/B/C/D 各占 25%（模型不能靠猜字母），随机基线 25%
- 候选池 = 至少 5 条序列的 483 个家族，覆盖 97.8% 样本
- 产出 113,525 候选题 + 34,057 结构题（`--struct-ratio 0.3` 降采样）

**数据量核算**：

```
候选题 113,525（train 87,307 / val 9,103 / test 17,115）
真实步长 p50 = 156（比原数据集长 40%，question 里多了 4 个候选）
token-budget 8192 -> 约 52 条/步 -> 1 epoch ≈ 1,679 步
训 8,000 步 ≈ 4.8 epoch（原来是 0.28）
```

长尾仍在：483 个家族里只有 184 个有 ≥10 条样本。但前 63 个（≥50 条）覆盖大部分样本，
预期准确率上限约 70–80%，远超随机的 25%。

### 30.5 工程事实

- **`--sort-by-length` 是码表编码提速的关键**：9 条/秒 → 163 条/秒（**17×**），
  ETA 从 220 分钟降到 10 分钟。同批序列长度接近，padding 浪费从几倍降到接近零。
  加线程（16→32）和 batch（8→16）也有贡献，但排序是决定性的。
- batch 48 会 OOM（attention 是 O(n²)，1024 长度下要 8.3 GiB）。
- 本机是 AMD APU 共享内存（99.7 GiB 总量），`torch.cuda.memory_allocated()` 常显示 0，
  但实际可用受系统内存限制。
- Windows 路径不能直接传给 `from_pretrained`（会被当成 HF repo id 报
  `Repo id must use alphanumeric chars`）。生产服务 `gguf/rna_prompt.py` 现在就是
  因为这个 bug 处于 degraded 状态。

### 30.6 当前状态与下一步

- notebook 已切到候选式数据集（`RNAQwen_v3_Full.ipynb`，commit `1f8375b`）
- 训练脚本**无需修改**：`_mk_new` 直接读 `question`/`answer`，`build_batch` 天然兼容
  多行 question + 单字母 answer
- 码表 `_rnaqa_v3_cand_codes.*` 生成中
- 待办：上传 Release（`dataset-v3-cand` / `codes-v3-cand`）→ A100 开训 8,000 步
  → 用新增的 4 选 1 准确率 cell 评测
- 更远的目标：生产可用（真实序列进、自然语言答案出）。需要先修服务的路径 bug，
  再决定 GGUF 导出（磁盘 87G 不够，导出链路要 ~125G）

### 30.7 定位修正：这是「序列理解层」，不是 RNA-GPT 复现

读完原论文（`data/external/paper_2411.08900.txt`）后要修正一个根本性误解。

**RNA-GPT 的核心不是"让模型读序列"，而是"把文献知识接到序列上"：**

> "The embedded sequence information is encoded into **the soft prompts** and
> prepended to the question prompt."

它把整条 RNA 压成 **1 个软提示向量**，回答的是文献摘要里的语义内容
（"The RNA is involved in transcript splicing regulation along with RNVU1-18 and CLK1"）。
这类答案**不可复算**，来自 GPT-4o-mini 对 407,616 条 RNA 相关文献的分组摘要
（LDA 分主题 + divide-and-conquer 总结）。

**我们没有资源做这件事**（要跑 40 万条 RNA 的文献抓取 + 摘要），
而且**在 2026 年也不需要**：

| 能力 | 2026 年的做法 | 需要微调吗 |
|---|---|---|
| 家族 / 类型判定 | Rfam / HMMER 直接查，或序列检索 | 否 |
| 结构参数（MFE、配对比例） | ViennaRNA 直接算 | 否 |
| 文献语义问答 | 序列→相似 RNA 检索→文献库 RAG→长上下文 LLM | **否**（RAG） |
| **读 RNA 序列本身** | 无成熟方案 | **是** ← 这才是本项目 |

**所以本项目的正确定位：**

> 训练一个能"读"RNA 序列的 adapter，把序列转成可靠的结构化事实
> （家族、类型、结构参数、序列细节）。文献语义问答交给上层 RAG 系统，
> **不在本项目范围内**。

**这个定位带来三个好处：**

1. **能力边界诚实** —— 不承诺做不到的文献知识
2. **评测指标对齐** —— 4 选 1 家族准确率正好衡量"读序列"这一步
3. **答案可验证** —— 可复算事实有客观对错，不像 GPT-4o 摘要可能有幻觉

**架构上的分工（本项目只做第 1 层）：**

```
用户: "这条 RNA 是什么，有什么功能？"
   [1] 序列 → 家族/类型/结构   ← 本项目（FSQ 码 + SOS-LoRA）
   [2] 家族/序列 → 文献        ← RAG 检索（不需要训练）
   [3] 汇总成自然语言答案       ← 长上下文 LLM
```

**现有数据的题型构成（100% 可复算，无一条依赖文献）：**

| 类型 | 条数 | 占比 | 答案示例 |
|---|---|---|---|
| 结构参数（l3_*） | 370,740 | 34.9% | `0.70` / `-26.9` / `15` |
| 家族 accession | 145,201 | 13.7% | `RF00005` |
| RNA 类型 | 145,201 | 13.7% | `tRNA` |
| 家族名 | 116,070 | 10.9% | `tRNA` |
| 整段描述（l4_*） | 283,606 | 26.7% | 模板拼接（见下） |

⚠️ **`l4_*` 的"整段描述"是模板拼接，不是知识**：

```
This sequence belongs to Rfam family RF00005 (tRNA), which is a tRNA type RNA.
The submitted sequence is 80 nucleotides long.
```

它把上面几个可复算事实串成句子。**看起来像文献摘要，实际没有任何文献内容。**
这类数据留着只用于让模型学会组织语言，不能当作知识来源。

**下一步优先级（按此定位重排）：**

1. **候选判别训起来**（4 选 1 家族，数据已就绪）—— 这是"读序列"能力的第一个可验证指标
2. 结构参数任务（l3_*）单独评估 —— 它们本质是数值回归，可能需要不同输出形式
3. 生产服务修好后，把 RAG 层接在模型外，而不是试图训进模型里

### 30.8 任务形式：参考 ChatNT 的基础任务集

读了 ChatNT 原文（Nature Machine Intelligence 2025, DOI 10.1038/s42256-025-01047-1）
之后，把任务形式重新梳理了一遍。它的 27 个任务构成：

| 类别 | 任务 | 我们有数据吗 |
|---|---|---|
| DNA（21 个） | 启动子、增强子、剪接位点、染色质可及性、甲基化、组蛋白 | ❌（且我们做 RNA） |
| RNA（3 个） | 降解率、稳定性、polyadenylation | ❌ 需要实验测量数据 |
| 蛋白（3 个） | 熔解点等 | ❌ |

**关键事实**：ChatNT 的 RNA 任务全靠**实验数据**（APARENT2、RNA 稳定性测定），
我们没有、也拿不到。照抄它的任务集不现实。

**但有一块它覆盖很弱、我们现成的**：**RNA 二级结构参数**。ChatNT 的 RNA 任务
里没有 MFE、配对比例、茎长这类，而这恰好是 ViennaRNA 能精确算的。

**采用的任务组合（务实版）**

核心（训练用，数据现成）：

| 任务 | 形式 | 数据量 | 测什么 |
|---|---|---|---|
| 家族判别 | 4 选 1 | 113,525 | 能否读出序列身份 |
| 结构参数 | 数值回归（MFE / 配对比例 / 茎长 / 可及性） | 370,740 | 能否读出序列结构 |
| 序列细节 | 短文本（前 N 碱基、反向互补） | 可生成 | 基础能力 |

暂缓（缺数据）：RNA 降解率、稳定性、polyadenylation。

明确不做：DNA 类任务（不是本项目的模态）、文献语义（RAG 层负责）。

**为什么这样切**：这三个核心任务覆盖了"读序列"的三个层次 —— 身份（家族）、
结构（折叠参数）、字面（序列细节）。如果 adapter 在这三层上都成立，
它就是真的在读序列，而不是靠问题模板猜答案。

### 30.9 桥接方式对照实验（进行中）

ChatNT 用「ERNIE/NT 连续 hidden → **问题感知** Perceiver resampler → LLM」，
我们用「ERNIE hidden → FSQ 离散码 → 查表 → LLM」。论文里问题感知投影
比普通投影高 6 点 MCC（0.71 → 0.77），说明"投影层看得到问题"很重要。

而 FSQ 码是**与问题无关**的固定符号串 —— 这是信息瓶颈的极端版本。

`scripts/_bridge_compare.py` 做对照：固定 ERNIE 编码器和数据，只换桥。

| 臂 | 桥 | 特点 |
|---|---|---|
| A | FSQ 码 → 嵌入表 | 码固定，与问题无关 |
| B | 连续 hidden → 问题感知 resampler（gated cross-attention） | 论文路线 |

评测：候选式 4 选 1 准确率（随机 25%）。这是**桥本身**的能力上限，
不含 35B 的语言能力 —— 如果 B 明显优于 A，就该换桥，而不是继续训 35B。

（注意：实验里的"问题向量"目前用随机投影模拟，不是真实英文嵌入。
这会让 B 臂的问题感知能力被低估，所以 B 的优势如果出现，是下界。）

---

## 31. 2026-11-02：任务转向 —— 从「单序列问答」到「RNA-RNA 关系判别」

这是对 30 节方案的**根本性修正**。30.8 里我把「RNA 二级结构参数（MFE、配对比例、
茎长）」当成核心训练任务之一，这是错的：**ViennaRNA 能精确算的东西，不该让模型学。**
模型学一个可被 O(1) 算法精确计算的性质，既不可靠也不必要。

### 31.1 转向的内容

| 项 | 30 节（旧） | 31 节（新） |
|---|---|---|
| 结构参数（MFE/配对/茎长） | 训练进模型 | **移到 harness 当 tool**（ViennaRNA） |
| 核心任务 | 单序列 → 事实问答 | **双序列 → 关系二分类** |
| 标签来源 | 自造（4 选 1 家族） | **实验验证的真实标签** |
| 证据 | 无 | **文献片段**（RAG 提供，模型判断是否支持） |
| 家族/类型分类 | 核心 | 保留，作为「序列理解层」的能力基线 |

**分工原则**：

- **能算的** → tool（ViennaRNA 算结构、Rfam/HMMER 查家族、序列字符串操作）
- **能检索的** → RAG（文献事实、功能描述、已知相互作用）
- **不能算也不能检索的** → 模型（**读序列本身**，判断两条序列之间有没有关系）

模型只负责最后一层。这也正是 Harvard Zitnik 组（arXiv 2610.00898）的结论：
benchmark 准确率高不等于模型真的用了序列输入，必须做输入扰动/证据冲突检验。

### 31.2 新任务梯队

**第一梯队：miRNA-mRNA 靶向关系**（最干净的起点）

- 数据源：
  - **MiRBench**：来自 CLASH 和 chimeric eCLIP 实验的**偏差校正**数据集
  - **TargetScan**：curated 数据，约 **56 万对**人/鼠 miRNA-mRNA（28 万正 / 28 万负）
- 任务形式：
  ```
  序列 A（miRNA）+ 序列 B（mRNA）
  + "两者之间是否存在经实验验证的靶向关系？证据：[文献片段]"
  -> 是 / 否
  ```
- 为什么先做：**标签最可靠**（实验验证 + 偏差校正）、数据量最大、两条序列都放得下
  （miRNA ~22 nt，mRNA 取 3'UTR 片段）

**第二梯队：lncRNA-miRNA 海绵/调控**

- 数据源：**starBase**、**DIANA-LncBase**、**RAID**
- 挑战：lncRNA 常超编码窗口（1024 nt），需要**截断或片段化**；
  片段化后要看「取最大分」还是「平均」，这会影响标签语义

**第三梯队：一般 RNA-RNA 功能关联**

- 数据源：**NPInter v4**、**RNAInter**
- 挑战：正样本质量参差，**必须用文献证据片段筛选**，只保留有明确实验描述的关系

### 31.3 RNAcentral 能复用多少（实测）

当前 `rnaqa_v3_l234.jsonl` 来自 RNAcentral/Rfam：**145,201 个唯一 RNA，1,947 个家族**。

| 资源 | 现状 | 可复用到新任务吗 |
|---|---|---|
| 序列 + 家族注释 | 145k RNA | ✅ **高**：继续喂「序列理解层」 |
| **miRNA 序列** | 4,645 条 / 718 家族（~22nt） | ✅ **可直接用**：对 TargetScan 的 miRNA |
| mRNA 3'UTR 序列 | 2,708 条（且非真 3'UTR） | ❌ 要从 Ensembl/GENCODE 拿 |
| lncRNA 序列 | **0 条** | ❌ 要从 GENCODE/NONCODE 拿 |
| miRNA-mRNA 配对标签 | **0** | ❌ 必须从 TargetScan/MiRBench 拿 |
| 文献证据片段 | **0** | ❌ 要从 PubMed 拿 |

**RNA 类型构成**（唯一序列）：

| 类型 | 家族数 | 行数 | 占比 |
|---|---|---|---|
| rRNA | 10 | 75,775 | 52% |
| tRNA | 5 | 17,850 | 12% |
| miRNA | 718 | 4,645 | 3% |
| mRNA | 6 | 2,708 | 2% |
| lncRNA | 0 | 0 | 0% |

序列长度 p0=20 / p50=307 / p90=782 / p99=1,000 / max=1,022 nt。

**结论：RNAcentral 是「序列语料」，不是「关系标签」。**
它能喂饱序列理解层，但关系判别任务的数据必须外部拿。
唯一能直接接上的是 **miRNA 序列（4,645 条 / 718 家族）**。

### 31.4 三个待定的技术问题

1. **双序列怎么输入**
   - 各自编码后拼接 / 交叉注意力
   - 还是拼成一个输入 `<RNA>A</RNA><RNA>B</RNA>`
   - 前者更可控（两个编码器可共享），后者更接近文本对话

2. **超长序列怎么处理**
   - lncRNA / mRNA 可能数千 nt，编码窗口 1024
   - 截断？滑窗取最大分？还是只编码种子匹配区（miRNA 的 2-8 位种子）
   - **注意**：如果只取种子匹配区，模型就退化成「查种子」，学不到东西

3. **标签不平衡**
   - TargetScan 是正负各半，真实分布极度不平衡（绝大多数 pair 无关）
   - 训练用平衡集，**评测要不要用真实分布**？两者结论可能相反

### 31.5 桥接方式的实验证据（30.9 的结论）

`scripts/_bridge_compare.py` 在 2,000 训练样本上跑出的结果：

```
FSQ(码) 桥：ep1 test acc 25.80%  ep2 25.20%（随机 25%）
```

**FSQ 桥在 4 选 1 家族判别上完全没学起来。** 这支持转向 ChatNT 式的
「连续表示 + 问题感知投影」：

- ChatNT (Nat Mach Intell 2025) 的投影层是 **English-aware Perceiver resampler**，
  论文实测比普通 Perceiver 高 6 点 MCC（0.71 → 0.77）
- 我们的 FSQ 码是**与问题无关**的固定符号串，是信息瓶颈的极端版本

（注：该实验的 ChatNT 臂因设备 bug 未跑完，结论待补。FSQ 臂的结果本身已足够
说明「离散码 + 固定查表」这条路在小模型上不可行。）

### 31.6 下一步

1. **查数据可用性**：TargetScan / MiRBench 的规模、格式、许可
2. **确定双序列输入形式**（31.4 问题 1）
3. **在 miRNA-mRNA 上做二分类原型**：用 RNAcentral 的 4,645 条 miRNA 序列
   + 外部 mRNA 3'UTR，先验证「模型能否判定靶向关系」
4. 通过后再扩到 lncRNA-miRNA

**放弃的东西**：把结构参数当训练任务（应交给 ViennaRNA）、
把文献语义训进模型（应交给 RAG）。

### 31.7 最终任务定义（2026-11-02 定稿）

**任务名**：RNA-RNA 关系判别（RNA relationship discrimination）

**输入**

```
miRNA 序列（21–23 nt）
+ mRNA 片段（50 nt，种子匹配窗口）
+ 问题："这两个序列之间是否存在经实验验证的靶向关系？"
+ [可选] 文献证据片段（RAG 提供）
```

**输出**：`是 / 否`（二分类），可选附依据。

**为什么是这个任务**（四条，缺一不可）

1. **标签可验证** —— CLASH / chimeric eCLIP 实验数据，经偏差校正
2. **双序列都放得下** —— 22 + 50 = 72 nt，远小于 1024 编码窗口
3. **直接对应生产** —— 文献库维护就是判断两条 RNA 有没有关系
4. **工具替代不了** —— ViennaRNA 算不出靶向，Rfam 查不出配对

**数据源实测**

| 数据集 | train | test | 备注 |
|---|---|---|---|
| **AGO2_eCLIP_Manakov2022** | **528.6 MB** | 68.3 MB | 主力；另有 leftout 4.2 MB |
| AGO2_CLASH_Hejret2023 | 1.7 MB | 0.2 MB（1,018 条） | 小，适合快速验证 |
| AGO2_eCLIP_Klimentova2022 | — | 0.2 MB | 仅测试 |

- 许可 **CC-BY-4.0**，Zenodo record `20540907`
- `pip install miRBench` 即可取（`miRBench.dataset.get_dataset_df`）
- 格式：`gene`(50 nt) + `noncodingRNA`(21–23 nt) + `label`(0/1)
  外加 `noncodingRNA_name`、`feature`（exon / 3'UTR / intron）、
  `chr/start/end/strand`、`gene_phyloP`、`gene_phastCons`
- 标签平衡：CLASH test **523 负 / 495 正**
- **TargetScan 反爬**（Anubis JavaScript 挑战，curl 拿不到），需要浏览器；
  有了 Manakov 528 MB，暂不依赖它

**三层分工**

| 层 | 负责 | 实现 |
|---|---|---|
| Tool | 结构参数、家族查询、序列细节 | ViennaRNA / Rfam-HMMER / 字符串操作 |
| RAG | 文献事实、功能描述 | OpenHarness 检索 |
| **模型** | **读序列 + 判断关系** | SOS-LoRA adapter on Qwen3.6-35B |

**评测设计（关键：不能只看准确率）**

| 检验 | 预期 | 失败含义 |
|---|---|---|
| 正常 AUROC | > 0.7 | 学不会 |
| **打乱 miRNA** | 掉到 ~0.5 | 没在读 miRNA |
| **打乱 mRNA** | 掉到 ~0.5 | 没在读 mRNA |
| **换成不配对 miRNA** | 预测随之改变 | 忽略配对关系 |
| **zero probe**（表示置零） | 掉到 0.5 | 序列输入没起作用 |

这套检验来自 Harvard Zitnik 组（arXiv 2610.00898）：
**benchmark 准确率高不等于模型真的用了序列输入**，
必须做输入扰动与证据冲突检验。ChatNT 在该文里被列为「序列输入确有贡献」的模型之一，
BioReason/BioReason-Pro 则在 97.9%/99.7% 的冲突中跟着文本走。

**桥接方式**

放弃 FSQ 离散码。证据：`scripts/_bridge_compare.py` 在 2,000 样本上，
FSQ 桥 2 epoch 后 test acc 25.80% / 25.20%（随机 25%），完全没学起来。

改用 **ChatNT 式：连续 hidden → 问题感知投影（gated cross-attention）**。
论文实测问题感知比普通投影高 6 点 MCC（0.71 → 0.77）。

**与当前 4 选 1 任务的关系**

4 选 1 家族判别**降级为「序列理解层」的代理任务**，不再是目标。
它便宜、能快速验证「adapter 能不能区分 RNA」，但不能代替关系判别。

**实施顺序**

1. 搭 MiRBench 数据管线（下载 → 转 jsonl：`(mirna, mrna_window, label)`）
2. 双序列输入形式定稿（各自编码 → 拼接 vs 交叉注意力）
3. 小规模原型：CLASH 的 1,018 条 test + Manakov 子集，先跑通二分类
4. 补做完整的输入扰动检验
5. 通过后再扩到 lncRNA-miRNA（starBase / DIANA-LncBase / RAID）

### 31.8 中文提示词：为什么用，以及后续定制（实验暂缓）

**结论：默认中文，保留英文做对照。**

理由：

1. **Qwen 中文原生强** —— 通义千问的中文指令遵循、中文输出控制都比英文稳
2. **与 ACGU 天然区分** —— 英文 prompt 里 `ACGU` 与英文子词容易混；中文语境下
   ACGU 是唯一的外来符号，边界清晰
3. **答案更短** —— 中文「是/否」一个 token；英文 `Yes/No` 也可能是一个 token，
   但存在子词切分歧义（尤其 `Yes` 常被切）

**当前实现**（`scripts/build_mirbench.py --lang zh`）：

```
Q: 这条 miRNA 与这段 mRNA 窗口之间是否存在经实验验证的靶向关系？只回答「是」或「否」。
A: 是
```

区域任务用缩写（`3'UTR` / `5'UTR` / `CDS` / `Intron`），中英通用，避免翻译带来的
标签漂移。**4 选 1 的 A/B/C/D 保持字母** —— 中文字符可能被拆成多 token。

**后续可以在训练中定制的中文点**（按优先级）：

| 定制点 | 做法 | 收益 |
|---|---|---|
| 问题模板多样化的中文表达 | 同一任务生成 3–5 种中文问法 | 提升对问法变化的鲁棒性 |
| 中文答案的同义归一 | 「是/是的/存在」归一到一个标签 | 减少评测口径误差 |
| 全角标点处理 | 统一用半角，或检测并归一 | 避免 tokenizer 意外切分 |
| 中英混合模板 | `miRNA` / `mRNA` 保留英文术语 | 保持领域术语一致性 |

**中英文对照实验（写进计划，暂不执行）**

- 变量：`--lang zh` vs `--lang en`
- 控制：同一批数据、同一随机种子、同一训练步数
- 指标：AUROC / AUPRC（不平衡任务优先看这两个，不看 accuracy）
- 这是 `_bridge_compare.py` 之后第二个该做的 A/B
- 暂缓原因：先确认双序列输入和桥接方式能跑通，否则对照没有意义

### 31.9 待实现：双序列输入 + 问题感知投影

当前 `50_routeC_e2e.py` 的 `build_batch` 只支持**一个** RNA 占位段。
关系判别需要两条序列。同时 FSQ 码桥已实测不可行（4 选 1 上 25% = 随机），
要换成 ChatNT 式的问题感知投影。

**目标架构**

```
miRNA (22nt) ──ERNIE──> [L1, 768] ─┐
                                   ├─> 问题感知 resampler ─> [8, 2048] 软提示
mRNA 窗口 (50nt) ─ERNIE──> [L2, 768]─┘        ↑
                                          问题 embedding
软提示 + 问题文本 + " A: " → LLM(SOS-LoRA) → 是/否
```

**关键设计点**

1. **两条序列分别编码**，不拼接 —— 拼接会让模型分不清哪段是 miRNA
   （且 miRNA 只有 22nt，在拼接序列里会被淹没）
2. **resampler 看得到问题** —— ChatNT 论文实测：问题感知 0.77 vs 普通 0.71 MCC
3. **软提示固定 8 个 token**（每条 4 个）—— 比 FSQ 的几十上百个码 token 短得多，
   也避免了"码与问题无关"的信息瓶颈
4. **ERNIE 冻结，resampler 可训** —— 与 ChatNT 一致（它微调 encoder，我们冻结，
   因为 35B 训练本身就贵）

**实现路径**

- 新脚本 `scripts/52_routeR_relation.py`，不破坏 `50_` 的候选式训练
- 复用 `qwen36.load_qwen36_text`（35B 加载）和 `sos_lora.replace_linear_with_sos`
- 新写：双序列数据加载、问题感知 resampler、训练循环、AUROC 评测
- 先用 CLASH 的 9,158 条验证跑通，再用 Manakov 528 MB 放大

### 31.10 MiRBench 数据管线实测结果（2026-11-02）

`scripts/build_mirbench.py` 把 Zenodo 20540907 的 7 个原始文件转成统一 jsonl
（中文 prompt）。实测规模：

| 任务 | 条数 | 分布 |
|---|---|---|
| **mirna_target**（二分类） | **2,849,872** | train 1,247,038 正 / 1,256,637 负；test 177,898 正 / 168,299 负 |
| **mirna_region**（四分类，仅正样本） | **1,279,216** | 3'UTR 575,419 / CDS 267,439 / Intron 360,300 / 5'UTR 76,058 |

按数据集拆：

| 数据集 | 行数 |
|---|---|
| AGO2_eCLIP_Manakov2022 | 4,114,768 |
| AGO2_CLASH_Hejret2023 | 12,939 |
| AGO2_eCLIP_Klimentova2022 | 1,381 |

输出 1.9 GB（`data/processed/mirbench.jsonl`），**不进 git**（已加 .gitignore）。

**关键数字**：target 任务 train 正负几乎完美 1:1（差 0.8%），
这是因为 miRBench 的负样本是用偏差校正方法按正样本配平生成的 —— 不是简单随机。

**注意**：这批数据比 ChatNT 每个任务（~1,000 QA 对）大三个数量级，
所以不需要担心"数据不够"。要担心的是：
1. 4.1M 条 × 每条 2 次 ERNIE 编码 = 大量编码时间（需要缓存 hidden）
2. 区域任务四分类的类别不平衡（3'UTR 占 45%，5'UTR 只占 6%）


## 32. 2026-11-02：关系判别的落地、代码彻查与首次 A100 准备

§31 定的是任务与架构，这一节记的是**实现之后**的事实，以及一次逐行审查改掉的错误。
凡是和 §31.7 / §31.9 冲突的，以本节为准（§32.6 列了出入）。

### 32.1 实际实现

- 训练脚本：`scripts/52_routeR_relation.py`（新链路，不动 `50_routeC_e2e.py`）
- 模型：ERNIE-RNA（冻结，768）分别编码 miRNA / mRNA 窗口 → 两个问题感知
  Perceiver resampler（各 4 个 query）→ 8 个软提示 token → 冻结的 Qwen3.6-35B-A3B
  int4 + SOS-LoRA（preset C，r=8，k=4）→ 答案首 token 的 logit
- 可训参数：SOS 8.13 M + 两个 resampler 11.03 M = **19.16 M**
- 输入序列由 tokenizer 的 `apply_chat_template(enable_thinking=False)` 渲染后切开，
  软提示插在 `"Q: "` 和问题正文之间：

```
<|im_start|>system\n<|im_end|>\n
<|im_start|>user\nQ: [4 soft: miRNA][4 soft: mRNA]这条 miRNA 与这段 mRNA 窗口之间是否存在经实验验证的靶向关系？只回答「是」或「否」。<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
是
```

- 损失：答案首 token 的 CE + `1.0 × sos_lora_penalty`（按层平均）
- 指标：`logit(是) − logit(否)` 的 AUROC
- 结束附加输入扰动：行置换 miRNA / 行置换 mRNA / 逐行打乱 miRNA 碱基 /
  逐行打乱 mRNA 碱基，四个都应掉回 ~0.5

### 32.2 彻查改掉的 6 个真 bug

按严重程度排。前两个的共同点是：**训练会正常收敛，日志看不出异常，但学的东西不是设计的东西。**

1. **`pred_pos` 坐标系错位（最严重）**
   `make_batch` 算的是 suffix 内部下标 `len(pre)-1`，而模型是
   `hs[arange, pred_pos]` 索引 `[prefix][soft][suffix]` 拼接后的整条序列。
   于是每一步读的是问题中间某个 token 的 hidden，不是答案前一位。
   修法：`len(left) + 2*n_query + len(right) - 1`；并在 `make_batch` 里对
   **每个 batch** 断言 `pred_pos` 的下一位就是目标 token，错位当场抛错。

2. **chat 模板是手搓的，而且对 Qwen3.6 是错的**
   Qwen3.6 的 no-think 段是普通字符串 `<think>\n\n</think>\n\n`
   （token `[248068, 271, 248069, 271]`），不是 Qwen3 的
   `"<think>\n\n<｜end▁of▁thinking｜>\n\n"`（14 个 token，含 158819）；
   而且 `<|im_end|>` 后面漏了 `\n`。修法：不再手拼，直接用
   `apply_chat_template` 渲染，用哨兵定位软提示插入点，找不到就报错。

3. **冻结的 ERNIE 被 `model.train()` 切成了训练模式**
   ERNIE-RNA 的 config 是 `hidden_dropout=0.1, attention_dropout=0.1`。
   结果：每条训练样本的序列表示被随机丢 10%，评测时不丢。
   修法：`RelationModel.train()` 覆写，始终 `encoder.eval()`；LLM 底座保留
   train 模式（SOS 的 LoRA dropout 0.05 要靠它）。

4. **`model.to(DEV)` 递归到 bitsandbytes int4 底座**
   bnb 不支持量化后再 `.to()`。修法：只把两个 resampler 移到 GPU。

5. **SOS 正交正则没接上**
   新脚本一开始只有 CE，等于 K 个独立 LoRA。修法：加
   `1.0 × sos_lora_penalty`，日志里 `ce` / `sos` 分开打印。

6. **checkpoint 与 resume 的可靠性**
   - 直接写目标文件 → 同步线程可能上传半截的 200 MB zip。改为 `.tmp` +
     `os.replace`，上传前再验 zip。
   - resume 可能挑中 `best.pt`（无优化器状态、step 可能更早）。改为优先
     `step_*.pt`。
   - resume 现在校验 `tag == "rel"`、`weights` 字段、以及**张量数**，
     对不上直接报错（旧候选式 checkpoint 连权重都没有）。

另外两处参数性改动：

- 学习率拆成两组：SOS `2e-4`（50_routeC 验证过的值）、resampler `1e-3`
  （随机初始化从零学）。此前统一给 1e-3。
- 删掉 `" A: "` 脚手架，答案直接跟在生成提示后（见 §32.6）。

### 32.3 数据审计（本地实测，4.1 M 行全扫）

| 项 | 结果 |
|---|---|
| target 行 | train 2,503,675（正 1,247,038 / 负 1,256,637）；test 346,197（正 177,898 / 负 168,299） |
| 跨 split 重复对 | 57 对（占 test 0.02%） |
| 同对异标签 | 15 对 |
| test 来源 | Manakov 344,225（99.4%）/ CLASH 1,018 / Klimentova 954 |
| train 来源 | Manakov 2,495,535（99.7%）/ CLASH 8,140 |
| 字母表 | DNA（含 T）；multimolecule 的 `RnaTokenizer` 默认 `replace_T_with_U=True`，启动时实测 `ACGT == ACGU` 才继续 |

含义：泄漏可以忽略，但 **AUROC 基本是 Manakov 的**，不能当成跨数据集泛化。
CLASH 的 test 只有 1,018 行，在 n=2000 的评测子集里大约 6 行，没有统计意义。

### 32.4 当前训练配置（定稿）

```
--max-train 200000 --max-test 20000     # 均匀蓄水池，覆盖所有来源
--batch 64 --steps 8000
--lr 2e-4 --lr-proj 1e-3
--eval-every 500 --eval-n 2000
--save-every 200 --save-minutes 10 --keep-ckpt 3
--grad-ckpt 1 --perturb 1 --tag rel
```

- 200,000 / 64 = 3,125 步/epoch；8,000 步 = **2.56 epoch**，约 41 M token。
- 对照上一轮候选式：8,192 token/步 × 8,000 步 ≈ 65 M token、4.8 epoch。
  也就是说步数没变，但**覆盖的数据量和 token 量都少了**。
- checkpoint 存到 release `train-v3-rel`，运行目录 `runs/e2e_rel`。

### 32.5 速度：已知与未知

**已知（真实日志）**：候选式那轮，grouped MoE 优化后 9.1–9.6 s/步
（8,192 token、batch 64、grad-ckpt 1、GPU 87–88%）。

**未知**：关系判别一次都没跑过，没有实测步速。

按 token 推：5,120 / 8,192 = 62%，但固定开销（35B 权重、优化器、MoE 派发）不随
token 缩小，所以估计 6–9 s/步——这是估计，不是测量。更需要警惕的是**利用率**：
5k token 对 A100 偏小，大概率低于 87%。

时长：8,000 × 6–9 s ≈ 13–20 小时；notebook 的 `timeout` 是 7.5 小时，**必然跨
至少两段**。Colab 换 runtime 会丢 `/content`，第二段要重下 67 GB 模型再 resume。

待定（不猜，等第一次真实步速）：

| 方案 | epoch | 步数 | 备注 |
|---|---|---|---|
| 现状 | 2.56 | 8,000 | 先跑，用日志决定 |
| batch 96 | 2.56 | 5,334 | token ≈ 7.7k，利用率更好 |
| batch 128 | 2.56 | 4,000 | token ≈ 10.2k，优化器步数减半 |

### 32.6 与 §31.7 / §31.9 的出入（以本节为准）

- §31.9 写的是 `软提示 + 问题文本 + " A: " → LLM`。**" A: " 已删除**：
  模型预训练时答案就是直接跟在生成提示后的，塞一个没见过的分隔符没有依据。
- §31.7 写 `pred_pos` 是「" A: " 的最后一个 token」。现在它是**生成提示的最后
  一个 token**（`</think>\n\n` 的末尾），且是拼接后的**绝对下标**。
- §31.7 的扰动检验只有行置换。现在加了**逐行打乱碱基**——行置换可能碰巧抽到同
  一条 miRNA，扰动会退化。
- §31.10 说 7 个原始文件；实际用其中 6 个
  （`AGO2_eCLIP_Manakov2022_full_dataset.tsv.gz` 未使用）。

### 32.7 已知但未解决的设计问题

1. **问题感知支路目前是恒定的。** 20 万条共用一个问题模板，`q_emb` 每行相同，
   `q_proj` 学到的东西不随问题变化——这条支路要等第二种任务（region / lncRNA
   问法）接进来才有意义。不是 bug，是单任务阶段的必然。
2. **`mirna_region` 的 1.28 M 行没有进训练**（`load_rows` 过滤 `kind`）。
   作为辅助任务的价值待验证；它的答案是字符串、不是单 token，且只有正样本，
   接入属于结构性改动。
3. **没有任何 GPU 实测**：grouped MoE + grad-ckpt + 新损失这条组合还没有跑过；
   `[profile]`、步速、显存都等第一次 A100 日志。
4. **Colab MCP 未打通**：`colab-mcp` 已注册并通过健康检查，但工具要会话重启才
   加载，且它桥接的是浏览器里的 Colab（标签页不能关）。退路是手动 Run All。

### 32.8 下一步

1. Colab A100 跑 setup → 训练；记录 `[profile]`、步速、显存、GPU 利用率。
2. 500 步第一个 AUROC 出来之后判断：是否在爬；是否需要调 batch / 步数。
3. 训练结束看扰动检验：正常高、四个扰动全掉回 ~0.5 才算「真的在读序列」。
4. 通过后再上第二任务；不通过先定位，别急着加数据。
