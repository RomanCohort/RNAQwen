# RNA-GPT 复现与开源 — 阶段计划

> 建立 2026-10-08 · **2026-10-09 大修**（重读原文 + 实测数据回填）
> 本文档里的**每条外部事实**都在 2026-10-08/09 逐个核实过；**每条本机实测数字**都有对应的报告文件。
> 原论文全文已下载到 `data/external/paper_2411.08900.txt`，下面引用的原句都能在里面搜到。

---

## 0. 目标与验收标准

**目标**：复现 RNA-GPT，在**它自己的评测口径**上达到与原版同量级。

**验收指标（照抄原论文的两套，别自己发明）**

1. **ROUGE-1 / ROUGE-2 / ROUGE-L**
2. **Semantic Precision / Recall / F1** —— 三种嵌入算余弦：`BERT` / `PubMedBERT` / `text-embedding-3-large`
   （记作 `S_BERT` / `S_Pub` / `S_GPT`）

**目标分数（原版报数）**

| 指标 | D&C 变体 | AIS 变体 |
|---|---|---|
| Rouge-1 | **0.4791** | 0.5031 |
| Rouge-2 | **0.2690** | 0.3667 |
| Rouge-L | **0.4405** | 0.4747 |
| F1 @ S_BERT | **0.8609** | 0.8494 |
| F1 @ S_Pub | **0.7165** | 0.7293 |
| F1 @ S_GPT | **0.7560** | 0.7700 |

### 打平门槛：原文原句

> "We trained RNA-GPT using the flagship Llama-3 8B model architecture using a **smaller 5K RNA,
> 121K QA samples subset** for our initial model. We are in the process of training the large
> RNA-GPT that uses all 407,616 RNAs of RNA-QA with millions of QA samples."

→ **打平的门槛是 5K RNA / 121K QA，不是 407,616。**

### 论文对数据集的**规格声明**（原文 Introduction）

> "For over **407,616 RNA samples**, we create QA pairs, each accompanied by a comprehensive
> research summary based on available literature, and **between 5 and 14 QA pairs**."

**这条 "5 到 14 QA pairs" 是论文自己给的规格区间，可以直接用来定位我们站在哪：**

| | 论文声明 / 原版训练子集 | 我们 | 比例 |
|---|---|---|---|
| RNA 数 | 5,000 | **3,488** | **70%** |
| **每变体 QA/RNA** | **声明区间 5–14**；训练子集实际 **12.1** | **5.9** | 区间的**下沿** |
| 总 QA | 121,000 | ~41,000 | 34% |

（算术自洽：5,000 × 12.1 × 2 变体 ≈ 121,000 ✓）

→ **我们在论文声明的区间内，但贴着下沿。** 这句话本身就是"要不要加 QA 密度"的方法学依据 ——
不是拍脑袋，是论文自己划的合法范围。

### ⚠️ 但论文**从未声明发布**

全文搜 `release / publicly / we will / github / huggingface / data availability / supplementary`
只有一个"we will" —— 说的是**扩大训练规模**，不是发布数据。没有任何仓库链接、没有数据集许可证。
唯一的 license 是 arXiv 页面上的 **CC BY 4.0**（那是**论文**的，不是数据集的）。

**实测确认它确实没发布**：HF 搜 `RNA-QA` 零命中；GitHub 组织 `RNA-GPT` 只有一个 ★3 的静态页。

→ **所以本复现的第一个实质贡献，就是把这个缺口补上**（§P4）。
→ 也意味着"打平"没法严格验证：自建评测集上的绝对分与原报数**不可直接比较**。

### "打平"本身没法严格验证

原版**代码、权重、数据、评测集全都没开源**。你只能自建评测集，而自建评测集上的绝对分
与原报数**不可直接比较**。

→ 所以本计划的**第一个实质贡献**就是：**把 RNA-QA + 评测集一起开源**，给这个方向一个公开基准。

---

## 1. 原版配方（复现目标 — 严格照抄）

**2026-10-09 重读原文（arXiv 2411.08900v1 全文）后的修正版。**

| 项 | 值 |
|---|---|
| 底座 LLM | **Meta Llama-3 8B**（原文写 "the flagship Llama-3 8B model architecture"，**没写 Instruct**） |
| 序列编码器 | **RNA-FM**（冻结）；原文写它 "12 transformer layers, trained with 23 million RNAs" |
| 投影层 | **线性投影**（冻结编码器，只训投影） |
| **Stage 1 的输入模板** | `Q: <RNA><RNAHere></RNA><QuestionPrompts>` / `A: <Description>` |
| **`<RNAHere>` 是个专门的特殊 token** | 序列嵌成软提示，**插在这个位置**、"prepended to the question prompt" |
| **Stage 1 的 Q 是空的** | 原文："In stage 1 training, the question Q is left empty" |
| **Stage 1 的目标 A** | "the **full annotation from RNA Central**" |
| Stage 2 指令微调 | 10 epochs，batch=1，**wd=0.05（与 Stage 1 相同）**，warmup **200** |
| Stage 2 的模板 | 改成 **Llama-3 风格**：`###Human: …` / `###Assistant: …` |
| Stage 1 超参 | 10 epochs，batch=1，wd=0.05，warmup **2048**，AdamW β=(0.9, 0.98)，LR 1e-4→8e-5，warmup LR 1e-6，linear warmup + cosine，AMP |
| **数据选择标准** | RNAcentral 里 **indexed with "Lit Scan"** 的条目 ≈ **420,000** 条 → 收到 **≤ 1024 nt** → **407,616 条** |
| 注释构造 | 抓全部相关文献摘要 → **LDA 主题分组** → 每组单独摘要 → **再合并 + refine 一轮** |
| 指令数据 | GPT-4o-mini 拆成 **open-ended + close-ended 两种** QA（"atom-level QA pairs"） |
| 训练规模 | 初版只用 **5K RNA / 121K QA 子集** |
| 算力 | **2 × A100 40GB，两阶段合计 1 周** |
| 评测 | 评测集就是 RNA-QA 本身（AIS / D&C 两个变体各一张表） |

**两个数据集变体**（论文都报了）：`AIS`（abstract → sentence）与 `D&C`（divide-and-conquer 摘要）。

**Stage 1 那个设计要照抄** —— Q 留空，从结构上杜绝"模型绕过编码器去抄文本"这条捷径。
这是本计划里最重要的一条防坑设计。

**原版报数（原文 Table 2 / 4，ROUGE）**

| 配置 | AIS: R-1 / R-2 / R-L | D&C: R-1 / R-2 / R-L |
|---|---|---|
| RNA Sequence（裸文本） | 0.2364 / 0.0935 / 0.2037 | 0.2472 / 0.0964 / 0.2182 |
| Modality Fusion（只训投影） | 0.2239 / 0.1364 / 0.2091 | **0.0922 / 0.0393 / 0.0799** |
| RNA-GPT（完整） | 0.5031 / 0.3667 / 0.4747 | 0.4791 / 0.2690 / 0.4405 |

---

## 2. 本复现与原版的差异（**已经不是"就一处"了 —— 如实列全**）

最初设计想只改底座。实测走下来，不得不改的一共 **5 处**，按严重度排列：

| # | 项 | 原版 | 本复现 | 严重度 | 为什么必须改 |
|---|---|---|---|---|---|
| 1 | **Stage 2 训练方式** | 8B **全参**继续训 | **LoRA 43.65 M（0.533%）** | 🔴 高 | 单卡全参 8B+AdamW 需 >100 GB（权重 16 + 梯度 16 + AdamW fp32 m/v 65 + 激活），是量级问题 |
| 2 | **训练数据规模** | 5K RNA / 121K QA | 2,506 RNA / ~30K QA | 🔴 高 | 带官方注释的 RNA 只有这么多（见 §0） |
| 3 | **数据源** | RNAcentral "Lit Scan" 全量 | `RNAcentral/litsumm`（HF 镜像） | 🟡 中 | EBI 全线不通（`rnacentral.org` / `ftp.ebi.ac.uk` SSL EOF）；HF 上有官方镜像且**就是同一个 pipeline 的产物** |
| 4 | **生成器** | GPT-4o / GPT-4o-mini | **DeepSeek**（`deepseek-flash`） | 🟡 中 | 复现性 + 成本；本地 Qwen3-8B 作为离线备选 |
| 5 | **`S_GPT` 嵌入** | OpenAI `text-embedding-3-large` | **Qwen3-Embedding-8B** | 🟢 低 | 闭源、要 key、不可离线复现；换成同量级开源旗舰 |

**唯一"零代价"的那一处（也是选 Qwen3-8B 的全部理由）：**

| 底座 | hidden_size | 层数 | vocab | 投影形状 |
|---|---|---|---|---|
| Meta Llama-3 8B（原版） | **4096** | 32 | 128,256 | 基准 |
| Qwen2.5-7B（原拟用，**已否**） | 3584 | 28 | 152,064 | ✗ 要改投影 |
| **Qwen3-8B（采用）** | **4096** | 36 | 151,936 | **✓ 与基准一致，投影层零改动** |

> RNA-FM 输出 640 维，原版投影是 `640 → 4096`。Qwen3-8B 的 hidden 也是 4096，
> 所以这条投影的 in/out 维度**与原版逐位相同** —— 换底座这件事**没有污染投影层**。

**⚠️ 认准型号：`Qwen/Qwen3-8B`（2025-04 那代 Qwen3），不是本机硬盘上的 Qwen3.5 / 3.6 / 3.8。**
（本地 config.json 已核对：`hidden_size=4096, num_hidden_layers=36, vocab_size=151936`。）

**⚠️ 这一版不要用 RiNALMo** —— 改编码器就不叫复现，而且会同时 debug 两个变量。留到 v2。

---

## 3. 阶段计划与**实际进度**

### P0 · 环境与数据管线 ✅ **已完成**

- [x] 环境：`C:\ana\envs\comfyui\python.exe`（用户指定）+ `rnaqwen.env.shim` 给残废的 ROCm torch 补分布式符号
- [x] **RNA-QA 重建**：`litsumm`（官方注释）⋈ `rnacentral.1024`（序列）→ 2,506 条
- [x] **D&C 实现照抄原文三步**：`split_by_source`（按 `[PMCxxxx]` 切段）→ **LDA 主题分组** → 每组摘要 → 合并收口
- [x] 指令数据：**open-ended + close-ended 两种**（原文 §2.3 明确要两种，之前漏了一半）
- [x] **完成判据**：序列泄漏闸门 PASS（0 泄漏 / 0 拒答 / 0 空字段）

### P1 · Stage 1 模态对齐 —— 代码就绪，**待跑**

- 冻结 RNA-FM + 冻结 Qwen3-8B，只训投影（**2,625,536 参数**）
- 超参照抄；模板已按原文实现（`Q: <RNA>` + 软提示 + `</RNA>\nA: <Description>`，**Q 留空**）
- **实测速度：1.633 s/step（mean），峰值显存 29.23 GiB**
- **完成判据**：`22_zero_encoder_probe.py` —— 置零编码器后 loss 必须**明显上升**

### P2 · Stage 2 指令微调 —— 代码就绪，**待跑**

- LoRA 四个预设：`attn`(15.34M) / **`attn_mlp`(43.65M，默认)** / `attn_mlp_r8`(21.82M) / `attn_mlp_r64`(174.59M)
- 模板 `###Human: {q}\n###Assistant: {a}`，**loss 只落在答案上**
- **完成判据**：ROUGE 明显高于"裸文本输入"基线

### P3 · 评测与打平 —— 尺子已校准

- [x] ROUGE 与 Google `rouge_score` **逐样本 Δmax = 0.000e+00**
- [x] 语义指标口径锁定（含**地板修正**，见 §8.3）
- [ ] 三档消融阶梯（等训练完）
- [ ] **置零编码器消融** —— 原版没做，这是复现能超过原版的第一个点

### P4 · 开源
- 代码 + **LoRA 权重** + **RNA-QA 数据集** + **评测集** + 评测脚本
- MLSB：**承诺开源代码/权重/数据集的投稿优先给 spotlight**

---

## 4. 消融阶梯（复现成功的**真正判据**）

D&C 变体：

| 配置 | Rouge-1 | Rouge-2 | Rouge-L | F1 @ S_BERT |
|---|---|---|---|---|
| RNA Sequence（裸文本输入） | 0.2472 | 0.0964 | 0.2182 | 0.7625 |
| Modality Fusion（只训投影） | 0.0922 | 0.0393 | 0.0799 | 0.7466 |
| **RNA-GPT（完整）** | **0.4791** | **0.2690** | **0.4405** | **0.8609** |

**两个要点**

1. **复现出这条"越接越强"的阶梯 = 复现成功的真正判据**，比单看最终分数更重要。
2. **Modality Fusion 在 D&C 上塌了（0.0922）**，但在 AIS 上正常（0.2239）。
   这个异常**值得解释**，别当噪声放过 —— 写清楚了就是一个发现。

---

## 5. 风险登记（**更新：哪些真的发生了**）

| 风险 | 状态 | 实际结果 / 对策 |
|---|---|---|
| **评测集不可比** | ⚠️ 已发生 | 从头声明不可严格比较；评测集开源 |
| **静默损坏** | 🔴 **真发生了** | Qwen3-8B 5 个 shard 里 4 个 sha256 对不上、**字节数一个不差**，模型能加载只输出乱码。对策：`.part` + 原子改名 + **必须校 sha256** |
| **数据源断供** | 🔴 **真发生了** | EBI 全线不通，RNAcentral 走不了。改用 HF 镜像 |
| **数据关联错位** | 🔴 **真发生了** | NCBI 路线的 1000 条只对应 24 个 pmid（一篇挂 800 条）。**放弃该路线** |
| **数据生成烧钱** | 🟡 已量化 | 实测 ¥0.0066/条 RNA（4 次调用），全量 2,506 条 ≈ **¥16.6** |
| **DeepSeek 思考模式** | 🟡 已修 | 默认开思考且 effort=high，把输出预算烧光并**截断 JSON**。已默认关闭 |
| **投影塌缩 / 编码器被绕过** | 待验 | Stage 1 Q 留空（照抄原版）+ 置零检测当 CI |
| **QA 里复述序列** | ✅ 已挡住 | 闸门三道：精确子串 + 长核苷酸串 + 负对照 |
| **8B 全参微调带不动** | ✅ 已预判 | 走 LoRA（见 §2 差异 #1） |
| **跨 split 同源泄漏** | 🟡 已量化 | 见 §8.4 |
| **"打平"变成长期漂移目标** | ⚠️ | P3 锁定口径，接受"同量级"而非"小数点一致" |

---

## 6. 明确不做

- ❌ 不碰 RiNALMo / 结构编码器 / 修饰（v2）
- ❌ 不做全量 407,616（原版自己都没做）
- ❌ 不用 35B 模型（原版是 8B）
- ❌ 不宣称"预测免疫原性"（那是 ImmuneRNAQwen 的方向）
- ❌ 不用生成的结构训结构理解（除非过独立物理校验）

---

## 7. 【新】R 轨 / E 轨分离原则

**这条规矩是为了让"打平"这个结论不失效。**

| 轨 | 内容 | 能不能跟原版比 |
|---|---|---|
| **R 轨（复现）** | 严格照原文 + §2 的 5 处必要偏离 | ✅ **只有这一轨的数字能说"打平/没打平"** |
| **E 轨（改进）** | 任何现代技术，**每次只动一个变量** | ❌ 只能自己内部纵向比 |

**混在一起就变成"我改了一堆东西分数涨了，不知道是哪个起作用"** —— 这正是 §2 否定 RiNALMo 的同一个理由。

**E 轨已排队的改进项**（按性价比）：

1. **RNA token 压缩** ✅ 已实现（`compression.py`，三种 method + `alignment.py` 开关，默认关）
   - 实测 **880 → 213 token，快 2.55×**（序列短 4.13×，但 wall-clock 只快 2.55× —— 有固定开销）
   - **但 R 轨不启用**：17.2h 本来就在预算内，不值得引入新变量
2. **推理蒸馏**：用 `deepseek-think:deepseek-flash` 生成时连推理链一起要
3. **验真式数据筛选**：每条 QA 再让模型判定"这段文本支不支持这个答案"
4. **偏好优化**（DPO/ORPO/SimPO）：拿"好答案 vs 我们早先生成的烂答案"配对
5. **rsLoRA**（α/√r 缩放）：一行改动，已实现 `--use-rslora`

---

## 8. 【新】实测档案（每个数字都有报告文件）

### 8.1 硬件与速度

| 项 | 值 | 来源 |
|---|---|---|
| GPU | AMD Radeon 8060S (gfx1151)，**99.7 GiB** 统一内存，20 CU | `reports/stage1_timing.json` |
| Stage 1 s/step | **1.633 s**（mean；median 1.433，右偏） | 同上 |
| 峰值显存 | **29.23 GiB** —— **显存完全不是约束** | 同上 |
| grad-checkpointing 开/关 | 差 0.5%，**关掉没有收益** | 同上 |
| 长度依赖 | `s/step ≈ 0.226 + 3.13 ms/nt`，r=0.971 | 同上 |

**外推**（2,506 条 × 10 epoch = 25,060 步）：

| 配置 | 步数 | 预估耗时 |
|---|---|---|
| 全量 RNA（不压缩） | 25,060 | **11.4 h** |
| 压缩 256 latent | 25,060 | 7.2 h |
| 压缩 64 latent | 25,060 | **4.5 h** |

**⚠️ 外推假设长度分布与当前语料一致。语料一变结论就变**（若平均 1000 nt，单步涨到 3.36 s）。

### 8.2 数据

| 项 | 值 |
|---|---|
| litsumm 原始 | 4,618 条 |
| 过质量闸门（`truthful`） | 3,800 条（丢弃 818） |
| **配上序列** | **2,506 条**（丢 1,279 条 <30nt 的成熟 miRNA + 15 条重复） |
| QA 产出 | 约 30,000 组（12/记录：2 变体 × 2 类型 × 3 对） |
| 锚定度（QA 点到本条 RNA 的 ent_id） | **95.3%**（`dc` 99.3% / `ais` 91.3%） |
| JSON 解析失败率 | 0.125% |
| 序列长度 | 30–1022 nt（**RNA-FM 实测上限 1022，不是 config 写的 1024**） |

### 8.3 评测尺子

| 项 | 结论 |
|---|---|
| ROUGE 自实现 vs `rouge_score` | **Δmax = 0.000e+00**（7 个用例逐样本一致） |
| 语义 F1 的真实地板 | **BERT 0.52 / PubMedBERT 0.95**（跨域随机配对） |
| **词序全打乱仍得 0.83 / 0.97** | ⇒ 这条指标量的是**语域 + 词汇重叠，不是语义** |
| 有效动态范围 | BERT **[0.52, 1.00]** 宽 0.479 / PubMedBERT **[0.95, 1.00]** 宽 0.048 |

**⇒ `S_BERT` / `S_Pub` / `S_GPT` 三列不可横向比较**；原版表里 0.8609 vs 0.7165 的差距
很可能主要是嵌入性质差异。**ROUGE 才是主判据。**

### 8.4 评测集划分与泄漏

```
train 1,990 / val 243 / test 273   =  2,506
```

**踩过的坑：12-mer 判据本身是错的。**

| k | 真实泄漏 | 零模型基线 | 判别力 |
|---|---|---|---|
| 12 | 54,919 | 23,205 | 只高 2.4 倍 ← **生日悖论**（4^12=1.67e7 装不下 1e6 个 12-mer） |
| **20** | **5,176** | **0.3 ~ 1.3** | 高 4 个数量级 ✓ |

k=20 下按 Jaccard≥0.5 聚类：**10 个同源簇 / 21 条记录 / 跨 split 只有 3 对**：
`RNU1-2↔RNVU1-7`、`hsa-mir-425↔MIR425`、`rno-mir-15a↔bta-mir-15a`。

### 8.5 成本

| 项 | 实测 |
|---|---|
| 单价 | **¥0.0066 / 条 RNA**（4 次调用：2 变体 × 2 类型） |
| 全量数据生成 | **¥16.6** |
| 若加"验真筛选"一轮 | +40% |

**教训**：DeepSeek V4 系**默认开思考模式**（effort=high），实测单次输出 1032 tok 里绝大部分是
reasoning，而且大量调用顶到 `max_tokens` 上限 → **JSON 被截断**。关掉后单次输出降到 257 tok。

---

## 9. 下一步（按依赖排序）

1. **Stage 1 开跑**（11.4 h）→ `22_zero_encoder_probe.py` 验收（置零必须让 loss 上升）
2. **Stage 2 LoRA** → 三档消融出阶梯
3. **P3 评测**：两套指标 + 三档阶梯 + **置零消融**（原版没做的那个）
4. **E 轨**：token 压缩 / 推理蒸馏 / 验真筛选 / 偏好优化（**一次一个变量**）
5. **P4 开源**

---

## 附录 A · 已核实的资源

| 资源 | 状态 |
|---|---|
| RNA-GPT 论文 | arXiv **2411.08900**（2024-10-29）· MLSB Workshop **NeurIPS 2024** |
| RNA-GPT 代码 | ❌ **无**（GitHub 组织 `RNA-GPT` 只有 `RNA-GPT.github.io`，★3） |
| RNA-GPT 权重 | ❌ **无**（HuggingFace 无命中） |
| RNA-QA 数据集 | ❌ **无**（HF datasets 搜 `RNA-QA` 无命中） |
| RNA-FM | ✅ HF `multimolecule/rnafm`；**实测 640 维，可用上限 1022 nt** |
| Qwen3-8B | ✅ ModelScope `Qwen/Qwen3-8B`（16.38 GB，5 shard，sha256 已校） |
| **RNAcentral 镜像** | ✅ HF `multimolecule/rnacentral.1024`（3.7 GB，10M~100M 条） |
| **官方文献注释** | ✅ HF `RNAcentral/litsumm-v1.5`（4,618 条，带质量标记） |
| Qwen3-Embedding-8B | ✅ ModelScope（14.1 GB）—— 替代 `S_GPT` |
| MLSB 站 | `https://www.mlsb.io/`；**non-archival**；**承诺开源者优先 spotlight** |

## 附录 B · 相邻工作（MLSB 2025 accepted，62 篇里相关项）

```
#31  Improving RNA 3D Structure Prediction via Language Model-Augmented AlphaFold 3   ← 最对口
#55  Curriculum-Augmented GFlowNets For mRNA Sequence Generation
#28  Rewriting protein alphabets with language models
#14  g-DPO: Scalable Preference Optimization for Protein Language Models
#26  ProteomeLM · #56 ProFam · #3 Self-Distillation FT · #59 SAE on Antibody LMs
```

## 附录 C · 后续（v2，本计划之外）

- 换 RiNALMo（HF `multimolecule/rinalmo-giga`，hidden=1280，33 层，**max_position_embeddings=1024**）
- **长度问题**：2,013 nt > 1024 → 先读 `RIBOSPAN`（arXiv 2026-08，长上下文 RNA FM）；备选 3Di / 重叠窗口
- 加结构这一路；加 m6A / m5C
- 词表扩展 + 继续预训练（**ProstT5 路线**：Heinzinger 等，NAR Genomics and Bioinformatics 2024，DOI `10.1093/nargab/lqae150`）
- 若走免疫方向：标签来自物理流水线（TorusFold），**不是实验真值** —— 对外口径必须写清


---

# ⚠️ 转向声明（2026-10-10）

**本文件描述的路线已被放弃。** 新的计划见 `PLAN_v2.md`。

放弃的是「冻结编码器 + 投影 → 冻结 LLM」这条软提示路线。原因不是做不下去，
而是**逐条测出了它的天花板**（见 `PLAN_v2.md` §1）：

- 投影没在丢信息（去均值后可分性 -0.05）
- 换掉学习式压缩器不改变逐 token 剖面
- val_loss 与 grounding 反向（两次独立实验）
- 去各向异性不伤性能但也没解决依赖衰减
- **根因是冻结的 LLM 永远学不到序列理解**，换任何 Stage 1 目标都修不了

四条外部证据指向同一结论：ChatMultiOmics（arXiv 2412.19191，LoRA 在生物序列上接近随机）、
MAPO（arXiv 2605.27741，late-stage modality collapse）、Moirain（arXiv 2605.23961）、
以及我们自己的 Δzeroed 与逐 token 测量。

**本文件保留作为被证伪路线的完整记录** —— 它的价值在于证明了"为什么不能这么做"。
新路线：RNA 走原生文本 token → 序列继续预训练（可选）→ 指令微调。
