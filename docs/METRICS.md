# 评测口径（锁定版）

> **这份文件定义尺子。任何报告里的数字都必须能被这份文件复现。**
> 改口径 = 换尺子，必须同步改 `protocol` 字段并说明为什么。

---

## 1. 两套指标（照抄原论文）

| 指标 | 定义 | 实现 |
|---|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | F-measure | `src/rnaqwen/eval/rouge.py` |
| Semantic Precision / Recall / F1 | 句子级 max-cosine | `src/rnaqwen/eval/semantic.py` |

三种嵌入对应原论文的 `S_BERT` / `S_Pub` / `S_GPT`：

| 记号 | 原版 | 本复现 | 本地路径 / 调用 |
|---|---|---|---|
| `S_BERT` | `bert-base-uncased` | 同 | `models/bert-base-uncased` |
| `S_Pub` | PubMedBERT | 同 | `models/pubmedbert` |
| `S_GPT` | OpenAI `text-embedding-3-large` | **`Qwen3-Embedding-8B`**（同量级开源旗舰） | `models/Qwen3-Embedding-8B`，`build_embedder("qwen3emb")` |

> **⚠️ `S_GPT` 这一列是替换过的，不能跟原报数逐位对齐。**
> 原版用 OpenAI 的闭源嵌入（要 key、不可离线复现）。换成 Qwen3-Embedding-8B 之后，
> 这一列**只能跟自己比**（横向比较三档消融），不能跟论文表格里的数比。
>
> Qwen3-Embedding 的调用规约有两个坑：
> 池化取**最后一个非 padding token**（所以要**左 padding**），
> 查询侧加指令前缀 `Instruct: {task}\nQuery: {text}`、文档侧不加。
> 实现见 `HFEmbedder(pooling="last", instruction=...)`。

---

## 2. ROUGE 的具体口径

* **分词**：`re.findall(r"[a-z0-9]+", text.lower())` —— 对齐 Google `rouge_score` 的 `_split_into_words`
* **ROUGE-N**：n-gram 计数，`F = 2PR/(P+R)`
* **ROUGE-L**：LCS，`F = (1+β²)PR/(R+β²P)`
  * **默认 β = 1.0**（`rouge_score` 口径）
  * 备选 β = 1.2（原版 ROUGE-1.5.5 / pltrdy `rouge` 的默认值）—— 报告里**必须写明用了哪个**
* **多参考**：逐参考算 F 后取 max
* **不做 stemming**

**对数验证**：`scripts/03_eval_selfcheck.py` 的 A 关拿自实现跟 `rouge_score` 逐样本对数，
7 个用例（identical / paraphrase / partial / unrelated / empty / punctuation / numbers）
最大偏差 **0.000e+00**。改了口径就要重跑这一关，不一致就不许上量。

---

## 3. 语义指标的具体口径

* **粒度**：句子级。切分规则 = 换行 + `[.!?]` 后跟空白；丢弃长度 < 2 的碎片
* **相似度**：句向量 L2 归一化后的点积（= 余弦）
* **句向量**：最后一层 hidden state 的 **mean pooling**（attention mask 加权），
  **不**做 BERTScore 的 baseline rescaling
* **P/R/F1**：
  ```
  P  = mean_i max_j cos(gen_i, ref_j)      （gen 每句对 ref 所有句取最大，再按句数平均）
  R  = mean_j max_i cos(gen_i, ref_j)
  F1 = 2PR/(P+R)
  ```
  逐样本算完再对整个数据集取平均

### ⚠️ 这条指标有一个**必须先说清楚的缺陷**（实测数字，`reports/semantic_floor.json`）

先纠正一个我一开始搞错的直觉：**"无关文本"的分数高度依赖语域**。
RNA vs 蛋白质晶体学同属生物医学语域、共享大量词根，所以靠得近（BERT 0.77）；
换成真正的跨域配对（RNA vs 经济学 / 法律 / 菜谱），同一段文本立刻掉到 **0.45**。
**做无关对照必须按语域设计，否则地板会被抬高 0.3，足以把结论带反。**

真正的地板与动态范围：

| | bert-base-uncased | pubmedbert |
|---|---|---|
| 上界（identical） | 1.0000 | 1.0000 |
| 跨域无关随机配对（文级） | **0.5205** ± 0.089 | **0.9518** ± 0.008 |
| 同域不同事实 | 0.7102 | 0.9682 |
| **词序全打乱、词汇全对** | **0.8289** | **0.9718** |
| 纯乱码 | 0.2483–0.3913 | 0.9170–0.9250 |
| **有效动态范围** | **[0.52, 1.00]，宽 0.479** | **[0.95, 1.00]，宽 0.048** |

**最要命的一条**：把每个句子内部**词序完全打乱**（句子彻底不成话，
词 3-gram Jaccard 只剩 0.0035），BERT 仍给 **0.8275**、PubMedBERT 仍给 **0.9679**。
也就是说这条 max-cosine 指标**量的是"领域 / 语域 + 词汇重叠"，不是"句子到底说了什么"**。
PubMedBERT 上"词汇全对但语义全错"（0.968）与"完全无关的领域"（0.952）
甚至和"纯乱码"（0.92）都分不开 —— 机制是各向异性：
PubMedBERT 跨域句级 cosine 已经 0.9466、同域 0.9684，**可分性只有 0.022**（BERT 是 0.241）。

（权重忠实性已排除：`reports/embedder_weight_verification.json` ——
从 ModelScope 重下上游 `pytorch_model.bin`，sha256 与官方声明逐位相同；
本地 safetensors 与它 **199/199 张量 bitwise 相同**，前向 `max|Δ| = 0.000e+00`。
所以 0.95 这个地板是**真模型的性质**，不是文件坏了。）

**读数的正确姿势**：

1. 报 semantic F1 时**必须同时报同语料、同嵌入下的地板**（跨域随机配对）与上界；
2. `S_BERT` / `S_Pub` / `S_GPT` 三列**不可横向比较** ——
   原版表里 `0.8609 vs 0.7165` 的差距，很可能主要是**嵌入性质差异**，不是"BERT 语义更准"；
3. 在 BERT 口径下，原报的 0.8609 高出地板 0.3404，**占窗口的 71%**，是有信息量的；
   但同一个数放在 PubMedBERT 口径下**低于地板 0.09**，等于"完全跑题"。
   → 所以**必须两套指标一起看**，ROUGE 才是主判据。

**自检脚本的阈值也要跟着改**：`03_eval_selfcheck.py` 里那条
`unrelated_lt_0.95` 是在 BERT 上标定的，在 PubMedBERT 上**数学上不可能通过**
（地板就是 0.9518）。这条 FAIL 不是 bug，是阈值不可移植 —— 已改为按嵌入分档。

---

## 4. 消融阶梯的三个配置（PLAN §4）

必须复现出这条"越接越强"的阶梯，**这比单看最终分数更能说明复现成功**。

| 配置 | 输入方式 | 脚本 |
|---|---|---|
| RNA Sequence（裸文本） | 序列当字符串塞进 prompt | `30_infer.py --mode rna_sequence` |
| Modality Fusion（只训投影） | RNA-FM → 投影 → LLM（无 Stage2） | `30_infer.py --mode modality_fusion` |
| RNA-GPT（完整） | 投影 + Stage-2 LoRA | `30_infer.py --mode rna_gpt` |

原版报的 D&C 数值（作为方向参照，**不是**可比分数）：

```
RNA Sequence      0.2472 / 0.0964 / 0.2182   F1@S_BERT 0.7625
Modality Fusion   0.0922 / 0.0393 / 0.0799   F1@S_BERT 0.7466   <- 在 D&C 上塌了（AIS 上正常 0.2239）
RNA-GPT           0.4791 / 0.2690 / 0.4405   F1@S_BERT 0.8609
```

> `Modality Fusion` 在 D&C 上塌方、在 AIS 上正常 —— 这个异常要**解释**，不要当噪声放过。
> 写清楚了就是一个发现。

---

## 5. 报告模板（任何数字都必须带这四条声明）

```
1. 评测集为**重建**（原版评测集未开源），绝对分与原报数**不可直接比较**。
2. 原版训练/报数用的是 5K RNA / 121K QA 子集，不是全量 407,616。
3. 数据生成器与嵌入都有替换：
   - 注释/QA 生成：原版 GPT-4o / GPT-4o-mini → 本复现 **DeepSeek**（`deepseek-chat`），
     本地 Qwen3-8B 作为可离线复现的备选；
   - S_GPT 这一列：OpenAI text-embedding-3-large → **Qwen3-Embedding-8B**，
     该列**不可**与原报数直接比较。
4. 底座 LLM：Llama-3 8B → Qwen3-8B（hidden 均为 4096，投影层零改动，见 PLAN §2）。
```
```
