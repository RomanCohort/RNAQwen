# 复现日志

> 每条都记 **做了什么 / 验收到什么程度 / 踩了什么坑**。
> 只写有证据的部分；没验证的不许写成"完成"。

---

## 2026-10-08 · 环境勘察与启动

### 机器事实（实测，不是推断）

| 项 | 值 | 怎么测的 |
|---|---|---|
| GPU | AMD Radeon 8060S (gfx1151)，**99.7 GB** 统一内存，20 CU | `torch.cuda.get_device_properties(0)` |
| torch | `2.12.0a0+rocm7.13.0a20260313` | 装于 `C:\ana\envs\comfyui` |
| **torch.distributed** | **`is_available() == False`** —— 构建里没编 | `import torch.distributed as dist` |
| transformers | 5.9.0（装进本项目 venv） | `pip install` |
| 解释器 | **`C:\ana\envs\comfyui\python.exe`**（用户指定用 comfyui 环境） | 见下方"环境变更" |

### 环境变更记录（装进 comfyui 环境，便于回滚）

按用户要求，本项目直接用 **comfyui 环境**跑，不另起环境。为此装进 `C:\ana\envs\comfyui`：

```
transformers          4.57.6  ->  5.9.0        （multimolecule 的硬需求：
huggingface-hub       0.36.2  ->  1.33.0        它 import transformers.utils.output_capturing，
hf-xet                 1.5.2  ->  1.7.0         4.57 里没有这个模块）
新增：peft 0.21.2 · accelerate 1.15.0 · sentence-transformers 6.1.0 · rouge-score 0.1.2 · nltk 3.10.3
```

`.venv`（在项目目录下，`--system-site-packages` 继承 comfyui 的 torch）保留为备用，
包集合与 comfyui 一致，不动 comfyui。**若 ComfyUI 的自定义节点依赖 transformers 4.x，用 `.venv` 跑本项目。**

### 坑 1：`import multimolecule` 在本机直接不可行

`danling`（MultiMolecule 的运行时依赖）在 import 阶段无条件引用
`torch.distributed.distributed_c10d`、`torch.distributed.tensor`、`dist.ReduceOp`，
而这个 torch 构建**没有分布式后端**。四级连锁失败：

```
ImportError: cannot import name 'dcp' from 'danling.runners.checkpoints.torch_distributed'
ModuleNotFoundError: No module named 'torch._C._distributed_c10d'
AttributeError: module 'torch.distributed' has no attribute 'ReduceOp'
ModuleNotFoundError: No module named 'torch.distributed.tensor.device_mesh'
```

**解法**：`src/rnaqwen/env/shim.py` —— 把缺失符号补成占位符，
并在 `sys.meta_path` **末尾**挂一个 permissive finder（真模块优先，缺的才伪造）。
现在 `from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer` 通过（transformers 5.9.0）。

### 坑 2：网络 —— HF 走得通但慢到不可用，EBI 完全不通

| 通道 | 实测 |
|---|---|
| huggingface.co CDN（`us.aws.cdn.hf.co`） | **0.06 MB/s** |
| hf-mirror.com | 0.06 MB/s（回源到 hf.co，没用） |
| **ModelScope（`cdn-lfs-cn-1.modelscope.cn`）** | **11.2 MB/s** |
| NCBI eutils | 通（esearch / efetch / elink 全 200） |
| rnacentral.org / ftp.ebi.ac.uk / europepmc.org | **SSL EOF / ReadTimeout，全不通** |

推论：
* 权重一律走 ModelScope（`scripts/00b_download_modelscope.py`，并发 + 断点续传）。
* 数据源从 RNAcentral 换成 **NCBI nuccore + pubmed**（差异已写进 README §5）。
* 另外：`huggingface_hub` 走 hf-mirror 会抛
  `FileMetadataError: Distant resource does not seem to be on huggingface.co`
  —— 镜像不给 HEAD 元数据头。别用镜像。

### 坑 3：`urllib`/`requests` 的下载超时用法不一样，且 CDN 会「滴水」

- `urlopen(req, timeout=(20, 120))` 抛
  `TypeError: 'tuple' object cannot be interpreted as an integer`
  —— 元组超时是 `requests` 的用法。四个下载任务全灭了一次。
- 换回 `requests` 之后，**读超时给太长（30s）等于卡死没有出口**：
  CDN 会让连接保持存活但几乎不吐字节，进程静默挂在 0.00 MB/s 上半小时。
  最终解法是 `timeout=(15, 10)`（连接 15s / **读 10s**）＋一个**速率看门狗**
  （45 秒内增量 < 200 KB 就主动断开，带 `Range` 续传重试）。
- 另一个坑：**`Get-ChildItem` 读到的文件大小是缓存的**，明明在下载却一直显示
  旧值，害得我以为卡了三次。改用 Python 的 `os.path.getsize` 才看到真实速率
  （5 个 shard 各 ~2 MB/s，合计 ~10.7 MB/s）。**别用 PowerShell 看下载进度。**

### 坑 4：磁盘写满（Qwen3-8B 下到一半全灭）

`OSError [Errno 28] No space left on device` —— 5 个 shard 同时卡在同一位置反复重试。
C 盘当时只剩 **4.4 GB**（共 918 GB，本来就快满了）。清理：

| 动作 | 释放 |
|---|---|
| 删掉多余的 conda clone `C:\ana\envs\rnaqwen`（我们有 `.venv`，用不到它） | 6.09 GB |
| 清掉 HF 缓存里的 `.incomplete` 残片 | 0.15 GB |
| 删掉 pubmedbert 里冗余的 `pytorch_model.bin`（已与 safetensors 逐位比对一致） | 0.44 GB |

清理后 11.16 GB 可用，15.3 GB 的权重才下得完。
**教训：这台机器 C 盘长期紧张，下大模型前先看 `Get-PSDrive C`。**

### 坑 5：HF 不是慢，是**单连接**被限速

`aria2c -x16 -s16 -k1M` 从 huggingface.co 拉 RNA-FM 的 398 MB：

```
25 秒下完（约 16 MB/s），而单连接只有 0.06 MB/s
```

所以真正的结论不是"HF 不能用"，而是**单连接被掐到 0.06 MB/s，多连接能跑满**。
（ModelScope 单连接就能到 2–11 MB/s，但同样会在并发下载时互相挤。）
以后再遇到"某个源很慢"，先试 `-x16`，别急着换源。

### 坑 6：下载并发多了会互相拖死

5 个下载任务 + 2 个 subagent 同时拉，结果是**所有任务一起停**（每个都显示 0.00 MB/s）。
最后按"一个源最多 5 条流、别再多开进程"来跑才稳定。

### 模型就位情况

| 模型 | 来源 | 状态 |
|---|---|---|
| `Qwen/Qwen3-8B` | ModelScope | 下载中（15.3 GB / 5 shard） |
| `multimolecule/rnafm` | HF（aria2 -x16） | 398 MB，即将完成 |
| `google-bert/bert-base-uncased` | ModelScope `google-bert/bert-base-uncased` | ✅ 441 MB，sha256 全对 |
| `microsoft/BiomedNLP-BiomedBERT-…` | ModelScope | ✅ 见下方说明 |

**PubMedBERT 的一个事实**：上游仓库（HF 和 ModelScope 一致）**根本没有
`model.safetensors`，也没有 `tokenizer.json`** —— 只有 `pytorch_model.bin` + `vocab.txt`。
所以本地这两个文件是**生成的**（safetensors 由 .bin 导出，tokenizer.json 由 vocab.txt 构造），
并已验证：safetensors 与 .bin 前向输出**逐位一致（max|Δ| = 0.0）**。
→ 报告里要写明这一点，别让人以为是从上游拿的原文件。

### 坑 7（最贵的一个）：**静默损坏** —— 大小完全正确，内容是坏的

Qwen3-8B 下完 16.38 GB，`safetensors` 加载成功（399/399 张量全部到位），
但生成出来是乱码：

```
prompt: "Name the four RNA nucleotides, separated by spaces."
sdpa  / bfloat16 -> ',es numpy React sys React matplotlib React React React ...'
eager / bfloat16 -> ',es numpy React sys React matplotlib React React ...'
eager / float32  -> ',es matplotlib React React React React React React ...'
```

三种 attention 后端 × 两种精度**输出同样的乱码** → 不是算子问题，是**权重本身坏了**。

`scripts/00c_verify_models.py` 拿本地 sha256 跟 ModelScope 声明的值对比：

```
model-00005-of-00005.safetensors  1244659840  1244659840  OK  20c2d636... == 20c2d636...
model-00001-of-00005.safetensors  3996250744  3996250744  BAD 8f63b006... != 31d6a825...
model-00002-of-00005.safetensors  3993160032  3993160032  BAD bd732b5f... != 5991236c...
model-00003-of-00005.safetensors  3959604768  3959604768  BAD 52d91d30... != c5185c47...
model-00004-of-00005.safetensors  3187841392  3187841392  BAD 41b73ee3... != b5ee7de7...
```

**4/5 个 shard 坏了，而字节数一个不差。**

成因（复盘）：
1. 这些文件被**多轮续传**拼过 —— 磁盘写满中断 → 换进程重下 → 再断 → 再续；
2. 中途还出现过**两个进程同时写同一个文件**（先起了 26/27 号任务，又起了 29/30 号任务，旧任务没死透）。

`Size` 校验**完全挡不住这种错误**。修复：
* `00b_download_modelscope.py` 改成先写 `.part`、**完整拿到后原子改名**；
* 下完必须跑 `00c_verify_models.py` 对 sha256 —— **不看 sha256 不算下完**。

> 这条要写进开源仓库的 CONTRIBUTING：权重只做 size 校验是危险的，
> 静默损坏的模型会加载成功、跑得动、只是输出垃圾，排查成本极高。

### 坑 8：`aria2c` 会把文件**预分配**成完整大小

aria2 先按最终大小占位，进度只记在 `.aria2` 控制文件里。
我用 `Get-ChildItem` 看到 398.12 MB 就以为下完了，结果 `safetensors` 报
`invalid JSON in header: EOF while parsing a value`。
**判断下载是否完成：看控制文件 / 校验和，不看文件大小。**

### 型号核对：Qwen3-8B 的投影层确实零改动

已下到本地的 `models/Qwen3-8B/config.json`：

```
hidden_size = 4096       num_hidden_layers = 36
vocab_size  = 151936     intermediate_size = 12288
```

与 Llama-3-8B 的 hidden=4096 **逐位相同** → RNA-FM 那条 `640 → 4096` 线性投影
**与原版维度一致，不用改**。这就是选 Qwen3-8B（而不是 Qwen2.5-7B 的 3584）的理由。

> 注意区分：本机 `C:\qwen-api\models\` 下的 `Qwen3.8-27B-*` / `Qwen3.6-35B-A3B-*` 是**别的系列**，
> 不能当底座。复现用的是原始 Qwen3 家族、8B 规模：`Qwen/Qwen3-8B`。

---

## PLAN §7.3 · 评测脚本（**已通过**）

`scripts/03_eval_selfcheck.py`，报告 `reports/eval_selfcheck_bert.json`。

### A 关：ROUGE 对数（自实现 vs Google `rouge_score`）

7 个用例：identical / paraphrase / partial / unrelated / empty / punctuation / numbers

```
-> 最大偏差 0.000e+00（阈值 1e-09）：一致
```

**逐样本完全一致。尺子准了。**

### B 关：语义指标不变量（嵌入 = bert-base-uncased）

```
[high] identical    P=1.0000 R=1.0000 F1=1.0000
[high] paraphrase   P=0.9396 R=0.9396 F1=0.9396
[mid ] partial      P=0.9366 R=0.8748 F1=0.9046
[low ] unrelated    P=0.8004 R=0.8321 F1=0.8159
[ok] identical_ge_0.999 / paraphrase_gt_partial / partial_gt_unrelated / unrelated_lt_0.95
```

### B/C 关：PubMedBERT（`reports/eval_selfcheck_pubmedbert.json`）

A 关同 BERT（7 用例 Δmax = 0.000e+00）。B 关**没过**，C 关过了：

```
identical    P=1.0000 R=1.0000 F1=1.0000
paraphrase   P=0.9938 R=0.9938 F1=0.9938
partial      P=0.9950 R=0.9857 F1=0.9903
unrelated    P=0.9701 R=0.9669 F1=0.9685      <- 比 BERT 的 0.8159 还高

[ok  ] identical_ge_0.999 / paraphrase_gt_partial / partial_gt_unrelated
[FAIL] unrelated_lt_0.95
```

**这个 FAIL 不是 bug，是阈值本身不适用于 PubMedBERT。**
`unrelated_lt_0.95` 是在 BERT 上标定的；而 PubMedBERT 的跨域地板是 0.9518
（§7.3b 实测），**它在数学上不可能满足这条断言**。
C 关同理可信度有限：

```
perfect_100pct   1.0000/1.0000/1.0000   F1=1.0000
half_50pct       0.4800/0.4583/0.4800   F1=0.9907
one_third        0.4800/0.4583/0.4800   F1=0.9907
bare_sequence_like 0.0000/0.0000/0.0000 F1=0.8889   <- 纯 ACGU 乱串也有 0.89
empty            0.0000/0.0000/0.0000   F1=0.0000
-> ROUGE-1 随答案完整度单调：是
```

ROUGE 阶梯正常（1.00 → 0.48 → 0.00）；
但 semantic 那列「50% 答案 0.9907 / 乱码 0.8889」的区分度基本为零。

### C 关：消融阶梯的响应方向

```
variant                   R-1      R-2      R-L   F1@bert
perfect_100pct         1.0000   1.0000   1.0000   1.0000
half_50pct             0.4800   0.4583   0.4800   0.9370
one_third              0.4800   0.4583   0.4800   0.9370
bare_sequence_like     0.0000   0.0000   0.0000   0.2633
empty                  0.0000   0.0000   0.0000   0.0000
-> ROUGE-1 随答案完整度单调：是
```

### 🔎 第一个发现（**后来被部分推翻**）：语义 F1 的"地板很高"

「无关的两段文本」在句子级 max-cosine 下拿到 **F1 = 0.8159**。
当时据此推断分辨率被压缩在 **[0.82, 1.00]**。

> ⚠️ **这个推断后来被证伪了，见 §7.3b。** 0.8159 之所以那么高，是因为
> RNA 与蛋白质晶体学**同属生物医学语域**。换成真正的跨领域配对（RNA vs 经济学/法律/菜谱），
> 同一把尺子立刻掉到 **0.45 左右**。§7.3b 的结论比这里更严重，但方向不同：
> 地板在 BERT 上其实只有 0.52（不是 0.82），真正的病在 PubMedBERT（0.95）。
> 原文保留，记录判断的演进过程。

---

## PLAN §7.3b · semantic F1 的**分辨率地板**（`scripts/03b_semantic_floor.py`）

报告：`reports/semantic_floor.json`（bert + pubmedbert 各一遍，n=2000，seed 20261008）。

### 先纠一个前提错误

原来把 `unrelated`（RNA vs 蛋白质晶体学）的 **0.8159** 当成"完全无关"的下限。
**这是错的**：两者同属生物医学语域、共享大量词根，句子向量本来就靠得近。
按领域两两配（RNA vs 经济学 / 法律 / 菜谱），立刻掉到 **0.45 左右**。

| 无关对（8句 vs 8句） | F1@bert | F1@pubmedbert |
|---|---|---|
| rna vs crystallography（都属生物医学） | 0.7663 | 0.9640 |
| rna vs economics | 0.4490 | 0.9592 |
| rna vs legal | 0.4649 | 0.9498 |
| rna vs cooking | 0.4549 | 0.9522 |
| crystallography vs legal | 0.4771 | 0.9426 |
| economics vs cooking | 0.5374 | 0.9585 |

→ 教训：**"无关"必须按语域设计对照，不能顺手拿两个技术领域当无关。**
语域相邻的"无关"会把地板抬高 0.3，足以把一个结论带反。

### 地板 / 随机基线 / 上界

| | bert-base-uncased | pubmedbert |
|---|---|---|
| 上界（identical） | **1.0000** | **1.0000** |
| 地板 A：跨域无关随机配对 | **0.5205** (std 0.089) | **0.9518** (std 0.008) |
| 地板 B：同域不同事实（分子生物学） | **0.7102** | **0.9682** |
| 地板 C：词序全打乱、词汇全对 | **0.8289** | **0.9718** |
| 地板 D：纯乱码 | 0.2483–0.3913 | 0.9170–0.9250 |
| 句级随机配对（"瞎猜"） | 0.5357 (std 0.147) | 0.9514 (std 0.015) |
| 同域句级随机配对 | 0.7236 | 0.9683 |
| **有效动态范围**（跨域地板 → 1.0） | **[0.52, 1.00]，宽 0.479** | **[0.95, 1.00]，宽 0.048** |

### 最严重的一条：这条指标量的**不是语义**

把每个句子内部的词序全打乱 —— 句子彻底不成话，词 3-gram Jaccard 只剩 **0.0035**
（几乎没有保留任何连续词组），但：

```
bert-base-uncased    乱序 F1 = 0.8275   （原句 1.0000）
pubmedbert           乱序 F1 = 0.9679   （原句 1.0000）
```

**PubMedBERT 上，"词汇全对但语义全错"（0.968）与"完全无关的领域"（0.952）
基本无法区分，和纯乱码（0.92）也差不了多少。**
它对"领域/语域 + 词汇重叠"敏感，对"句子究竟说了什么"不敏感。

### `0.8609` 在不同嵌入上的含义完全不同

| embedder | 地板 | 0.8609 的位置 |
|---|---|---|
| bert | 0.5205 | 高出地板 +0.3404，占可用窗口 **71%** —— 有信息量 |
| pubmedbert | 0.9518 | **低于地板 0.0909** —— 在这个尺度上等于"完全跑题" |

同一把尺子的同一个读数，在两种嵌入上一个算好、一个算跑题。
**说明分母（地板）比分子更决定读数**，也说明 `S_BERT` 与 `S_Pub` 的分数
**不可横向比较** —— 原版表里 0.8609 vs 0.7165 的差距，很可能主要来自嵌入本身的性质，
而不是"BERT 语义更准"。

> 0.8609 是原论文在其**自有评测集**上的数，本地是自建小语料，绝对值不可直接比较；
> 这里只用它定位"同一把尺子在本地语料上的零点"。

### 机制：表示各向异性

| | 跨域句级 cos | 同域句级 cos | 跨/同域差 | 各向异性 |
|---|---|---|---|---|
| bert | 0.4844 | 0.7250 | **+0.2406** | 0.7281 |
| pubmedbert | 0.9466 | 0.9684 | **+0.0218** | 0.9752 |

PubMedBERT 的句向量挤在一个极窄的锥里（任意两句 cosine 就有 0.95），
max-cosine 于是无论如何都被抬到 0.95 以上 —— **地板高是几何必然，不是评测集的问题。**
它的同/跨域可分性比 BERT 低一个数量级（0.022 vs 0.240），
这才是它无法分辨对错的根本原因。

### 落在报告口径上的三条

1. **不能单独用 semantic F1 判复现好坏。** 必须同时报 ROUGE + 同语料的随机/无关基线。
2. **`S_BERT` 与 `S_Pub` 不可跨列比较**（分母不同）。原版表把两种嵌入并排摆，
   容易被读成"BERT 上语义更好"，实际不是。
3. **报 PubMedBERT 的 semantic F1 时量级要特别小心** ——
   0.95 以下没有信息，0.97 与 0.99 之间的差别基本是噪声。

---

## PLAN §7.3c · PubMedBERT 权重忠实性验证（`scripts/00d_verify_embedder_weights.py`）

报告：`reports/embedder_weight_verification.json`。

**为什么必须做**：`pubmedbert` 目录下的 `model.safetensors` 与 `tokenizer.json` 是**本地生成的**
（上游只给 `pytorch_model.bin` + `vocab.txt`），而这里踩过一次
"字节数完全正确、内容全是坏的"的静默损坏（坑 7）。所以"PubMedBERT 地板 0.95"
必须先排除"权重坏了"才成立。

从 ModelScope 重下上游 `pytorch_model.bin`（440,474,434 B）：

```
sha256 = ad7bbb66376cfd6b2db3447192b034efe016337cbef135c35c411fd61b13c193
       == ModelScope 声明值（逐位相同）
上游 208 张量（含 MLM 头 9）→ 去 `bert.` 前缀、剔除 cls.* 后共同键 199
bitwise 相同: 199/199    shape 不一致: 0    max|Δ| = 0.000e+00
前向: last_hidden_state 逐位相同: True      max|Δ| = 0.000e+00
句向量余弦矩阵 max|Δ| = 0.000e+00
[OK] 权重忠实
```

**结论：本地 safetensors 是上游权重的忠实无损导出。地板 0.95 是真模型性质，不是损坏。**

> 顺带修掉一个自己犯的错：第一版比对脚本写成
> `missing, unexpected = m_up.load_state_dict(sd_up, strict=False)`，
> **把返回值丢掉了**，于是 `m_up` 还是随机初始化，前向 max|Δ| = 1.49e+01，
> 差点得出"权重坏了"这个完全相反的结论。
> `load_state_dict` 确实是 in-place 的，但**必须断言 `missing_keys` 为空**才能拿去比前向 ——
> 现在脚本里有 `assert not report.missing_keys`。

> 另一个易踩的坑：`model.safetensors` 是从 `AutoModel`（裸 encoder）导出的，**不含 MLM 头**。
> 用 `AutoModelForMaskedLM` 加载会报 `cls.predictions.*` MISSING，然后拿**随机初始化**的
> MLM 头输出垃圾词（实测：`The patient was diagnosed with [MASK] cancer.` → `vast/grand/looks`，
> 而真 BERT 给的是 `breast/lung/liver`）。
> **别用 MLM 探针验 encoder 权重** —— 会得出错误结论。semantic 指标只吃 encoder 输出，不受影响。

---

## PLAN §7.1 · RNA-FM 探针（**已通过**）

`reports/rnafm_probe.json`（CPU 上跑的；GPU 另见下方"待查"）。

```
config : max_position_embeddings=1026  hidden=640  vocab=28
buffer   embeddings.position_ids: shape=(1, 1026)
[ok]  demo_small             nt=  81  ids=(1,  83)  out=(1,  83, 640)
[ok]  demo_yeast_trna_phe    nt=  76  ids=(1,  78)  out=(1,  78, 640)
[ctx] 1022 nt -> ids (1, 1024)
```

**输出维度 640，逐位对上 config 的 `hidden_size`。** tokenizer 会额外加 BOS/EOS 两个特殊 token。

### ⚠️ 一个和 PLAN 假设不一致的实测结果：**上下文其实是 1022 nt，不是 1024**

PLAN §3 P0 的过滤标准写的是"序列长度 30–1024 nt（RNA-FM 上下文 1026 − 2）"。
`scripts/01b_probe_rnafm_limit.py` 逐长度逼边界，结果：

```
  nt  tokens  result
1022    1024  OK   out=(1, 1024, 640)
1023    1025  FAIL IndexError: index out of range in self
1024    1026  FAIL IndexError: index out of range in self
-> 可用上限：1022 nt
```

`config.max_position_embeddings=1026` 和 `position_ids` 缓冲区的 1026 都对得上，
但**真正生效的 token 上限是 1024**（1024 个 token 能过，1025 就炸）。

所以：**数据过滤上限要从 1024 nt 收到 1022 nt。**
（当前这 50 条最长 966 nt，没踩到；上量到 1K/121K 时必须改，
否则会有一批样本在 encoder 里直接抛 IndexError。）

### GPU 上前向崩过一次 —— 是**暂时的**，已恢复

同一段探针 `--device cuda` 下崩在 `at::native::to` → `empty_strided_cuda` →
`c10::cuda::SetDevice`（HIP 侧抛异常，连 `torch.zeros(8, device='cuda')` 都过不去）。
CPU 上一切正常，说明不是权重问题。

排查过程：
1. 先怀疑 GPU 被别的进程占着 —— `Get-Process` 显示没有大进程，可用内存 52 GB，排除；
2. `aria2c` 式的环境因素排除（两个解释器都崩）；
3. **等了约 20 分钟后原样重试，直接通过**（`CUDA OK`），探针在 GPU 上也跑通了。

结论：gfx1151 + ROCm 7.13 nightly 偶发地会把设备卡住，**重启内核态的驱动状态需要时间**。
→ 遇到 `c10_cuda_check_implementation` / `SetDevice` 这类异常，
**先重试 + 等一会儿，别急着改代码或重装 torch**。

（`reports/rnafm_probe_gpu.json` 是 GPU 版结果，`reports/rnafm_probe.json` 是 CPU 版，内容一致。）

---

## PLAN §7.2 · RNA-QA 重建（采集已通，生成待 GPU）

### 管线逻辑离线自检（**已通过**）

真跑要等 15.3 GB 的权重，但**管线逻辑与模型无关**。
`scripts/99_selftest_pipeline.py` 用一个"假生成器"把 11/12/13 的代码路径整条跑一遍，
假生成器**故意做两件坏事**：按频率吐出序列子串、按频率输出非 JSON 废话。
全部 8 项检查通过：

```
[1] divide(): 20/20 条切块干净                      <- 块边界都落在句末
[2] 解析出 40 组 QA；非 JSON 输出 2 次；闸门拦下 4 条带序列的记录
  [ok  ] divide_keeps_sentence_boundaries
  [ok  ] json_parse_handles_fences                  <- ```json 围栏能剥
  [ok  ] guard_catches_leak                         <- 序列泄漏真的被拦
  [ok  ] guard_flags_bad_json
  [ok  ] guard_positive / guard_negative            <- 闸门不误杀
  [ok  ] guard_ngram_detects_bare_bases             <- 裸碱基串（非 reference 子串）也能抓
  [ok  ] wrote_jsonl
[verdict] PASS —— 管线逻辑没问题，等权重到位即可上量
```

**意义**：真跑之后如果数据是脏的，那一定是模型的问题，不是代码的问题 —— 归因干净。

### 已跑通：采集 50 条 (序列, 文献)

`scripts/10_collect.py`，最终链路 **nuccore → GenBank → pubmed**：

```
[batch 0] ids=60 gb=60 可用=60 pmids=18 已收 50/50
[stats] {"nt_scanned": 60, "bad_len": 0, "bad_alpha": 0, "no_pubmed": 0,
         "short_abstract": 0, "no_abstract": 0, "dup": 0, "kept": 50}
[lens] min=116 median=416 p90=825 max=966          <- 全部落在 RNA-FM 的 1024 内
```

产物：`data/raw/rna_literature_50.jsonl`

**两条被实测否掉的链路**（写下来省得重走）：

1. `pubmed → nuccore`（正向）：拿到的全是 2–4 kb 完整 mRNA（`NM_*`），
   越过 1024 nt 上限，**半小时一条都收不到**。
2. `elink` 反查：传**多个** UID 时会合并成一个 linkset，不告诉你哪条链接属于哪条序列
   （实测 5 个 id 的 pubmed 链接全挤在一起）。
   → 改用 GenBank 自带的 `PUBMED` 字段，一次 efetch 同时拿序列和 PMID，一一对应。

### Qwen3-8B 权重：**已修复并校验通过**

静默损坏的 1/2/3 号 shard 删掉**重新单写者下载**后：

```
model-00001-of-00005.safetensors   3996250744  3996250744  OK  31d6a825... == 31d6a825...
model-00002-of-00005.safetensors   3993160032  3993160032  OK  5991236c... == 5991236c...
model-00003-of-00005.safetensors   3959604768  3959604768  OK  c5185c47... == c5185c47...
model-00004-of-00005.safetensors   3187841392  3187841392  OK  b5ee7de7... == b5ee7de7...
model-00005-of-00005.safetensors   1244659840  1244659840  OK  20c2d636... == 20c2d636...
model.safetensors.index.json            32878       32878  OK  f9fdbcb9... == f9fdbcb9...
[OK] 全部文件与 ModelScope 声明一致。
```

**连贯性验证**（之前是乱码的那次）：

```
[gen] 载入 models/Qwen3-8B 用时 17.9s  device=cuda dtype=torch.bfloat16
prompt: "Name the four RNA nucleotides, separated by spaces. Answer in one line."
OUT: 'Adenine Cytosine Guanine Uracil'          <- 正常了
```

至此 PLAN §1 的两个模型（RNA-FM 640 维 + Qwen3-8B 4096 维）**都在本机跑通**。

### ⚠️ 采集质量的一个硬伤：文献复用 98%（**必须改，否则训不出东西**）

第一版 50 条记录里 **只有 7 篇不同文献**，其中 42 条共用同一篇
（C. elegans 基因组论文 PMID 9851916）——**84% 的"文献"跟这条 RNA 没关系**。

把 `--target` 提到 1000 后又复现了同一件事（子代理实测）：
1000 条里**只有 24 个不同 pmid**，其中一篇（1987 年 S. pombe 线粒体 LSU rRNA）
一条配了 **800 条**序列。后果：`literature` 字段作为监督信号基本无信息量。

根因：`esearch` 的 relevance 排序是确定性的，**第一条 query 一个批次就把 target 填满了**
（`by_query = {ncrna_refseq: 1000}`），后 4 条 query 一次都没跑到；
而每条 RefSeq 记录的 GenBank `PUBMED` 字段大量指向少数几篇"大论文"。

已加的闸门（`10_collect.py`）：
1. `--max-per-pmid 1`：同一篇文献最多给 1 条记录（默认开）；
2. `--require-mention`：DEFINITION 里的**特征标识**（如 `(Lncpnky)`、`(mcmdc2as)`）
   必须出现在摘要里 —— 这是弱相关性检查，但能把"挂名文献"挡掉。

加闸门后实测：单批 100 个 id 只留下 **5 ~ 20 条**（之前是 50）。
**这正是应该的** —— 宁可少，不能脏。

> 顺带一个标注噪声线索：那 800 条被引论文讲的是**线粒体 LSU rRNA 基因**，
> 这批 RefSeq 条目却注释成 "non-coding RNA … lncRNA"。值得单独看一眼。

---

## 下一步

1. Qwen3-8B 权重下完 → 跑 11/12/13，出 50 条 RNA-QA，人工抽查质量（PLAN §3 P0 判据）。
2. RNA-FM 权重到位 → 跑 §7.1 探针，把 output_shape / 上下文长度落成 `reports/rnafm_probe.json`。
3. bert / pubmedbert 到位 → 跑 §7.3 的 B/C 关（语义指标不变量 + 消融阶梯响应方向）。**已完成，见 §7.3 / §7.3b / §7.3c。**
4. 三者都绿之后，进 P1（Stage1 模态对齐：冻结 RNA-FM + 冻结 Qwen，只训 640→4096 投影）。

### 评测口径上必须落地的三条（来自 §7.3b，改 `04_eval.py` 时要照做）

- [ ] **报 semantic F1 时必须同时报基线。** 每次评测都跑一遍"跨域无关对 + 同域句级随机配对"，
      把地板跟着主数字一起报。否则读者无法判断读数落在窗口里的什么位置。
- [ ] **`S_BERT` / `S_Pub` / `S_GPT` 三列不可横向比较。** 每列只能和**它自己**的基线比。
      原版表把三个嵌入并排摆，容易被读成"哪个嵌入更好"，实际是三个不同的分母。
- [ ] **PubMedBERT 那列接近饱和，别当主指标。** 有效窗口只有 0.048 宽；
      报告里应以 ROUGE 为主、`S_BERT` 为辅、`S_Pub` 仅作参考。

---

## E 轨 Stage 1 进度 + 三个"读 checkpoint"的静默错误（已修）

### E 轨 Stage 1 跑到 38.7%

`checkpoints/stage1_run1/`，27 个 checkpoint，step 500 → 13500（`--compress-latent 64`）。
10 epoch × 3488 条 = 34880 步，**还差 21380 步（~5.9 h @ 1.0 s/step）**。

* `val_loss` 从 step 6200 起就在 1.365–1.393 横着，train 掉到 0.76–0.91 —— 典型过拟合。
* `proj_std` 0.32 → 0.39，**没有**退化成常向量。
* Δloss(置零) 峰值 **0.2732 @ step1600**，衰减到 **0.047 @ step13800**。

### ⚠️ 但"Δloss 衰减 = 投影塌缩"这个读法是**错的**

`scripts/31_stage1_rouge_curve.py` 的试点（这是论文的指标，才是判据）：

| step | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| 500 | 0.2084 | 0.0261 | 0.1337 |
| 8000 | **0.3030** | **0.1234** | **0.1914** |

原论文 Modality Fusion (D&C) = **0.0922** ⇒ 我们是它的 **~3.3 倍**。
step-8000 的生成结果里正确写出了 `LINC00115` —— **模型确实在读 RNA**。

**教训（值得单独记一条）**：Δloss 是**逐 token 平均**的。答案变长、变准、变多样，
分母变大，Δloss 天然被摊薄。**它不能替代 ROUGE 做收敛判据。**
`Stage1Config` 里 `collapse_rel_drop=0.75` 那套相对判据同样受这个问题影响，
所以 PLAN 里"置零编码器指标明显下降"这条**只能当辅助信号**，最终以 ROUGE 曲线为准。

### 修掉的三个静默错误（都是"读 checkpoint 时少装东西"，不报错但结果全错）

1. **`scripts/22_zero_encoder_probe.py` 直接崩。**
   `Stage1Config(llm_path=args.llm, **(blob["config"]))` ——
   checkpoint 里的 config 含 `llm_path`（和 `args.llm` 撞名 → `TypeError: got multiple values`），
   且含 `collapse_delta` 这种 dataclass 里根本没有的旧字段。
   → 改成只挑 `dataclasses.fields(Stage1Config)` 认识的键，命令行优先；
   并**装回压缩器**（以前只 load `projection`，E 轨 checkpoint 的压缩器是随机初始化的）。
   同时加了 `--pass-delta/--weak-delta`，并在输出里写明 Δloss 只是参考。

2. **`src/rnaqwen/models/finetune.py` 不支持压缩，E 轨没法进 Stage 2。**
   `build_stage2_model` 硬编码 `Stage1Config(...)`，`encode_samples` 直接
   `model.projection(rna)` —— 压缩器根本没有插入点。
   → `Stage2Config` 加 `compress_latent/compress_method`；
   `build_stage2_model` 传下去、从 blob 装 `compression`、
   **R/E 配错直接 `SystemExit` 报错**（不让它静默跑歪）；
   `encode_samples` 在投影前插 `if model.compression is not None` 分支（None 时逐位不变）；
   `save_stage2` 一并存 `compression` + `compression_spec`。

3. **`scripts/30_infer.py` 的压缩器权重从来没载过。**
   原来只有 `compressor = RnaTokenCompressor(encoder.hidden_size, args.compress_latent)`
   —— **随机初始化**，E 轨推理输出看着像样但全是错的。这是最危险的一个。
   → 改成从 `blob["compression"]` load，配置自动从 `compression_spec` 读，
   配错直接报错退出。

### 冒烟验证（E 轨 Stage 2 全链路，已通过）

```
[stage2] 载入 Stage1 投影 <- checkpoints\stage1_run1\projection_step13500.pt (step 13500)
[stage2] 载入压缩器 <- checkpoints\stage1_run1\projection_step13500.pt (resampler n_latent=64)
trainable params: 15,335,424 || all params: 8,206,070,784 || trainable%: 0.1869
[stage2] {"step": 1, ..., "loss": 1.3883, "seq_len": 135, "elapsed_s": 3.4}
...
[stage2] 完成 6 步，用时 0.1 min
```

* `attn` 预设的 15,335,424 与静态估算**逐位对上**（见 `docs/METRICS.md`）。
* 压缩后 `seq_len` 135–169（不压缩会是 600–1100）。
* 存回的 `projection.pt` 里 `compression` + `compression_spec` 都在，
  `config.compress_latent = 64`，往返一致。
* 实测 Stage 2 吞吐 **0.9 s/样本** ⇒ 80,907 条 × 10 epoch ≈ **202 小时**。
  **所以 Stage 2 默认从 3 epoch 起跑（≈61 h），不是 10。**

> 顺手修的第四个：`--kind` 的 help 写的是 `open | closed`，但数据里的值是
> `open_ended | closed`。按 help 写 `--kind open` 会筛出 0 条样本。
> **文档写错选项名 = 静默的空数据集。** 已改。

### 手动运行

`RUN.md` 是给用户自己跑的命令单（工具 shell 拉起来的长任务已静默死过 5 次以上，
无 traceback、无 abort、事件日志里也没有）。

---

## 🔴 重大事故：RNA-FM 权重一直就是坏的，13500 步 Stage 1 全部作废

**发现方式**：用户贴了一段 `RnaFmModel LOAD REPORT` 问正不正常。报告本身确实正常
（UNEXPECTED 的是 `lm_head.*`/`ss_head.*` 预训练头，MISSING 的是从没用过的 `pooler.*`）。
但顺手验一下「权重真的装进去了吗」，验出问题。

### 症状

```
模型 state_dict 张量数 : 200
checkpoint 原始键数     : 205
剥掉 `model.` 前缀后共有键 : 198   （前缀差异是 from_pretrained 的正常行为，不是问题）
逐位相同                 : YES
未覆盖的模型键           : ['pooler.dense.bias', 'pooler.dense.weight']   <- 只有这个
```

到这里一切正常。继续查：

```
word_embeddings.weight (28, 640)  absmax = 0.000000e+00   nnz = 0 / 17920
序列敏感性：ACGUACGU… vs GGGGGG…   max|Δ| = 0.000000e+00
```

**`word_embeddings` 全零，且不同序列的输出逐位相同。**

前向路径是 `inputs_embeds = self.word_embeddings(input_ids)`（从 `RnaFmEmbeddings.forward`
的源码确认），全零 ⇒ 输入 = 只有位置编码 ⇒ **encoder 完全看不见序列**。

### 量化

```
总张量 205 / 正常 50 / 全零 155
12 层 encoder 的 attention / intermediate / output 几乎全在零里
```

### 根因：文件是对的**大小**，错的是**内容**

```
本地 sha256  38f3ae9aadbf4d3a5ff6873476fe9d16600b83a54711637ec3760e8e83a2408d
上游声明     ec69333eb894dbefb580e690b0cbe1cbae9845fefd2bc78da914cd14226a2828
大小        398,117,588  ==  398,117,588   ← 逐字节相同
```

**和 Qwen3-8B 那 4 个 shard 一模一样的故障模式。**

### 现场复现（这次抓到了机制）

重下时实测 `.part`：

```
逻辑大小 377,701,268 B      ← 只下了 57 MiB，文件却已经接近满
@  1 MB : 有数据          @ 10 MB : 零填充
@ 30 MB : 有数据          @ 50/100/200/300/350 MB : 零填充
Valid Data Length = 377701268   ← NTFS 认为整段都是「有效数据」，零就是真零
```

aria2 **预分配**了整份文件，只有真实分片有数据，其余是零。
（命令行确实带了 `--file-allocation=none`，aria2 1.37.0；大概率是上一轮中断留下的
预分配文件没被截断，新一轮 `--continue=false` 从 0 覆写，尾部那些零是上回的。）

**中断 + 预分配 = 大小正确、内容一半是零。** 这就是全部。

### 为什么没被发现（三个自己给自己发的假通行证）

1. **`from_pretrained` 不报错。** LOAD REPORT 只提 `lm_head`/`ss_head`/`pooler` 这三个
   无关紧要的键，看起来像"只是架构差异，可以忽略"。
2. **探针只查形状。** `01_probe_rnafm.py` / `01b_probe_rnafm_limit.py` 验的是
   `out=(1,83,640)` —— **一个全零的模型照样过**。我据此写过 `[OK]`。
3. **没有 sha256。** Qwen3-8B 坏过一次之后明明立了「单写者 .part + 强制 sha256」的规矩，
   但**只用在了 Qwen3 上，RNA-FM 漏了**。规矩没有强制力，就等于没有。

### 作废清单

| 东西 | 状态 |
|---|---|
| `checkpoints/stage1_run1/` 27 个 checkpoint（13500 步 ≈ 3.75 h） | ❌ 废 |
| `checkpoints/stage1` 500/1000/1500、`_losscheck`、`_timing_gc` | ❌ 废 |
| ROUGE 曲线试点 0.2084 → 0.3030 | ❌ 废 |
| 「step-8000 正确写出 LINC00115 ⇒ 模型在读 RNA」 | ❌ **收回**。encoder 对序列零敏感，这个结论不成立 |
| Δloss 置零探针 0.273 → 0.047 的"塌缩"判读 | ❌ 无意义 |
| 数据 `rnaqa_n6.jsonl` / `evalset` | ✅ 与 encoder 无关，有效 |
| 代码（含本轮修的 4 个 bug） | ✅ 有效 |
| 计时测量 | ✅ 大致仍然成立（encoder 本来就便宜） |

### 修复

`scripts/00e_refetch_rnafm.py`（新）——所有通路都走同一个闸门：

```
下载 -> .part -> sha256 必须 == ec69333e…a2828 -> 才允许原子改名
```

配套规则（每条都对应一个踩过的坑）：

* `archive_corrupt()`：坏文件改名留档，不删；
* **哈希不符时绝不改名**；
* 哈希不符 + 大小已满 ⇒ 判定为「预分配空洞」。这种情况**续传救不了**
  —— 控制文件会说"分片都完成了"，每次续传立刻返回同一个坏文件，卡死循环。
  必须连 `.part.aria2` 一起清掉；
* 哈希不符 + 大小未满 ⇒ 真没下完，保留 `.part` 续传；
* **下载完还要独立验一次「换序列输出会变」** —— 哈希对了也可能装错文件。

### 立规矩

**禁止一直换源。** 选定一条通路就走到底，慢就慢。
把关的不是速度，是校验。遇到"慢"，不要去找更快的源，先把校验补上，然后等。
（已写进 memory.db #1011。）

---

## ✅ 换到 ERNIE-RNA（E 轨）+ 下载与健全性检查全过

### 决定

用户拍板：**改用 ERNIE-RNA 作 backbone**，理由是结构增强（attention bias / pairwise 表示），
性能强于 BERT 基座的 RNA-FM。
**这是 E 轨**（PLAN §7）。论文用的是 RNA-FM，所以 ERNIE-RNA 的任何数字都要单独标 E 轨，
不能和论文 / R 轨并列。

### 取数（三条路都试了）

| 源 | 结果 |
|---|---|
| HF `multimolecule/ernierna` | ✅ **最终成功**。345,071,800 B，sha256 `02274cb7…08eb`，与上游声明逐位一致 |
| hf-mirror.com（并行） | 下到 210/329 MB 后放弃（同一份 artifact，谁先完成算谁的） |
| GitHub `Bruce-ywj/ERNIE-RNA` | 不放权重，只有代码 + 一个 Google Drive 文件 ID |
| Google Drive `ERNIE-RNA_pretrain.pt` | 作者**原始 fp32 权重**，1,035,221,383 B（1.04 GB），实测传不动。**留作兜底** |
| ModelScope | 没有 ernierna。有 `ZhejiangLab-LifeScience/{rnafm,rnaernie,rnamsm,rnabert,dnabert2}`，但都是**浙江lab 自己的转换版**（vocab 26，键名不同），不是 multimolecule 的 |

### 健全性检查：**PASS**

```
sha256            02274cb7…08eb   ✅ 与上游声明一致
张量总数 207，全零 1                ✅ 只有 pooler.dense.bias（非主干，从不使用）
键覆盖 202/202，未覆盖 0            ✅ 权重真的装进模型了
输出形状 (4, 202, 768)             ✅ hidden 768
base vs polyG      max|Δ| = 3.1138e+00   ✅ 序列敏感
base vs shuffled   max|Δ| = 2.3979e+00   ✅ 次序敏感
base vs base       max|Δ| = 0.0000e+00   ✅ 确定性
VERDICT: PASS
```

### 关键实测差异：ERNIE-RNA **没有** RNA-FM 的 1022 nt 硬上限

```
RNA-FM     position_embedding_type = 'absolute'    -> 查表，表只有 1026 行，1023 nt 直接 IndexError
ERNIE-RNA  position_embedding_type = 'sinusoidal'  -> 算出来的，无硬上限；实测 1030 nt（1032 token）也能过
```

但 config 的 `max_position_embeddings = 1026` 是**预训练窗口**，超出去是外推。
所以 `ARCHS["ernierna"].max_nt = 1024`（= 1026 token − BOS/EOS），取预训练范围而不是拍一个数。
（现有数据按 1022 过滤过，所以这个数目前不会真的被用到。）

### 新增/改动

| 文件 | 作用 |
|---|---|
| `src/rnaqwen/paths.py` | **新**。集中默认路径 + 轨道归属。R 轨 `models/rnafm`，E 轨 `models/ernierna`。覆盖优先级：`--encoder` > `RNAQWEN_ENCODER` > 默认 |
| `src/rnaqwen/models/encoder.py` | 重写为多 backbone `RnaEncoder`，按 `config.json` 的 `model_type` **自动识别**；arch 与 config 不符、hidden 不符都直接报错 |
| `scripts/01c_check_encoder_sanity.py` | **新**。四条闸门（哈希 / 结构 / 键覆盖 / 序列+次序敏感性+确定性）。已用已知损坏文件验证过会 FAIL |
| `scripts/00d_fetch_verified.py` | **新**。通用哈希闸门取数；拿不到上游 sha256 就拒绝下载大文件；单写者锁；真实进度读数 |
| `scripts/01b_probe_rnafm_limit.py` | 改为按 arch 解析，不再硬编码 `RnaFmModel` |

### 下一步

1. Stage 1 冒烟（小样本几步），确认 `768 -> 4096` 的链路通
2. Stage 1 全量（E 轨，10 epoch）
3. ROUGE 曲线






---

## 🔄 转向：彻底放弃 RNA-GPT 软提示路线（2026-10-10）

**性质变更：本项目不再是 RNA-GPT 的复现。** 变成「RNA-QA 上软提示路线 vs 原生文本路线的头对头对比」。
新计划见 `PLAN_v2.md`；旧计划留档为 `PLAN_v1_softprompt.md`。

### 触发点不是"做不下去"，是逐条测出了天花板

用户最初提出「Stage 1 对齐目标与 Stage 2 指令目标脱节」的假设，我核实后发现它是**症状不是病根**。
真正的病根在核实过程中浮出来：**我们冻结了 LLM。**

| 检验 | 结果 |
|---|---|
| 投影在丢信息？ | ❌ 减掉公共分量后 inter = **-0.05**，信息完整 |
| 学习式压缩器造成偏执依赖？ | ❌ meanpool 与 resampler 逐 token 剖面几乎一致（负值 58.4% / 63.3%） |
| val_loss 能选投影？ | ❌ 两次独立配对都是 val 低 ↔ grounding 低 |
| 去各向异性是解？ | ❌ val 打平（1.4892 / 1.4926），Δzeroed 差 8× 但消融口径不同、不可比 |
| 原生文本要改 tokenizer？ | ❌ 0.526 tokens/nt，扰动中位影响跨度 **0 nt**，局部稳定 |

### 四条外部证据（都核实过原文，不是转述）

- **ChatMultiOmics** arXiv 2412.19191：`directly applying LoRA ... near-random performance in
  classification and regression tasks ... LoRA primarily leverages pre-trained knowledge to generate
  well-formed answers based on the output format learned from SFT data`
  → 精确命中我们的观测：val_loss 降到 1.384（格式学会了），Δzeroed 衰减到 25%（内容没进去）
- **MAPO** arXiv 2605.27741（JHU + 腾讯混元，作者 Sanjeev Khudanpur / Liefeng Bo）：
  术语 **late-stage modality collapse**；`abandon the primary source signal in favor of
  compressed textual priors`。其仪器（消融参照的跨模态差分熵）与我们的 Δzeroed 是同一个东西，
  但**逐 token、更强**。
- **Moirain** arXiv 2605.23961：RNA + 多模态 SFT + DPO；DPO 动机是
  `without collapsing the learned natural distribution`。
- **RNA-GPT** arXiv 2411.08900：Stage 1 的 Q 留空是**故意**的，为堵死"抄 prompt 文本"——
  这是它的解，也是它的病。

### 今天用血换来的六条方法论

1. `mem_get_info()` 在 gfx1151(APU) 上不可信（报 107 GB，物理内存只 63.6 GB）→ **只看物理内存**
2. 哈希闸门是唯一能挡「大小对、内容空」的手段（已两次）
3. 单测过了还要跑真模型冒烟（SOS-LoRA 单测 8/8，冒烟仍抓出 dtype 与作用域两个 bug）
4. **「测出来的东西和以为测的不是一回事」今天犯了四次** —— 每次都在
   「全零 / max 也零 / 比值太整齐」这类信号上露馅。**看到这种信号先怀疑自己。**
5. 优先用**直接量**（可分辨性、逐 token），不用聚合标量（val_loss 已被证伪两次）
6. 换源不算办法 —— 先修校验，再等

### 新路线

```
RNA 序列 → Qwen3 tokenizer 原生 token（0.526 tokens/nt）
         → ① 序列继续预训练（可选，第一版先跳过）
         → ② 指令微调（SFT / LoRA / SOS-LoRA）
         → ③ 推理微调（可选）
```

**保留**：4,481 RNA / 92,680 QA 数据集、evalset_merged、评测协议、Qwen3-8B、SOS-LoRA 实现。
**零新架构、零 tokenizer 改动**，只需要一段几十行的语料预处理。


---

## 2026-10-10 小实验：FSQ 的码到底是词表还是哈希（44/45/46）

用户问「能做个小实验吗」。这条路线能不能走，取决于一个**不用训 LLM 就能回答**的问题：
离散码是不是可复用的词表。跑了三个递进的对照，**其中两个指标被我自己的设计否掉了**。

### 44_ 覆盖率 —— 第一次自我否证
v1 用 64 条序列 (29k 位置) + 词表 65536，得到「52% 的码只出现一次」。
**这是算术必然，不是发现** —— 平均每码 2.6 个位置，中位重用必然是 1。
v2 改成 256 条序列 (116k 位置)、扫词表 {256,1024,4096,6561,65536} + iid 自助基线。
结果：所有配置的留出覆盖率**都略低于**基线（4096: 95.8% vs 97.5%）。

### 45_ 序列指纹对照 —— 第二次自我否证
「低于基线」的唯一解释应是码在序列内成簇。于是加对照：
(a) 同序列内位置留出 vs (b) 跨序列留出。
- burstiness：平均游程 **1.02**（iid 应为 1.0）—— **码在序列内不成簇**
- (a) -2.0pp，(b) -2.7pp —— **两者一样**

「序列指纹」的解释被推翻。但更重要的：**覆盖率这个指标本身没有判别力。**
词表 4096、训练位 9.3 万时，任何映射都会让每个码被见到几十次 —— 哈希也会给出
90% 利用率和 95% 覆盖。词表远大于样本时又样本不够。**两头都不成立。**

### 46_ 含义迁移 —— 唯一非循环的测试
靶子换成**局部 6-mer**（来自输入序列，不是 encoder 学的，所以不循环）：
在训练序列上学 code -> 最常见 6-mer 的表，在**没见过的序列**上考。

| 量 | 值 |
|---|---|
| 测试位置中码在训练集见过 | 99.1% |
| 按码预测 6-mer 准确率 | **8.48%** |
| 打乱 code<->位置 对照 | 2.95% |
| 常数预测（最常见 6-mer） | 3.02% |
| **增益 / 倍数** | **+5.53pp / 2.88x** |

打乱对照回落到常数水平（2.95% ≈ 3.02%），说明对照有效。

### 结论
**是词表，但是一个不精确的词表。** 码确实携带能跨序列迁移的局部结构信息（2.9x），
不是给每条序列发指纹。但要老实说清两个限制：
1. 局部 6-mer 是**很弱**的靶子 —— ERNIE-RNA 在一位点上编码的远不止 6-mer。
   所以 2.9x 是**下界**，不是上限；但它也说明码与「局部序列组成」的相关性并不强。
2. within/重建 err 这两个指标是**循环的**：FSQ 是确定性映射、AE 又照重构误差训，
   within<1 是构造保证的。**不能拿它们当证据。**

### 对齐踩的坑（记下来）
ERNNIE-RNA 的 tokenizer 每条序列加了 [CLS] 和 [SEP]：
encoder 位置数 = 序列长度 + 2。115886 - 115374 = 512 = 2 x 256，一眼可验。
不做这个偏移，k-mer 靶子整体错位一格，结果全是废的。

### 方法论：一天内第 5、6 次「测的不是我以为的东西」
- 「52% 的码只出现一次」= 词表比样本大（算术必然）
- 「覆盖率低于基线」= 指标无判别力（两头都不成立）
两次都是**先看到太干净的数字，然后才去怀疑自己**。规则 4 又生效了。


---

## 2026-10-10 路线 C 端到端短实验启动（47_）

PLAN_v2.md 已更新：§6.4 第 2 条（去各向异性「前置必需」）**被自己的测量证伪**，降级为消融项；
§6.5 记录 44/45/46 与那两次自我否证；§6.6 记 [CLS]/[SEP] 偏移的坑；§7 换成短实验清单。

### 脚本 scripts/47_routeC_short_sft.py
单变量：底座冻结 + LoRA(attn) + **一张 4096x4096 可训练码表**（16.8M 参数，与臂 A 的 3M 投影同量级）。
RNA 位置不经过 tokenizer，直接用 inputs_embeds 把码 id 映射到码表（不用 resize_token_embeddings，
不污染 Qwen 原词表）。判据是 MAPO 式**逐 token 消融**，不是 loss —— loss 降下来也可能只是模型学会忽略 RNA。

四路消融：真表 / 零表 / 随机表 / 常数码。**关键在对照，不在第一个数：**
- vs 零表 低 => RNA 位置上「有东西」在起作用
- vs 随机表 低 => 起作用的是**码的内容**
- vs 常数码 低 => 起作用的是**码的区分度**
后两个 ≈50% 就是「Qwen 眼里的 RNA 是乱码」，只是换了形式。

### 冒烟（60 条 RNA / 30 步）发现的两个问题 —— 都是我的错
1. |grad_code| 0.0000 每步都是 0。**因为我在 opt.zero_grad() 之后才读 .grad。** 打印全废。
   修到 step 之前读，实测梯度和 19.65 -> 0.79，码表确实在学。
2. **判定逻辑漏了对照。** 第一版只看「vs 零表 < 58.4%」就判 ✅。
   实际冒烟结果是 vs 零表 32.9%、**vs 随机表 48.1%、vs 常数码 47.6%** ——
   后两个是掷硬币，说明模型用的是「RNA 位置有个可训练向量」而非「这个码是什么」。
   **差点报出一个假的成功。** 判定已改成必须三个对照同时 < 45%。

另一个接线 bug：
na_emb 建在 CPU 上（只有 peft 模型走 device_map 上了卡）-> 必须在
ase_emb.weight.device 上建。**方法论规则 3：单元测试过 != 接线对。**

### 冒烟的定位
30 步、码表漂移仅 **3.64%**，对照 ≈50% 是必然的，**没有判别力**。
冒烟只证明端到端能跑通（梯度非零、初始行范数 1.28 与 Qwen 词表 1.38 同量级、四路前向都通）。

### 正式短实验（已在后台启动）
1500 条 RNA / 31190 QA，600 步，每 150 步跑一次三路消融。
**要看的是对照差距随码表漂移「拉开还是一动不动」** —— 那才是判据。
FSQ 已在 668273 个位置上训完，mse 0.045。



---

## 2026-10-10 路线 C 第一阶段（全量）完成 —— 判据 ② 通过

scripts/48_stage1_full_tokenizer.py，4481 条 RNA 全量，4096 码。

`
位置          2018834  (中位序列长 335)
重建 mse      0.0444   归一化 err 0.4934
用码          4092/4096 (99.9%)   位置/用码 493.4   中位重用 264   仅1次 0.3%
迁移          8.62%  vs 打乱 3.27%  vs 常数 3.40%   =>  **2.63x**
用时          17.5 min
`

判据 ② 阈值 2.0x，**通过**。产物 checkpoints/fsq_full/{tokenizer.pt 1.1MB, codes.pt 8.8MB}。

### 一个反常，必须记下来

数据从 1500 条 -> 4481 条（3 倍），**利用率从 90.2% 涨到 99.9%、中位重用从 12 涨到 264**，
但**迁移倍数反而从 2.88x 掉到 2.63x**。

这再次印证 44_/45_ 的教训：**利用率不是判据**。利用率涨了 10 个点、重用涨了 20 倍，
真正有判别力的那个数没涨。

但我不据此下结论：0.25 的差来自**两组不同的划分**（1500 条 vs 4481 条，
测试序列也不同），可能只是划分噪声。要判数据量对迁移有没有用，
得在同一划分上比不同数据量 —— 现在没这个实验，不做判断。

### 内存事故（已处理）

两个任务并行时 RAM 冲到 **63.3 / 63.6 GB**（用户的 llama-server + Qwen3-8B 16GB +
两个训练进程）。短实验（pwsh-79）自 step 1 后无任何进展，判定为卡死。

**杀掉 79，保 80。** RAM 回落到 20.3 GB，80 正常跑完。
理由：79 用的是 1500 条码表，本就会被全量码表取代；而且用户明确要求优先全量第一阶段。

**教训补充到规则 1**：APU 上 mem_get_info() 不可信，Get-CimInstance Win32_OperatingSystem
的物理内存才是真的；**但看内存要一直看，不能只在启动时看一次** —— 这次是并行两个任务时
才撞上的。

---

## 2026-10-11 · 路线 C 码信道审计（`66_code_channel_audit.py`）：三个假设被自己的测量否掉

PLAN_v2 §7.2.e 的分支是「判据④若不动 -> 加复用压力再试一次」。要定这个分支，得先回答一个
**stage-0 之后就没再问过**的问题：e2e 训练之后，码表还是不是那个可复用的词表？

判据②只在**冻结 tokenizer** 上量过（2.63x，逐核苷酸一个码）。而实际接进 LLM 的是**另一版**：
前面多了 pool=4 的学习式压缩，`enc` 从 stage-0 载入后又被 CE 梯度训了 16000 步
（日志 `|grad_fsq|` 0.03~0.7）。**这一版从来没测过判据②。**

三路对照，同一批隐藏态 / 同一套 6-mer 靶子 / 同一个跨序列划分（hold=0.2, seed=7）：

| 版本 | 按码预测 6-mer | 打乱 | 常数 | 倍数 | 全量用码 | 硬熵 |
|---|---|---|---|---|---|---|
| A 冻结 · 逐核苷酸 | 8.62% | 3.27% | 3.40% | **2.63x** | 4092/4096 | 7.72/8.32 |
| B 冻结 enc + 平均池化 4x | 5.83% | 3.17% | 3.40% | **1.84x** | 3782/4096 | 6.78/8.32 |
| C **e2e 训练后** · 池化 4x | 23.57% | 3.03% | 3.40% | **7.79x** | 3885/4096 | 7.44/8.32 |

A 行精确复现 48_ 的 2.63x —— **尺子先自检，C 行的读数才作数**。

### ① 「码本塌缩到 350/4096」不成立 —— 又一次采样量幻觉

训练日志里的 `[diag] 码本: 用到 350/4096 熵 5.86/8.32` 只覆盖 **40 条序列 / 约 4.5k 窗口**。
全量 4481 条 / **50.3 万窗口**上是 **3885/4096 在用、硬熵 7.44/8.32、最大码占比 0.43%**。
**没有塌缩。**

44_ v1 的教训（「52% 的码只出现一次」是算术必然、不是发现）**原样又犯了一次**，
而且这次是犯在**已经写进训练日志结论**的那一行上。

### ② e2e 训练没有破坏 tokenizer，反而把它变强了（2.63x -> 7.79x）

原假设「CE 梯度把码改成了每条序列各自不同的片段」被否掉。附带读数：

```
码 ID 与冻结版一致率（窗口首位）  0.1%     <- 键全换了，内容映射还在
enc 相对漂移                      31.6%
compress 距平均池化               36.5   （平均池化自身范数 13.9，即已远离 3 倍）
```

**必须写清楚的口子**：7.79x 只在**窗口首位**那个 6-mer 上量过。若 `compress` 学到的是
「只编码窗口首位」，这个数会虚高，而窗口内另外三位可能已经丢了。要坐实得补
「窗口内四位各自的可预测性」对照。**未做。**

### ③ 熵正则是惰性的 —— `--ent-coef 1.0` 这 16000 步几乎什么都没干

软使用率饱和在 **8.31/8.32**（上限 8.32），梯度 ≈ 0。原因是构造性的：真码在
`tau=6`、`d2∈[0,48]` 的 softmax 里只分到约 6% 权重，其余摊给 12 个邻居，平均下来必然接近均匀。
所以**「复用压力」从来没真正施加过** —— 这既解释了 §6.5 那句「2.88x 是在完全没有复用压力下
拿到的」，也说明不能拿 `--ent-coef 1.0` 当「加过压力」的证据。

### 仪器问题：训练日志的 `ce` 被样板文本稀释

A_v3 报 `ce 0.2~0.9`，但 labels **从 assistant 头开始**（有意如此，否则生成学不会停）：
`<|im_end|><|im_start|>assistant\n` + 空 think 块(13 token) + ` A:` + eos ≈ **18~20 个几乎确定的
token**，而答案本体常常只有 **1~3 个 token**（「GC 二核苷酸出现几次」-> 「 0」）。
**ce 0.4 完全可能意味着「样板学会了、答案一个字没学会」。** 已加 `68_answer_only_ce.py` 拆三段量。

### 数字上的落地：mode 地板（`_mode_baseline.py`）

「不看 RNA、只按问题答训练集众数」在这批 90 条上只有 **3.3%**（全测试集 5.07%）。
所以 acc(A) 16.7% / acc(C) 25.6% **不是**先验预测器能解释的水平 —— 但也说明
「不给 RNA」那一臂另有东西在起作用，必须查（见下一条）。

### 结论

```
判据④ 在 12000/15000 步就全过（33.0/36.2/44.0/28.9 与 37.4/41.6/24.2/39.5，门槛 45%）
但 acc(A 真 RNA) 16.7% < acc(C 不给 RNA) 25.6%
=> 消融过线 ≠ 模态对齐。瓶颈在 LLM 侧：码里有东西，模型不读。
```

---

## 2026-10-11 · 码信道的天花板（`69_code_answer_probe.py`）：局部强、全局弱

§9 只证明了「码是可复用词表」。这只说明码携带**局部结构**，不说明**码里装着这批问答的答案**。
两个岔路（动 LLM 侧 vs 动 tokenizer）靠训更多步分不开，得用**不需要 LLM** 的探针量天花板。
设计：在训练序列上拟合「码直方图 → 答案」，在没见过的序列上考（划分与 LLM 评测同一份）。

### 先补掉 §9 留的口子：窗口内四位各自的可预测性

```
窗口内第 0 位 92.1%   第 1 位 92.9%   第 2 位 92.4%   第 3 位 89.8%   （随机 25.0%）
```

**四位都还在。** 也就是说 7.79x 不是「compress 只编码了窗口首位」造成的虚高 ——
池化没有把窗口内容压掉。这个口子关上了。

### 探针（只看码）逐模板

| 模板类 | 探针准确率 |
|---|---|
| dinucleotide at positions 1 and 2 | **81.0%** |
| first 4 nt / 它的反向互补 | **67.4%** |
| first 6 nt / positions 1-6 / 反向互补 first 6 | **55.3%** |
| first 8 nt | 44.6% |
| What type of RNA is this? | **62.8%**（众数地板 34.7%） |
| last 4 / last 6 / last 8 nt | 14.9% / 11.3% / 8.9% |
| **全部计数类**（4 种碱基计数、6 种二核苷酸计数） | **0.1% ~ 3.7%** |
| **全部百分比类**（GC、各碱基占比、嘌呤/嘧啶） | **0.0%** |
| G in first 50 | 3.2% |

总体 18.3%（4807/26312）。对照：LLM 旧口径 acc(A) 16.7%、众数地板 3.3%。

### 计数天花板：不训练任何东西，看码里到底有多少计数信息

精确匹配被 0.1% 的精度要求卡住，不能区分「码里没有」和「线性探针做不到」。
所以另做**无状态最优估计**：每个码在训练窗口里的平均碱基个数，逐窗口求和。

```
A 计数：MAE 8.41 个   中位 4.79   |误差|<0.5 的比例 10.5%
C 计数：MAE 8.09 个   中位 4.50                     11.4%
G 计数：MAE 7.90 个   中位 4.06                     11.2%
U 计数：MAE 8.22 个   中位 4.80                      8.7%
GC 含量：MAE 4.21 个百分点                           5.8%
（对照：众数地板在这些模板上是 1.6%~2.6%）
```

### 结论：码信道是**局部强、全局弱**

- **局部序列**（前几位、位置 1-2 的二核苷酸）从码里读得出来：55%~81%，远离概率水平。
- **精确的全局计数**读不出来：最好的无状态估计也只有 ~11% 精确，MAE 约 8 个碱基。
- 而 30 个模板里 **16 个是计数/百分比类**。

这条结论有两个必须写清的限制：
1. 「无状态」是**下界**。码是上下文化的，序列感知的估计器（含 LLM）可以更好；
   所以 MAE 8 个碱基不等于「不可能」。
2. 反过来，局部序列那几类模板**探针能做到 55%~81%**，这是 LLM 必须至少达到的水平。

---

## 2026-10-11 · 仪器翻车：评测 prompt 里带着金标准答案（`_probe_eval_prompt.py`）

**`64_acc_nothink.py` / `67_acc_dump.py` 的准确率全部作废，PLAN §8.5 那三个数（16.7/8.9/25.6）也是。**

`gen()` 把 `build()` 返回的 ids 整个喂给了 `generate(inputs_embeds=...)`，而那个 ids 是

```
head + tk.enc(f" A: {r['answer']}{tk.eos}")
                 ^^^^^^^^^^ 金标准答案
```

`lab` 才是拿来算 loss 的。而 transformers 在**只给 `inputs_embeds`** 时会把 `input_ids`
初始化成 `(B, 0)`（`transformers/generation/utils.py:694`，已读源码核实），
**返回的序列只含新生成的 token**。于是这个评测问的其实是：

> 「金标准答案和 `<|im_end|>` 都已经写在上下文里了，请接着写。」

实测（`reports/probe_eval_prompt.log`，同一个样本，逐字）：

```
prompt 结尾 : '... A: CUGGCC'                       <- 金答案在里面
续写(原样)  : '<|im_start|>assistant\n A: GGGGAG<|im_end|>'   <- 模型另起一轮，再答一遍
评测口径的预测: 'GGGGAG'   金标准 'CUGGCC'   ✗
clean（只到 assistant 头 + 空 think 块）:
续写(原样)  : ' A: GGGGAG<|im_end|>'                 <- 同一个答案
```

三件事同时说清楚：
1. 旧口径**不是**「第一次作答的准确率」——它是「第二轮作答」；
2. 金答案摆在眼前，模型**也没有去抄**（leak 与 clean 给的是同一个 'GGGGAG'）——
   这本身说明它连"复制上下文里的数字"都没做；
3. 但只要有一半样本会去抄，旧口径就会被**系统性抬高**，方向不明。所以口径必须废掉重测，
   不能拿旧数字下任何结论。

**修正后的 `67_acc_dump.py`**：ids 只到 `assistant 头 + 空 think 块`，答案段不再进 prompt；
同时修掉另一个口径错误：split 有三个值（train / val / test），
旧写法「非 train 即 test」把 **val 的 13,167 条混进了测试池**（64_ 只用 test，两边样本本就不同）。

**教训（补进 §5 方法论）**：`generate(inputs_embeds=...)` 的返回值**只含新 token**，
这个语义与「把 ids 一起传进去」放在同一行代码里，是极容易读错的组合。
凡评测口径，都要像 §9 那样**先用一个样本逐字打印 prompt 与输出**再去跑全量。



