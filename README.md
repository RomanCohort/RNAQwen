# RNA-GPT 复现（RNAQwen）

复现 [RNA-GPT](https://arxiv.org/abs/2411.08900)（MLSB @ NeurIPS 2024），
在**它自己的评测口径**上达到同量级。计划与验收标准见 [`PLAN.md`](PLAN.md)。

> **一句话**：原版代码/权重/数据/评测集**全都没开源**。本仓库的目标是把它们做出来 ——
> RNA-QA 数据集、评测集、训练代码、LoRA 权重，全部公开。

---

## 0. 当前进展

> 与站点「我们在做什么」同源；只写有报告文件可对证的部分。

| 项 | 状态 |
|---|---|
| 管线 / 数据 / 训练产物 | **已入库并公开**：重建的 RNA-QA 数据集、LoRA 适配器、码本检查点 |
| 端到端训练 | 已推进到 **step 15,000**（`reports/acc_by_source_15000.json`） |
| 码信道（从序列码读回短序列） | **成立**：三档 45% / 7.5% / 17.5%，概率地板 4.5% |
| 自由文本 / Rfam 家族 | **尚未学起来**（贴近地板），如实记录，不写成"完成" |
| 满级评测（ROUGE-1/2/L + 三种嵌入的语义 P/R/F1） | 进行中 |
| 评测口径 | 2026-10-11 发现旧口径把**金标准答案放进了 prompt**，那批准确率全部作废并已重测（`PROGRESS.md`） |

复现配方见 §1；**任何数字都必须连同 §6 的口径声明一起引用**。

---

## 1. 复现配方

| 项 | 原版 | 本复现 |
|---|---|---|
| 底座 LLM | Meta Llama-3 8B Instruct | **Qwen3-8B**（`Qwen/Qwen3-8B`，hidden=**4096**，与 Llama-3-8B 同维 → **投影层零改动**） |
| 序列编码器 | RNA-FM（冻结） | 同（`multimolecule/rnafm`，hidden=640，max_position_embeddings=1026） |
| 投影层 | 线性 640→4096 | 同（**不改维度**） |
| 阶段 | Stage1 模态对齐 → Stage2 指令微调 | 同 |
| Stage1 超参 | 10 ep / bs=1 / wd=0.05 / warmup 2048 / AdamW β=(0.9,0.98) / LR 1e-4→8e-5 / cosine+AMP | 同 |
| Stage2 | 10 epochs, warmup 200 | 同（单卡退化为 LoRA） |
| 数据 | RNA-QA 407,616 条（5K RNA / 121K QA 子集训练） | 重建，生成器由 GPT-4o/4o-mini 换成**本地 Qwen** |

**验收指标**：ROUGE-1/2/L + 三种嵌入的 Semantic P/R/F1（`S_BERT` / `S_Pub` / `S_GPT`）。
目标分数表见 `PLAN.md` §0。

---

## 2. 环境

本机（AMD Radeon 8060S / gfx1151，ROCm 7.13 Windows nightly）上**必须**先过一层 shim：

```python
import rnaqwen.env.shim   # 必须早于任何 multimolecule / transformers import
```

原因见 `src/rnaqwen/env/shim.py` 顶部注释：这个 torch 构建**没有编入分布式后端**，
而 MultiMolecule 的运行时依赖 `danling` 在 import 阶段就无条件引用
`torch.distributed.distributed_c10d` / `torch.distributed.tensor` / `dist.ReduceOp`。
shim 把这些缺失符号补成占位符（我们从不真正做分布式通信）。

### 解释器

| 路径 | 用途 |
|---|---|
| **`C:\ana\envs\comfyui\python.exe`** | **主环境**（按用户指定使用）—— torch ROCm 就在这个环境里；本项目需要的 `transformers 5.9.0 / peft / accelerate / sentence-transformers / rouge-score / nltk` 已装进它 |
| `C:\qwen-api\RNAQwen\.venv\Scripts\python.exe` | 备用：venv `--system-site-packages`，包集合与上面一致，不改动 comfyui |

> ⚠️ **装进 comfyui 环境会造成版本变动**（记录在案，便于回滚）：
> `transformers 4.57.6 → 5.9.0`、`huggingface-hub 0.36.2 → 1.33.0`、`hf-xet 1.5.2 → 1.7.0`，
> 新增 `peft 0.21.2 / accelerate 1.15.0 / sentence-transformers 6.1.0 / rouge-score 0.1.2 / nltk 3.10.3`。
> 其中 transformers 5 是 `multimolecule` 的硬需求（它 import `transformers.utils.output_capturing`，4.57 没有），
> 不升就没法 load RNA-FM。若 ComfyUI 的自定义节点依赖旧版 transformers，用 `.venv` 跑本项目。

跑任何脚本前：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"      # 否则 OpenMP 冲突直接崩（libiomp5md 重复初始化）
$env:PYTHONIOENCODING="utf-8"         # 否则中文输出乱码
& C:\ana\envs\comfyui\python.exe scripts\<脚本> ...
```

不需要手动 import shim —— `import rnaqwen` 时**自动**装载（见 `src/rnaqwen/__init__.py`）。

---

## 3. 目录

```
PLAN.md              计划与验收标准（含已核实的资源清单）
PROGRESS.md          复现日志：每步做了什么、验收到什么程度、踩了什么坑
src/rnaqwen/
    env/shim.py      AMD ROCm Windows 环境修补
    eval/            ROUGE + 语义 P/R/F1
    data/            数据管线（源 / 记录 / 注释 / QA / 闸门 / 提示词）
    llm.py           生成器（本地 HF / OpenAI 兼容）
scripts/
    00_download_models.py        从 HuggingFace 下载（慢，仅备用）
    00b_download_modelscope.py   从 ModelScope 下载（**主力**，实测快 180 倍）
    01_probe_rnafm.py            PLAN §7.1 RNA-FM 探针
    03_eval_selfcheck.py         PLAN §7.3 评测脚本自检
    10_collect.py                P0 采集 (序列, 文献)
    11_generate_annotations.py   P0 D&C 注释
    12_generate_qa.py            P0 指令数据 + 序列泄漏闸门
    13_guard_check.py            P0 数据验收
models/              权重（Qwen3-8B / rnafm / bert-base-uncased / pubmedbert）
data/raw/            采集产物
data/processed/      注释与 QA
reports/             自检与验收报告
```

---

## 4. 跑法（按顺序）

```powershell
$py = "C:\ana\envs\comfyui\python.exe"          # 主环境（见 §2）
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"

# 权重（走 ModelScope；下完**必须**校验 sha256，见 §5.5）
& $py scripts\00b_download_modelscope.py --repo Qwen/Qwen3-8B --dest models\Qwen3-8B --workers 4
& $py scripts\00c_verify_models.py --repo Qwen/Qwen3-8B --dest models\Qwen3-8B --include safetensors

# §7.1 RNA-FM 探针（维度 + 上下文上限）
& $py scripts\01_probe_rnafm.py --model models\rnafm --out reports\rnafm_probe.json
& $py scripts\01b_probe_rnafm_limit.py --model models\rnafm --device cpu

# §7.3 评测脚本自检（ROUGE 对数 + 语义指标不变量 + 消融响应方向）
& $py scripts\03_eval_selfcheck.py --embedder models\bert-base-uncased

# P0 数据管线
& $py scripts\10_collect.py --target 50 --out data\raw\rna_literature_50.jsonl
& $py scripts\11_generate_annotations.py --in data\raw\rna_literature_50.jsonl --out data\processed\annotations_50.jsonl --model models\Qwen3-8B
& $py scripts\12_generate_qa.py --in data\processed\annotations_50.jsonl --out data\processed\rnaqa_50.jsonl --model models\Qwen3-8B
& $py scripts\13_guard_check.py --in data\processed\rnaqa_50.jsonl --report reports\guard_50.json

# 管线逻辑离线自检（不需要大模型权重，用假生成器把 11/12/13 的代码路径跑一遍）
& $py scripts\99_selftest_pipeline.py --limit 20
```

---

## 5. 数据源的一处替换（要写进报告）

PLAN 原定**RNAcentral**。实测本机到 EBI 全线不可达：

```
rnacentral.org/api/v1/...   SSLError: Max retries exceeded
ftp.ebi.ac.uk/...           ReadTimeout
europepmc.org/...           SSLError
```

而 NCBI eutils 通。所以最小版本改成：

```
esearch nuccore（短 RNA + 长度闸门）
  → efetch rettype=gb（一次拿到序列 + DEFINITION + ORGANISM + PUBMED）
  → efetch pubmed（摘要）
```

拿到的同样是 **(RNA 序列, 文献文本)** 对，管线结构不变；差异在报告里说清楚：
RNAcentral 的 RNA type / 描述字段在 NCBI 侧要用 `DEFINITION` 替代，字段不等价。

---

## 5.5 权重下载的两条铁律（血泪）

**① 下完必须校 sha256，`size` 对不算下完。**

Qwen3-8B 16.38 GB 下完、`safetensors` 加载成功（399/399 张量全到位），
但 5 个 shard 里 **4 个 sha256 对不上、字节数一个不差** —— 模型能加载、能跑、
**只输出乱码**（`React matplotlib sys React ...`）。
成因是多轮断点续传拼了不一致的字节 + 一度有两个进程同时写同一个文件。

```
& $py scripts\00c_verify_models.py --repo Qwen/Qwen3-8B --dest models\Qwen3-8B --include safetensors
```

**② 一个文件只能有一个写者，且先写 `.part` 再原子改名。**

`00b_download_modelscope.py` 已经改成这个模式。别为了"快点"并行起两个任务拉同一个文件。

另外两条实测结论：
* **HF 不是慢，是单连接被限速**：`aria2c -x16 -s16 -k1M` 拉 RNA-FM 的 398 MB 只要 25 秒，
  而单连接只有 0.06 MB/s。ModelScope 单连接就有 2–11 MB/s，两者都别开太多并发（会互相拖死）。
* `huggingface_hub` **不要**配 `HF_ENDPOINT=https://hf-mirror.com` —— 镜像不回 HEAD 元数据头，
  会抛 `FileMetadataError: Distant resource does not seem to be on huggingface.co`。

---

## 6. 口径声明（必须随任何数字一起出现）

1. 原版**报数不是全量训出来的** —— 他们用的是 5K RNA / 121K QA 子集，所以打平的门槛也是这个量级。
2. 原版**评测集未开源**，本仓库的评测集是**重建**的，
   **绝对分与原报数不可直接比较**，只能比"同一把尺子下的相对关系"。
3. 生成器从 GPT-4o/4o-mini 换成**本地 Qwen**，生成分布不完全一致。
