# 手动运行命令单（RNA-GPT 复现）

> 为什么给你手动跑：从 agent 的工具 shell 里拉起来的长任务，已经静默死过 5 次以上
> （无 traceback、无 abort、事件日志里也没有）。你自己的终端不会。
> 所有命令都在 **PowerShell 窗口**里跑，工作目录 `C:\qwen-api\RNAQwen`。

---

## 0. 每次开新窗口先贴这三行（缺一不可）

```powershell
cd C:\qwen-api\RNAQwen
$env:KMP_DUPLICATE_LIB_OK="TRUE"          # 不设：torch/ROCm 加载直接 abort
$env:PYTHONIOENCODING="utf-8"             # 不设：中文输出乱码/抛 UnicodeEncodeError
$PY = "C:\ana\envs\comfyui\python.exe"    # 必须用这个解释器
```

验证环境活着（10 秒）：

```powershell
& $PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. 先跑这个：ROUGE 收敛曲线（**决定后面怎么走，别跳过**）

```powershell
& $PY scripts\31_stage1_rouge_curve.py `
    --ckpt-dir checkpoints\stage1_run1 `
    --compress-latent 64 `
    --n-samples 24 --max-new-tokens 150 --every 2 `
    --resume `
    --out reports\stage1_rouge_curve.json
```

* 已有 27 个 checkpoint，`--every 2` 取 14 个点，**约 1.9 小时**。
* **断了就重贴同一条命令**——`--resume` 会读已有的 json 跳过测过的点。
* 想知道为什么先跑它：`Δloss` 是逐 token 平均，答案变长变准会把它摊薄。
  实测 step500→8000，Δloss 判据说"在塌缩"，ROUGE-1 却从 0.208 涨到 0.303
  （原论文 Modality Fusion 只有 0.0922）。**Δloss 不能替代论文的指标。**
* 看结果：`notepad reports\stage1_rouge_curve.json`，找 `rouge1` 那条曲线什么时候拐平。
  * 还在涨 → Stage 1 没训够，续跑；拐平在 8000 附近 → 记下来，E 轨的有效步数就这么多。
  * 想看某几个点：`--only 500,2000,8000,13500`

---

## 2. E 轨（改进轨）Stage 1 续跑 —— 从 step 13500 接到 34880

```powershell
& $PY scripts\20_train_stage1.py `
    --data data\processed\rnaqa_rnacentral_all.jsonl `
    --target-field annotation `
    --llm models\Qwen3-8B --encoder models\rnafm `
    --epochs 10 --batch-size 1 --grad-accum 1 `
    --lr 1e-4 --lr-final 8e-5 --warmup-steps 2048 --weight-decay 0.05 `
    --compress-latent 64 --compress-method resampler `
    --tag run1 `
    --val-from data\evalset --eval-every 200 --eval-samples 32 `
    --resume-from checkpoints\stage1_run1\projection_step13500.pt `
    --out checkpoints\stage1_run1
```

* 进度 **13500 / 34880 = 38.7%**，压缩后 ~1.0 s/step，**剩 ~5.9 小时**。
* ⚠️ 上面每个参数都必须和原 run 一致（checkpoint 里存了 config 可以核对）。
  尤其 `--grad-accum 1`：续跑的跳过逻辑是 `done_micro < start_step × grad_accum`，
  改了它会从错的位置接上。`--out` 也别改，否则新写的 checkpoint 会散到别的目录。
* ⚠️ checkpoint **没存 optimizer 状态**，续跑时 AdamW 的动量是重启的，loss 会先跳一下再降。
  这是已知的，不是坏了。
* 续跑再断，把 `--resume-from` 换成当前最大的那个 `projection_step*.pt` 即可。

---

## 3. R 轨（复现轨，论文原样）Stage 1

```powershell
& $PY scripts\20_train_stage1.py `
    --data data\processed\rnaqa_rnacentral_all.jsonl `
    --target-field annotation `
    --llm models\Qwen3-8B --encoder models\rnafm `
    --epochs 10 --batch-size 1 --grad-accum 1 `
    --lr 1e-4 --lr-final 8e-5 --warmup-steps 2048 --weight-decay 0.05 `
    --tag R `
    --val-from data\evalset --eval-every 200 --eval-samples 32 `
    --out checkpoints\stage1_R
```

* 34880 步 × 2.87 s ≈ **27.8 小时**。这是**投稿要报的那条**（和论文同一配置）。
* 跑之前先看第 1 节的曲线结论：如果 E 轨 8000 步就拐平，R 轨也可以只跑到拐点，
  但**报的时候必须说明跑了多少步**，不能拿 3 个 epoch 去对比论文的 10 个 epoch。
* 不要和 E 轨的分数并列汇报（PLAN §7）。

---

## 4. 置零探针（P1 完成判据）

E 轨 checkpoint（自动读出 `compress_latent=64` 并装回压缩器）：

```powershell
& $PY scripts\22_zero_encoder_probe.py `
    --ckpt checkpoints\stage1_run1\projection_step13500.pt `
    --data data\processed\rnaqa_rnacentral_all.jsonl --limit 24 `
    --out reports\zero_probe_run1_step13500.json
```

R 轨 checkpoint（不带压缩器，`--compress-latent` 不用给）：

```powershell
& $PY scripts\22_zero_encoder_probe.py `
    --ckpt checkpoints\stage1_R\projection_final.pt `
    --data data\processed\rnaqa_rnacentral_all.jsonl --limit 24 `
    --out reports\zero_probe_R_final.json
```

* 判据：置零编码器后 Δloss 要明显上升。**但阈值别当圣旨**——
  实测 E 轨 Δloss 只有 0.047，模型却明显在用序列。和 §1 的 ROUGE 一起读。
* `PASS/FAIL` 看退出码，报告在 `--out` 指的文件里。

---

## 5. Stage 2 指令微调（LoRA）

**E 轨（推荐先跑这个，快 2.87 倍）**——压缩配置必须和 Stage 1 配对：

```powershell
& $PY scripts\23_train_stage2.py `
    --data data\processed\rnaqa_n6.jsonl `
    --llm models\Qwen3-8B --encoder models\rnafm `
    --projection checkpoints\stage1_run1\projection_step13500.pt `
    --compress-latent 64 `
    --preset attn_mlp --variant both --kind both `
    --epochs 3 --batch-size 1 --grad-accum 8 `
    --lr 2e-4 --warmup-steps 200 --weight-decay 0.05 `
    --save-every 500 --out checkpoints\stage2_run1
```

**R 轨**（不给 `--compress-latent`）：

```powershell
& $PY scripts\23_train_stage2.py `
    --data data\processed\rnaqa_n6.jsonl `
    --llm models\Qwen3-8B --encoder models\rnafm `
    --projection checkpoints\stage1_R\projection_final.pt `
    --preset attn_mlp --variant both --kind both `
    --epochs 3 --batch-size 1 --grad-accum 8 `
    --lr 2e-4 --warmup-steps 200 --weight-decay 0.05 `
    --out checkpoints\stage2_R
```

* ⚠️ **`--epochs 3` 不是 10**。论文的 Stage 2 是 10 epoch，但我们的样本是
  80,907 条，10 epoch ≈ 202 小时（按实测 0.9 s/样本外推）。
  先用 3 epoch（≈ 61 小时）跑通，看 `run_summary.json` 里的 loss 曲线再决定要不要加。
* 实测吞吐（2026-10-09 冒烟）：6 步 0.1 min = **0.9 s/样本**，序列长 135–169 token
  （64 个 latent + 短 QA）。E 轨 Stage 2 是这个速度；R 轨不压缩会长 2.87 倍。
* ⚠️ `--kind` 取的是数据里的**原始值** `open_ended` / `closed`，
  不是 `open` / `closed`。写 `open` 会筛出 0 条样本然后报错。
  一般直接 `--kind both --variant both` 就行。
* 先干跑一遍确认数据/预算／不加载模型（秒级）：

```powershell
& $PY scripts\23_train_stage2.py --data data\processed\rnaqa_n6.jsonl --limit 100 --dry-run
```

* 配错压缩会**立刻报错**，不会静默跑歪（`build_stage2_model` 里做了配对校验）。
* 消融阶梯（PLAN 里的三档），一次只改一个变量：

```powershell
# 只调 attn
& $PY scripts\23_train_stage2.py ... --preset attn ...
# 默认 attn+mlp
& $PY scripts\23_train_stage2.py ... --preset attn_mlp ...
# r=64
& $PY scripts\23_train_stage2.py ... --preset attn_mlp_r64 ...
```

---

## 6. 推理 + 评测

```powershell
# Stage 1 之后的模态融合（没有 LoRA）
& $PY scripts\30_infer.py `
    --eval data\evalset\test.jsonl --mode modality_fusion `
    --projection checkpoints\stage1_run1\projection_step13500.pt `
    --compress-latent 64 `
    --limit 100 --out preds\stage1_run1.jsonl

# Stage 2 之后的完整 RNA-GPT
& $PY scripts\30_infer.py `
    --eval data\evalset\test.jsonl --mode rna_gpt `
    --projection checkpoints\stage2_run1\final\projection.pt `
    --compress-latent 64 `
    --lora checkpoints\stage2_run1\final\lora `
    --limit 100 --out preds\stage2_run1.jsonl

# 打分（ROUGE + 语义 + 分类指标）
& $PY scripts\04_eval.py --pred preds\stage2_run1.jsonl --out reports\eval_stage2_run1.json
```

* ⚠️ 以前 `30_infer.py` 的压缩器**权重从来没载入过**（随机初始化），E 轨推理结果是错的。
  现在会从 checkpoint 里 load，配错会直接报错退出。
* `--compress-latent` 现在可以省略（自动从 checkpoint 读），但显式写出来更安全。

---

## 7. 长任务的正确起法（还是建议用你自己的窗口）

想让它脱离窗口也能活，用 `Start-Process` 起一个**独立**的 PowerShell，
日志重定向到文件（不要用 `-WindowStyle Hidden`，实测会死）：

```powershell
Start-Process powershell -ArgumentList @(
  '-NoExit','-Command',
  'cd C:\qwen-api\RNAQwen; $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; & C:\ana\envs\comfyui\python.exe scripts\20_train_stage1.py ... 2>&1 | Tee-Object -FilePath logs\stage1_R.log'
)
```

最稳的还是：开一个 PowerShell 窗口，贴命令，让它开着。

---

## 当前状态速查

| 项 | 状态 |
|---|---|
| 数据 `data/processed/rnaqa_n6.jsonl` | ✅ 3,488 RNA / 80,907 QA，guard PASS |
| 评测集 `data/evalset/` | ✅ train 2766 / val 339 / test 383 |
| 模型权重 | ✅ Qwen3-8B（5 shard sha256 全过）、rnafm、Qwen3-Embedding-8B |
| E 轨 Stage 1 | 🟡 **38.7%**（step 13500 / 34880），checkpoint 27 个 |
| R 轨 Stage 1 | ⬜ 未跑（只有 step500/1000/1500 的早期残留） |
| ROUGE 曲线 | ⬜ 未产出 —— **先跑这个** |
| Stage 2 | ✅ E 轨加载路径已修好并冒烟通过（0.9 s/样本，压缩器正确载入/存回） |
| 磁盘 | 87.8 GB 可用 |
