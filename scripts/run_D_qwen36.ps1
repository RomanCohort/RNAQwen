# D_qwen36 —— Qwen3.6-35B-A3B 全量训练启动脚本
#
# 用这个而不是手敲长命令：配置里每一项都有理由，改之前先看下面的注释。
#
# ⚠️ 本文件必须存成 **UTF-8 with BOM**。存成无 BOM 的 UTF-8，PowerShell 5 会按 GBK 读，
#    中文注释里的引号会把参数吃掉，报一堆莫名其妙的错。
#    修法： $c = Get-Content -Raw -Encoding UTF8 .\scripts\run_D_qwen36.ps1
#           [IO.File]::WriteAllText($p, $c, (New-Object Text.UTF8Encoding $true))
#
# 前置（缺一不可）
#   data/processed/rnaqa_seqtask_full.jsonl   1,496,820 条 / 149,682 RNA / 10 个可复算模板
#   data/processed/_rfa_full_hidden.f16       145,201 条 rfam 隐藏态（68 GiB，副缓存）
#   data/processed/_fsq_full_pos.pt           4,481 条 seq8 隐藏态（主缓存）
#   data/processed/_rfa_full_index.pt         副缓存索引
#   checkpoints/fsq_full/code_emb_contentinit_d2048.pt   内容初始化（对齐 35B 的 0.5547 行范数）
#
# 用法
#   powershell -File scripts/run_D_qwen36.ps1 -Smoke                        # 3 步冒烟，先跑这个
#   powershell -File scripts/run_D_qwen36.ps1                               # 正式，默认 SOS
#   powershell -File scripts/run_D_qwen36.ps1 -Adapter lora -Tag D_qwen36_lora
#
# 配置理由速查
#   数据用 seqtask_full，不用 rnaqa_merged：后者的 loss 有 58.8% 落在信道无关的散文上
#     （PLAN §13.8），梯度主方向是「忽略信道」。
#   副缓存必须用全量那份：旧的 _rfam_hidden.f16 只有 19,726 条，而 rfa 支有 145,201 条。
#   内容初始化必须用 d2048 那份：8B 那份是 (4096,4096)，形状就对不上。
#   batch 8 / token-budget 1200：实测 35B int4 独占约 26 s/步，30,000 步约 9 天。
#     这是全流程最贵的一项 —— 步数和 batch 成反比，改小省不了总时间。
#   --trim-every 50：mmap 读过的页会留在 WorkingSet 里，不 trim 就是 1.6 GiB/层 x 40 层，
#     直接顶爆提交量、无 traceback 硬崩。RSS 靠它压到 0.02 GiB。
#   checkpoint：原子落盘 + 动态时机 + 只留最近 10 个。
#     **原子**：先写 *.pt.tmp，验过 zip 中央目录，再 os.replace 顶上正式名。
#       必须这样 —— torch.save 直写时写到一半被 OOM 杀掉会留下截断的 .pt，而它看起来
#       合法（文件在、后缀对、体积也不离谱）。实测踩过：step_000600.pt 254.9 MB
#       （正常 322 MB），torch.load 报 "failed finding central directory"；
#       更要命的是 resume 的 cands[-1] 按名字排序**正好会挑中这个坏文件**。
#       原子写保证正式名永远指向完整文件：要么是新的，要么是上一个。
#     **动态**：--save-every 300 步 是步数下限；--save-minutes 30 是墙上时钟兜底，
#       谁先到算谁。纯按步数时，步速多慢都不会提前存 —— 35B 实测 18.5 s/步，
#       300 步 = 92 分钟，崩一次最多丢 1.5 小时。加时间兜底后丢失窗口与步速无关。
#     **轮转**：--keep-ckpt 10，单个约 322 MB，10 个约 3.2 GB。
#       best.pt / final.pt 不参与轮转，也不被清理 —— 它们是唯一不可再生的两个档
#       （best 是早停要载回的，final 是最终产物）。
#   早停：监视 val 上 A 臂答案 CE，判据 need = max(min_delta*|best|, se)，**必须带 se 那一项**。
#     eval-n 32 时 CE 标准误约 0.15，而 min_delta*|best| 只有 0.004 ——
#     不带 se 就是完全跟着噪声走。patience 4 / eval-every 750 / min-steps 1500：
#     从 1500 步起每 750 步查一次，最迟 4500 步收敛。
#     另外早停必须真的 break 退出（已修）—— 只打印不退出会一路跑满 --steps。
#   --ablation-every：训练中跑 5 臂消融的频率。**默认 750，跟 eval-every 同步。**
#     5 臂 = 真码 / 零表 / 随机表 / 常数码 / 置换，每臂 eval-n 次单样本前向
#     （per_token_logp 是逐条循环的），35B 上一次约 5~8 min，占吞吐 2~3%。
#     值在哪：**置换臂是最干净的判据** —— 它对码表做一个随机双射，
#     边缘分布、尺度、表的大小全不变，只切断「码 ↔ 内容」的对应。
#       零表低 + 置换低 -> 码真的被读了
#       零表低 + 置换高 -> 模型只用了「RNA 位置有个向量」，没用码的身份
#     历史实测（reports/*.log 里的「最终消融」）：
#       A_v2_4096  置换 28.9%  -> 真读了
#       B_seq8     置换 40.7%  -> 真读了（seq8 那支 80% 的那轮）
#       C_merged   置换 50.6%  -> **没用码的身份**（正是后来判定被稀释的那轮）
#     所以它不只是终点的验尸报告，还是**过程中的预警**：早一步知道现在这轮
#     在往 A/B 那条路走还是往 C_merged 那条路走。
#     注意 50_ 末尾那段「最终消融」**总会跑**（跟 ablation-every 无关），
#     所以终点判据是免费的；--ablation-every 多给的是**趋势**。
#   --diag 0：诊断套件一次 4 臂 x diag-n 次前向，35B 上太贵。要看就单独跑 76_/87_。
#
# 适配器怎么选
#   lora：peft 标准 LoRA，8B 上验过（B_seq8 三臂 80.0% / 7.0% / 10.5%）。
#   sos ：arXiv 2607.16252v1 的正交多专家 LoRA。参数量与同 rank 的 LoRA **完全相等**
#         （(in+out)*r_tot），不额外吃显存和时间。K 个正交子空间按功能分工，
#         收益预期在**分模态**场景（码信道 vs 文本提示）更大 —— 所以 freetext 上才该见真章。
#
# target 预设：默认 D，不是 C
#   C = 9 个注意力投影，190 模块 / 8.13 M
#   D = C + shared_expert.{gate,up,down}_proj，310 模块 / 10.58 M（+30%）
#   为什么加 shared_expert：MoE 的路由专家是 top-8/256 **稀疏**激活，一个 token 只碰 8 个，
#     而且不同 token 碰的不同 —— 「两个模态争同一个子空间」这件事在专家那边是被**打散**的。
#     而 shared_expert 对**每一个 token** 都生效，是 FFN 侧唯一「所有人都必须过」的路径，
#     也是那里唯一存在模态冲突的地方。它同时也是几个 FFN 投影里**唯一没被量化**的
#     （bnb 只量化了 20,480 个专家 Linear），所以现成就是 nn.Linear，SOS 直接挂得上。
#   ⚠️ 不误伤专家：patch_moe_experts 把专家命名成 `gate_up` / `down`（ModuleList 子项名是下标
#      '0'..'255'），而权重名里的 `gate_up_proj` 会被 `up_proj` 后缀命中 —— 但那是**张量名**，
#     不是模块名。模块名 `gate_up` / `down` / `'0'` 对 D 的三个后缀全部为假。
#     这一条用 scripts/_scratch/_count_targets.py 核过，别凭权重表猜。
#
# 35B 上 SOS 挂得上的原因：int4 只量化 MoE 专家（40 层 x 512 = 20,480 个），
#   注意力层和 shared_expert 保持 nn.Linear —— preset C 的 9 个名字在真实权重里对应
#   190 个 nn.Linear（10 层全注意力 x4 + 30 层线性注意力 x5）。
#   挂载正确性见 scripts/_scratch/_test_sos_lora.py。
#   （8B 那条路是整模型量化、全是 bnb Linear，SOS 挂不上，50_ 里有显式 RuntimeError。）

param(
    [switch]$Smoke,
    [int]$Steps = 30000,
    [string]$Mode = "int4",
    [ValidateSet("lora", "sos")]
    [string]$Adapter = "sos",
    [string]$Tag = "",
    [double]$OrthoCoef = 1.0,
    [double]$SaveMinutes = 30,
    [switch]$Resume,
    [int]$VerifyLoss = 5,
    [string]$Preset = "D",
    [int]$AblationEvery = 750
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# HIP 分配器：expandable_segments 让大块按需扩展，专治碎片。
# 这个模型被拆成 20,480 个专家 Linear4bit（外加 190 个注意力投影），小块分配极多，
# 碎片实测能占到十几 GB。两个名字设同一个东西，ROCm 认 HIP_ 那个、
# 新版本 torch 也认 CUDA_ 那个，都写上省得猜。
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$PY = "C:\ana\envs\comfyui\python.exe"
Set-Location "C:\qwen-api\RNAQwen"

# ⚠️ 调 python 之前必须放宽错误处理。
#    $ErrorActionPreference='Stop' 配 `2>&1 | Tee-Object` 时，原生程序写往 stderr 的
#    **第一行**会被当成终止性错误（NativeCommandError），管道当场被拆掉、python 被带死。
#    而 torch 一启动就往 stderr 写 UserWarning（tensor_numpy.cpp:219），于是表现为
#    「装载到一半无声退出，exit 1，日志停在某个中途行」—— 看着像 OOM，其实是自己掐的。
#    上面那份严格设定只服务于下面的路径检查。
$ErrorActionPreference = "Continue"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
if (-not (Test-Path $PY)) { throw "找不到 python 解释器 $PY" }

if (-not $Tag) { $Tag = "D_qwen36_$Adapter" }

# ---- 开机前检查一：残留的 50_routeC ----
# ROCm 上 python 崩在第 1 步之后**不会真的退出** —— 它挂在上下文销毁上，
# 继续占着 18 GiB 显存和日志文件句柄。新一轮的 Tee-Object 会直接打不开日志
# （"being used by another process"），或者在一个看不见的地方少 18 GiB 显存。
# 这个坑连着踩了两次，所以宁可开机前先扫一遍。
$stray = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
           Where-Object { $_.CommandLine -match '50_routeC_e2e' })
foreach ($sp in $stray) {
    Write-Host "[pre] 停掉残留的 50_routeC PID $($sp.ProcessId)"
    taskkill /F /T /PID $sp.ProcessId 2>&1 | Out-Null
}
if ($stray.Count -gt 0) { Start-Sleep -Seconds 8 }

# ---- 开机前检查二：日志文件可写 ----
# 不检查的话，Tee-Object 会在装载前一刻失败（"being used by another process"），
# 报错在最后一行，很容易被当成训练崩了。
# ⚠️ 必须先判存在：文件不存在时 [IO.File]::Open 抛的是 FileNotFoundException，
#    和「被占用」是同一个 catch —— 直接进 catch 再去 Remove-Item 又会抛一次，
#    于是「日志还不存在」被误报成「日志被占用且删不掉」。这个假阳性踩过一次。
$LogPath = Join-Path (Get-Location) "reports\$Tag.log"
if (Test-Path -LiteralPath $LogPath) {
    $locked = $false
    try {
        $fs = [System.IO.File]::Open($LogPath, 'Open', 'ReadWrite', 'None')
        $fs.Close(); $fs.Dispose()
    } catch { $locked = $true }
    if ($locked) {
        try { Remove-Item -LiteralPath $LogPath -Force -ErrorAction Stop }
        catch { throw "日志 $LogPath 被别的进程占着且删不掉：$($_.Exception.Message)" }
    }
}

$common = @(
    "--hid2", "data/processed/_rfa_full_hidden.f16",
    "--hid2-idx", "data/processed/_rfa_full_index.pt",
    "--base", "qwen36-35b",
    "--qwen36-mode", $Mode,
    "--lora-preset", "$Preset",
    "--pool", "4",
    "--lora-r", "8",
    "--init-code-emb", "checkpoints/fsq_full/code_emb_contentinit_d2048.pt",
    "--batch", "8",
    "--token-budget", "1200",
    "--ent-coef", "1.0",
    "--tau", "0",
    "--trim-every", "50",
    "--ablation-every", "$AblationEvery",
    "--diag", "0",
    "--adapter", $Adapter
)

if ($Adapter -eq "sos") {
    $common += @(
        "--sos-k", "4",
        "--sos-gamma-max", "2.5",
        "--sos-dropout", "0.05",
        "--sos-ortho-coef", "$OrthoCoef"
    )
}

if ($Smoke) {
    $cli = $common + @(
        "--data", "data/processed/_smoke35b.jsonl",
        "--steps", "3",
        "--log-every", "1",
        "--eval-every", "0",
        "--save-every", "0",
        "--tag", "smoke35b_$Adapter",
        "--save-dir", "runs/smoke35b_$Adapter/artifacts"
    )
    Write-Host "[smoke] 3 步，验装载 + SOS 挂载数 + 前向 + 反相传回 FSQ"
    & $PY -u scripts/50_routeC_e2e.py @cli 2>&1 | Tee-Object -FilePath "reports/smoke35b_$Adapter.log"
    exit $LASTEXITCODE
}

$cli = $common + @(
    "--data", "data/processed/rnaqa_seqtask_full.jsonl",
    "--steps", "$Steps",
    "--log-every", "20",
    "--early-stop", "1",
    "--min-delta", "0.005",
    "--patience", "4",
    "--min-steps", "1500",
    "--eval-every", "750",
    "--eval-n", "32",
    "--save-every", "300",
    "--save-minutes", "$SaveMinutes",
    "--keep-ckpt", "10",
    "--answer-only-loss", "1",
    "--verify-loss", "$VerifyLoss",
    "--tag", $Tag,
    "--save-dir", "runs/$Tag/artifacts"
)

# 从最新 checkpoint 续训。默认关；-Resume 打开。
# 对齐的 checkpoint 里存了适配器 + FSQ + 码表 + 优化器状态 + RNG，
# load_ckpt 会整套恢复，所以 step 是从 301 接上去的，不是从 1 重来。
if ($Resume) {
    $ck = "runs/e2e_$Tag/ckpt"
    if (-not (Test-Path $ck)) { throw "-Resume 指定了，但 $ck 不存在" }
    $latest = Get-ChildItem $ck -Filter "step_*.pt" | Sort-Object Name | Select-Object -Last 1
    if (-not $latest) { throw "-Resume 指定了，但 $ck 下没有 step_*.pt" }

    # ⚠️ 换 target 预设后续训是**静默变化**，必须拦一下。
    #    表现全都"正常"：注意力那 190 个适配器接得上（strict=False），
    #    新增的 120 个从零开始，优化器因为参数量变了 load_state_dict 失败、
    #    被 catch 成一行"从零开始优化器"—— loss 曲线照跑，实验已经变味。
    #    checkpoint 里存了 args，所以能自证；比一下就知道。
    $prev = & $PY -c "import torch,sys; a=torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('args',{}); print(a.get('lora_preset','?'), a.get('adapter','?'), a.get('sos_k','?'), sep='|')" $latest.FullName 2>$null
    if ($prev) {
        $f = $prev -split '\|'
        Write-Host "[$Tag] $($latest.Name) 里记录的是 preset=$($f[0]) adapter=$($f[1]) K=$($f[2])"
        if ($f[0] -ne $Preset) {
            Write-Host ""
            Write-Host "  *** 警告：preset 从 $($f[0]) 变成 $Preset，这是在**改实验**，不是续训 ***"
            Write-Host "      注意力适配器会接上、新增的从零开始、优化器会被重置。"
            Write-Host "      如果本意是跑一个新的对照臂，请去掉 -Resume 从头训。"
            Write-Host ""
        }
    } else {
        Write-Host "[$Tag] （读不出 checkpoint 里的 args，跳过 preset 一致性检查）"
    }
    $cli += @("--resume", $ck)
    Write-Host "[$Tag] 续训起点 $($latest.Name)"
}

Write-Host "[$Tag] adapter=$Adapter mode=$Mode steps 上限 $Steps（早停会提前切）"
Write-Host "[$Tag] 装载 40 层约 9 min（int4 约 18 GiB，20,480 个专家 Linear）"
Write-Host "[$Tag] sos-ortho-coef=$OrthoCoef"
Write-Host "[$Tag] ckpt: 每 300 步或每 $SaveMinutes 分钟（谁先到算谁），原子落盘，只留最近 10 个"
Write-Host "[$Tag] loss: 只在答案位置算 logits；前 $VerifyLoss 步与 HF out.loss 逐位比对"
Write-Host "[$Tag] 分配器: $env:PYTORCH_HIP_ALLOC_CONF"
Write-Host "[$Tag] target 预设 $Preset（D = 注意力 190 + shared_expert 120 = 310 模块）"
& $PY -u scripts/50_routeC_e2e.py @cli 2>&1 | Tee-Object -FilePath "reports/$Tag.log"
exit $LASTEXITCODE
