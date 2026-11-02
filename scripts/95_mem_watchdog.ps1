# 95_mem_watchdog.ps1 —— 35B 训练期间的内存看门狗
#
# 为什么需要
# ----------
# 这台机器是 APU：GPU 显存就是系统内存。物理 63.6 GB（其余被 BIOS 划给核显），
# 提交上限 = 63.6 + 页面文件 ≈ 149 GB。35B 训练自己就占 GPU 提交 88.5 GB
# + 进程私有 36.8 GB，再叠加常驻服务就贴到上限。
#
# 2026-10-12 实测：提交可用跌到 0.56 GB，按当时速率**约 23 秒归零**；
# 归零后触发系统级 OOM（0x800705AF："页面文件太小，无法完成操作"），
# 连 echo 都跑不了，训练必死。
#
# 它做什么
# --------
#   1. 每 60 秒采一行：提交可用 / 物理可用 / 训练步数，滚存到 reports/watchdog.log
#   2. 跌破阈值时追加醒目告警行（带上当时的 step），并记录首次越线的时刻
#   3. 训练进程消失时记一行并退出
#
# 它**不**做什么
# --------------
# 不自动杀训练。checkpoint 已经是原子落盘（写 *.pt.tmp → 验 zip 中央目录 →
# os.replace），而且每 300 步或每 30 分钟必存一个 —— 所以最坏情况只是丢 ≤30 分钟，
# 没有必要为了保档去动一个正在正常收敛的训练。
#
# 用法
# ----
#   Start-Process powershell -ArgumentList '-NoProfile','-File',"$PWD\scripts\95_mem_watchdog.ps1" -WindowStyle Hidden
#   停止： Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
#          Where-Object { $_.CommandLine -match '95_mem_watchdog' } |
#          ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

param(
    [int]$IntervalSec = 60,
    [double]$WarnGB   = 15.0,     # 低于此值记 WARN
    [double]$AlarmGB  = 8.0,      # 低于此值记 ALARM（再往下就是 23 秒归零那条线）
    [int]$FullEvery   = 5,        # 每 N 个采样记一条常规行（其余只在上报阈值时写）
    [int]$GraceSamples = 15       # 连续多少次看不到训练进程才判定结束（15 x 60s = 15 min 宽限）
)

$ErrorActionPreference = "Continue"
$root = "C:\qwen-api\RNAQwen"
$log  = Join-Path $root "reports\watchdog.log"
Set-Location $root

function Write-Line($s) {
    $stamp = (Get-Date -Format 'MM-dd HH:mm:ss')
    "$stamp  $s" | Out-File -FilePath $log -Encoding utf8 -Append
}

function Get-Trainer {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match '50_routeC_e2e' } |
        Select-Object -First 1
}

function Get-Step {
    # 从训练日志里抠最后一条 step 行，报错就返回 -1（不影响看门狗本身）
    try {
        $tl = Join-Path $root "reports\D_qwen36_sos.log"
        if (-not (Test-Path $tl)) { return -1 }
        $m = Select-String -Path $tl -Pattern '^\s+step\s+(\d+)' -ErrorAction SilentlyContinue |
             Select-Object -Last 1
        if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    } catch { }
    return -1
}

Write-Line ("=== 看门狗启动  interval=${IntervalSec}s  warn<${WarnGB}GB  alarm<${AlarmGB}GB ===")

$i = 0
$inAlarm = $false
$alarmSince = $null
$miss = 0          # 连续多少次看不到训练进程

while ($true) {
    $i++
    $t = Get-Trainer
    if (-not $t) {
        # ⚠️ 不能一看不见就退出。重启训练时进程会消失十几分钟，看门狗会自杀
        #    —— 2026-10-12 踩过：重启完就没人盯内存了，而那恰恰是最危险的时段。
        #    给它一个宽限期，只有连续 $GraceSamples 次都不在才判定真的结束。
        $miss++
        if ($miss -eq 1) {
            Write-Line ("训练进程不在（可能在重启），宽限 $GraceSamples 次采样后再判")
        }
        if ($miss -ge $GraceSamples) {
            Write-Line ("=== 训练进程连续 $GraceSamples 次采样不在，看门狗退出 ===")
            break
        }
        Start-Sleep -Seconds $IntervalSec
        continue
    }
    if ($miss -gt 0) {
        Write-Line ("训练回来了（PID $($t.ProcessId)），中断了 $miss 次采样")
        $miss = 0
    }

    $os = Get-CimInstance Win32_OperatingSystem
    $commitGB = [math]::Round($os.FreeVirtualMemory / 1MB, 2)
    $physGB   = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $step     = Get-Step

    $tag = ""
    if ($commitGB -lt $AlarmGB) {
        $tag = "  <<< ALARM 提交可用已跌破 ${AlarmGB}GB"
        if (-not $inAlarm) {
            $inAlarm = $true
            $alarmSince = Get-Date
            Write-Line ("!!! ALARM 开始  commit=${commitGB}GB  step=$step  PID=$($t.ProcessId)")
        }
    } elseif ($commitGB -lt $WarnGB) {
        $tag = "  < WARN"
        if ($inAlarm) {
            $inAlarm = $false
            $mins = [math]::Round(((Get-Date) - $alarmSince).TotalMinutes, 1)
            Write-Line ("!!! ALARM 解除  持续 ${mins} 分钟  commit=${commitGB}GB  step=$step")
        }
    } else {
        if ($inAlarm) {
            $inAlarm = $false
            $mins = [math]::Round(((Get-Date) - $alarmSince).TotalMinutes, 1)
            Write-Line ("!!! ALARM 解除  持续 ${mins} 分钟  commit=${commitGB}GB  step=$step")
        }
    }

    if ($tag -or ($i % $FullEvery -eq 0)) {
        # Win32_Process 没有 PrivateMemorySize64，得回头问 Get-Process
        $priv = ""
        $gp = Get-Process -Id $t.ProcessId -ErrorAction SilentlyContinue
        if ($gp) { $priv = "  训练私有 $([math]::Round($gp.PrivateMemorySize64/1GB,1))GB" }
        Write-Line ("commit=${commitGB}GB  phys=${physGB}GB  step=$step$priv$tag")
    }

    Start-Sleep -Seconds $IntervalSec
}
