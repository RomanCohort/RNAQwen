# Chain watcher: wait for the R-track Stage 1 to exit, then automatically start the E-track
# (token compression).  Single GPU => the two Stage-1 runs must be serialized, and the R-track
# takes ~20h, so this has to run unattended.
#
# NOTE: this file is deliberately ASCII-only.  Windows PowerShell 5.1 reads .ps1 files as ANSI
# unless they carry a UTF-8 BOM, so non-ASCII comments here get mangled into parse errors.
#
# WARNING: E-track numbers must NEVER be reported side by side with R-track numbers (PLAN.md §7).

$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"

$py = "C:\ana\envs\comfyui\python.exe"
$root = "C:\qwen-api\RNAQwen"
$rTrackPid = [int]$env:CHAIN_R_PID
$log = "$root\logs\chain_e_track.log"

function Note([string]$msg) {
    "[chain] $msg  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append $log -Encoding utf8
}

Note "waiting for R-track PID $rTrackPid to exit"
while (Get-Process -Id $rTrackPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}
Note "R-track exited; waiting 30s for the GPU to be released"
Start-Sleep -Seconds 30

$proj = "$root\checkpoints\stage1\projection_final.pt"
if (Test-Path $proj) {
    Note "R-track checkpoint present: $proj"
} else {
    Note "WARNING: projection_final.pt not found - R-track may have failed; starting E-track anyway"
}

Note "starting E-track: --compress-latent 64 --compress-method resampler"
& $py "$root\scripts\20_train_stage1.py" `
    --data "$root\data\processed\rnaqa_rnacentral_all.jsonl" `
    --target-field annotation `
    --llm "$root\models\Qwen3-8B" --encoder "$root\models\rnafm" `
    --epochs 10 --batch-size 1 --grad-accum 1 `
    --lr 1e-4 --lr-final 8e-5 --warmup-steps 2048 --weight-decay 0.05 `
    --val-from "$root\data\evalset" --eval-every 250 --eval-samples 48 `
    --compress-latent 64 --compress-method resampler `
    --tag e_compress --out "$root\checkpoints\stage1" *>> "$root\logs\stage1_e_track.log"

Note "E-track exited with code $LASTEXITCODE"
