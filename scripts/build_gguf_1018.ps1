$ErrorActionPreference = 'Continue'
Set-Location C:\qwen-api\RNAQwen
$py = 'C:\ana\envs\comfyui\python.exe'
$common = @('--ckpt','runs/e2e_cont_rff/ckpt/final.pt',
            '--out','gguf/RNAQwen-preview-1018-Q8_0.gguf',
            '--q4','gguf/RNAQwen-preview-1018-Q4_K_M.gguf')
$log = 'C:\qwen-api\logs\export-1018.log'
function L($m) { Add-Content -LiteralPath $log -Value ('{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $m) -Encoding UTF8 }
L '=== build start ==='
foreach ($ph in 'patch','convert','quantize','verify') {
    L "phase $ph start"
    & $py -u scripts\98_export_gguf.py --phase $ph @common *>> $log
    $rc = $LASTEXITCODE
    L "phase $ph exit=$rc"
    if ($rc -ne 0) { L '=== build FAILED ==='; exit $rc }
}
L '=== build OK ==='