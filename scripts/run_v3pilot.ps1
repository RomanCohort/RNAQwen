param(
  [int]$Steps = 500,
  [string]$Tag = "v3pilot"
)
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$env:RNAQ_THREADS = '8'
$py = 'C:\qwen-api\RNAQwen\.venv\Scripts\python.exe'
$log = "C:\qwen-api\RNAQwen\logs\$Tag.log"

function L($m) {
  $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $m
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

L "start steps=$Steps tag=$Tag"
& $py -X utf8 -u 'C:\qwen-api\RNAQwen\scripts\50_routeC_e2e.py' `
    --data 'C:\qwen-api\RNAQwen\data\processed\rnaqa_v3_pilot.jsonl' `
    --hid2 'C:\qwen-api\RNAQwen\data\processed\_rfam_hidden.f16' `
    --hid2-idx 'C:\qwen-api\RNAQwen\data\processed\_rfam_index.pt' `
    --fsq 'C:\qwen-api\RNAQwen\checkpoints\fsq_stream\tokenizer.pt' `
    --init-code-emb 'C:\qwen-api\RNAQwen\checkpoints\fsq_stream\code_emb_contentinit.pt' `
    --base qwen36-35b --qwen36-mode int4 --adapter sos --lora-r 8 `
    --token-budget 512 --steps $Steps --min-steps $Steps `
    --eval-every 100 --log-every 50 --eval-n 25 --diag 1 --diag-n 40 `
    --ablation-every 100 --tag $Tag *>> $log
L "exit=$LASTEXITCODE"
