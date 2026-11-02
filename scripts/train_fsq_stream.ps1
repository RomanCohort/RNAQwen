param([int]$Threads = 32)
$env:PYTHONIOENCODING = 'utf-8'
$env:RNAQ_THREADS = "$Threads"
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$py = 'C:\qwen-api\RNAQwen\.venv\Scripts\python.exe'
$log = 'C:\qwen-api\RNAQwen\logs\fsq_stream.log'
$mutex = New-Object System.Threading.Mutex($false, 'Global\FSQStreamTrain')
if (-not $mutex.WaitOne(0)) { exit 0 }
function L($m) {
  $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $m
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}
L "start (threads=$Threads)"
& $py -X utf8 -u 'C:\qwen-api\RNAQwen\scripts\49_stream_tokenizer.py' --jsonl 'C:\qwen-api\RNAQwen\data\processed\rnacentral_2m.jsonl' --n-seq 2187209 --val 8000 --chunk 2048 --steps-per-chunk 12 --out 'C:\qwen-api\RNAQwen\checkpoints\fsq_stream' *>> $log
L "exit=$LASTEXITCODE"
