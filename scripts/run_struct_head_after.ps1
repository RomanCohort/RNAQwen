param()
$ErrorActionPreference = 'Continue'
$log = 'C:\qwen-api\RNAQwen\logs\struct_head.log'
function L($m) {
  $line = '[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] ' + $m
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}
L 'waiting for codebook training'
while ($true) {
  $p = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*49_stream_tokenizer*' -and $_.Name -eq 'python.exe' })
  if ($p.Count -eq 0) { break }
  Start-Sleep -Seconds 60
}
L 'codebook done; starting structure probe'
$env:PYTHONIOENCODING = 'utf-8'
$env:RNAQ_THREADS = '8'
$env:OMP_NUM_THREADS = '8'
& 'C:\qwen-api\RNAQwen\.venv\Scripts\python.exe' -X utf8 -u 'C:\qwen-api\RNAQwen\scripts\61_struct_head.py' *>> $log
L ('probe exit=' + $LASTEXITCODE)
