@echo off
rem Full MMLU + C-Eval sweep against the 8098 GPU instance (RNAQwen-preview Q4_K_M).
rem 5-shot is the standard MMLU/C-Eval protocol, so the numbers are comparable to
rem published tables.  Detached: started via Win32_Process.Create.
setlocal
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONIOENCODING=utf-8"
cd /d C:\qwen-api\RNAQwen
set "PY=C:\ana\envs\comfyui\python.exe"
set "SC=scripts\_scratch\_mc_bench.py"
set "OUT=reports\bench_rnaqwen_preview"
set "LOG=logs\bench_rnaqwen_preview.log"
if not exist "%OUT%" mkdir "%OUT%"
echo ==== START %DATE% %TIME% ==== >> "%LOG%"
"%PY%" -u "%SC%" --bench mmlu  --port 8098 --nshot 5 --workers 16 --out "%OUT%\mmlu_5shot.jsonl"  >> "%LOG%" 2>&1
echo ==== mmlu done %DATE% %TIME% ==== >> "%LOG%"
"%PY%" -u "%SC%" --bench ceval --port 8098 --nshot 5 --workers 16 --out "%OUT%\ceval_5shot.jsonl" >> "%LOG%" 2>&1
echo ==== ALL DONE %DATE% %TIME% ==== >> "%LOG%"
