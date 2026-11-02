@echo off
rem 8098: benchmark-only instance. Same model as 8099 but on the iGPU (HIP),
rem so a 15k-question MMLU/C-Eval sweep finishes in tens of minutes instead of days.
rem Started detached via Win32_Process.Create, so it survives this shell.
rem 16 slots + --cache-reuse 256: every question in a subject shares the same
rem few-shot prefix, so only the new question block should ever be re-evaluated.
setlocal
set "PATH=C:\Program Files\AMD\ROCm\7.1\bin;C:\qwen-api\build-hip\bin;%PATH%"
set "TEMP=C:\qwen-api\tmp"
set "TMP=C:\qwen-api\tmp"
cd /d C:\qwen-api\build-hip\bin
"C:\qwen-api\build-hip\bin\llama-server.exe" -m "C:\qwen-api\RNAQwen\gguf\RNAQwen-preview-Q4_K_M.gguf" --host 127.0.0.1 --port 8098 -ngl 999 -c 32768 --parallel 16 --cont-batching -t 16 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --ubatch-size 1024 --batch-size 4096 --cache-reuse 256 --no-warmup > C:\qwen-api\logs\llama-8098.out.log 2> C:\qwen-api\logs\llama-8098.err.log
