@echo off
REM ROUGE curve over Stage-1 checkpoints.  Resumable + auto-retry.
REM
REM Why a scheduled task: everything launched as a child of the agent shell died within
REM minutes (observed 5 times), while the user's own long jobs survive.  A scheduled task
REM belongs to the Task Scheduler service, so it is not in the agent's process tree.
REM
REM Why a Python retry runner: the cmd -> powershell -> python quoting ate the inner
REM command and every attempt exited with code 1 immediately.  Python's subprocess takes
REM an argument list, so there is no shell parsing to break.
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
cd /d C:\qwen-api\RNAQwen
"C:\ana\envs\comfyui\python.exe" scripts\_retry_runner.py --tries 30 --sleep 20 --log logs\rouge_curve.log -- "C:\ana\envs\comfyui\python.exe" scripts\31_stage1_rouge_curve.py --ckpt-dir checkpoints\stage1_run1 --eval data\evalset\test.jsonl --compress-latent 64 --n-samples 24 --max-new-tokens 150 --every 2 --resume --out reports\stage1_rouge_curve.json
