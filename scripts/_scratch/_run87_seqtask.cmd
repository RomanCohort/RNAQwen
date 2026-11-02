@echo off
REM One-shot: 87_channel_by_source.py over the seqtask provenance (rfa + seq8).
REM
REM The data file matters. rnaqa_merged.jsonl (the script default) carries
REM freetext/rfam/seq8 and answers the wrong question; rnaqa_seqtask_full.jsonl
REM carries rfa/seq8, which is the provenance the paper section needs.
REM
REM Why a wrapper instead of a direct call: the run takes about half an hour, and long
REM tasks started from an agent tool shell have died silently on this box before (see
REM RUN.md). This is launched detached through run_hidden.vbs so it outlives the shell.
REM
REM Env vars from RUN.md, both required: KMP_DUPLICATE_LIB_OK stops the torch load from
REM aborting, PYTHONIOENCODING keeps the non-GBK glyphs in this script's own log lines
REM from raising UnicodeEncodeError and killing a run that is otherwise finished.
REM
REM ASCII only, no BOM.
setlocal
cd /d C:\qwen-api\RNAQwen
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
set PY=C:\ana\envs\comfyui\python.exe

"%PY%" -u scripts\87_channel_by_source.py --ckpt runs/e2e_D_qwen36_sos/ckpt/final.pt --data data/processed/rnaqa_seqtask_full.jsonl --per-source 40 --out reports/channel_by_source_D_final_seqtask.json > reports\87_seqtask.log 2>&1
echo exit=%ERRORLEVEL% >> reports\87_seqtask.log
