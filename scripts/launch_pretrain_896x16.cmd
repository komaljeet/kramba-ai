@echo off
REM The real 896x16 hybrid pretrain (started 2026-09-29). Same command starts it
REM fresh AND resumes it: train_edge_conversational.py auto-resumes from the
REM checkpoint dir (model + optimizer + LR schedule, #112/#165).
REM
REM Before relaunching, check nothing else is on the GPU (nvidia-smi) and that no
REM other python is writing this log -- two writers on one log is a known incident.
REM
REM Runs from this worktree (pinned code), reads data and writes checkpoints/logs
REM in the main tree. Detached launch:
REM   powershell -Command "Start-Process cmd -ArgumentList '/c','C:\Users\komal\Documents\Codes\kramba-ai-wt-pretrain896\scripts\launch_pretrain_896x16.cmd' -WindowStyle Hidden"
REM
REM 118116 steps = ceil(30 x 129,013,721 body params / 32,768 tokens per step).
set MAIN=C:\Users\komal\Documents\Codes\kramba-ai
set PYTHONUNBUFFERED=1
cd /d C:\Users\komal\Documents\Codes\kramba-ai-wt-pretrain896
"%MAIN%\.venv\Scripts\python.exe" -u train_edge_conversational.py ^
  --data-dir "%MAIN%\data_cache" ^
  --arch hvn-hybrid --attn-layers 4,12 --d-model 896 --num-layers 16 ^
  --seq-len 4096 --batch-size 1 --grad-accum 8 ^
  --grad-checkpoint --chunked-loss 512 ^
  --pretrain-mix 0.0,0.0,0.0,1.0 --seed 0 ^
  --max-steps 118116 ^
  --save-every-steps 250 --keep-checkpoints 3 --milestone-every 10000 ^
  --checkpoint-dir "%MAIN%\checkpoints\pretrain_896x16_real" ^
  >> "%MAIN%\logs\pretrain_896x16_real.log" 2>&1
