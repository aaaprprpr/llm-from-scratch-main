@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 训练数据路径
set TRAIN_DATA=train.bin
set VAL_DATA=val.bin

:: 分词器文件
set VOCAB=trained_tokenizer/tokenizer.json
set MERGES=trained_tokenizer/tokenizer.json

:: 输出目录
set OUT_ROOT=train_logs
for /f "tokens=1-3 delims=:/ " %%a in ("%time%") do set hh=%%a&set mm=%%b&set ss=%%c
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%hh%%mm%%ss%
set OUT_DIR=%OUT_ROOT%\run_%TIMESTAMP%
set LOG_FILE=%OUT_DIR%\train.log

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo =====================================================
echo 工作目录: %cd%
echo 输出目录: %OUT_DIR%
echo 日志文件: %LOG_FILE%
echo =====================================================

:: 启动训练（Windows 版，后台运行）
start /B python main/run_train_model.py ^
    --train_data %TRAIN_DATA% ^
    --val_data %VAL_DATA% ^
    --tokenizer_vocab %VOCAB% ^
    --tokenizer_merges %MERGES% ^
    --out_dir %OUT_DIR% ^
    --batch_size 64 ^
    --max_iters 4200 ^
    --eval_interval 100 ^
    --eval_iters 20 ^
    --log_interval 10 ^
    --vocab_size 151665 ^
    --context_length 256 ^
    --n_head 16 ^
    --theta 10000 ^
    --n_layers 4 ^
    --d_model 512 ^
    --d_ff 1344 ^
    --weight_decay 1e-1 ^
    --max_norm 1.0 ^
    --max_lr 6e-4 ^
    --min_lr 6e-5 ^
    --warmup_iters 200 ^
    --lr_decay_iters 3600 > %LOG_FILE% 2>&1

echo 训练已启动！
echo 查看日志: type %LOG_FILE%
echo 实时查看日志: powershell Get-Content %LOG_FILE% -Wait
pause