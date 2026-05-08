# 路径配置
$TRAIN_DATA = "data/train.bin"
$VAL_DATA   = "data/val.bin"
$VOCAB      = "bpe/tokenizer"
$OUT_ROOT   = "train_logs"

# 纯数字时间戳，无中文无空格
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUT_DIR   = Join-Path $OUT_ROOT "run_$TIMESTAMP"
$LOG_FILE  = Join-Path $OUT_DIR "train.log"

# 创建目录
if (-not (Test-Path $OUT_DIR)){
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
}

Write-Host "====================================================="
Write-Host "work dir: $(Get-Location)"
Write-Host "out dir: $OUT_DIR"
Write-Host "log files: $LOG_FILE"
Write-Host "====================================================="


python main/run_train_model.py `
    --train_data $TRAIN_DATA `
    --val_data $VAL_DATA `
    --tokenizer_vocab $VOCAB `
    --out_dir $OUT_DIR `
    --batch_size 64 `
    --max_iters 50000 `
    --eval_interval 100 `
    --eval_iters 20 `
    --log_interval 10 `
    --vocab_size 8192 `
    --context_length 256 `
    --n_head 8 `
    --theta 10000 `
    --n_layers 12 `
    --d_model 512 `
    --d_ff 2048 `
    --weight_decay 1e-2 `
    --max_norm 1.0 `
    --max_lr 1e-3 `
    --min_lr 1e-5 `
    --warmup_iters 1000 `
    --lr_decay_iters 48000 `
    2>&1

Write-Host "`ntrain done"