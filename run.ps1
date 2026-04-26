

# 路径配置
$TRAIN_DATA = "data\train.bin"
$VAL_DATA   = "data\val.bin"
$VOCAB      = "bpe\outputs\qwen_style_tokenizer.json"
$MERGES     = "bpe\outputs\qwen_style_tokenizer.json"
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
Write-Host "工作目录: $(Get-Location)"
Write-Host "输出目录: $OUT_DIR"
Write-Host "日志文件: $LOG_FILE"
Write-Host "====================================================="

# 核心：前台直接运行python，不跳转、不后台、不吞报错
python main/run_train_model.py `
    --train_data $TRAIN_DATA `
    --val_data $VAL_DATA `
    --tokenizer_vocab $VOCAB `
    --tokenizer_merges $MERGES `
    --out_dir $OUT_DIR `
    --batch_size 16 `
    --max_iters 1000 `
    --eval_interval 100 `
    --eval_iters 20 `
    --log_interval 10 `
    --vocab_size 32000 `
    --context_length 256 `
    --n_head 16 `
    --theta 10000 `
    --n_layers 4 `
    --d_model 512 `
    --d_ff 1344 `
    --weight_decay 1e-1 `
    --max_norm 1.0 `
    --max_lr 6e-4 `
    --min_lr 6e-5 `
    --warmup_iters 200 `
    --lr_decay_iters 3600

Write-Host "`n训练结束"