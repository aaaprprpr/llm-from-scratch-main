#!/bin/bash

# 在项目根目录执行此脚本以启动语言模型训练。

# 数据必须先完成分词处理 - 否则请先运行分词脚本。
# 关于分词器训练与实现，可参考：https://github.com/Siyuan-Harry/bpe-optimized-from-scratch
TRAIN_DATA="data\train.bin"
VAL_DATA=  "data\val.bin"

# 你的分词器词汇表与合并规则文件路径
VOCAB= "bpe\outputs\qwen_style_tokenizer.json"
MERGES="bpe\outputs\qwen_style_tokenizer.json"

# 为每一次运行创建输出目录，记录所有输出与日志
OUT_ROOT="train_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/run_$TIMESTAMP"
LOG_FILE="$OUT_DIR/train.log"

mkdir -p $OUT_DIR

echo "====================================================="
echo "工作目录: $(pwd)"
echo "输出目录: $OUT_DIR"
echo "日志文件: $LOG_FILE"
echo "====================================================="

# nohup 使进程在登出后仍在后台继续运行
nohup python -u main/run_train_model.py \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --tokenizer_vocab $VOCAB \
    --tokenizer_merges $MERGES \
    --out_dir $OUT_DIR \
    --batch_size 64 \
    --max_iters 4200 \
    --eval_interval 100 \
    --eval_iters 20 \
    --log_interval 10 \
    --vocab_size 65536 \
    --context_length 256 \
    --n_head 16 \
    --theta 10000 \
    --n_layers 4 \
    --d_model 512 \
    --d_ff 1344 \
    --weight_decay 1e-1 \
    --max_norm 1.0 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 200 \
    --lr_decay_iters 3600 \
    > $LOG_FILE 2>&1 &

#--use_wandb \ 注释掉表示关闭

# 打印进程 ID
echo "训练已启动，进程 PID: $!"
echo "查看实时日志: tail -f $LOG_FILE"
