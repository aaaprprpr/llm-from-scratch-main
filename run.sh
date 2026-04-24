#!/bin/bash

# Run this script in the project root directory to start training the language model. 

# the data must be tokenized first - otherwise run the tokenization script first.
# For training and implementing the tokenizer, please refer to https://github.com/Siyuan-Harry/bpe-optimized-from-scratch
TRAIN_DATA="tokenized_data/your_train_data.bin" 
VAL_DATA="tokenized_data/your_val_data.bin" 

# path to your tokenizer vocab & merges file
VOCAB="trained_tokenizer/vocab_of_your_tokenizer.json" 
MERGES="trained_tokenizer/merges_of_your_tokenizer.json" 

# make one output directory for every run to record any outputs & logs
OUT_ROOT="train_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/run_$TIMESTAMP"
LOG_FILE="$OUT_DIR/train.log"

mkdir -p $OUT_DIR

echo "====================================================="
echo "Work Dir: $(pwd)"
echo "Output Dir: $OUT_DIR"
echo "Log File: $LOG_FILE"
echo "====================================================="

# nohup makes the process run in the background even after logging out
nohup python -u run_train_model.py \
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
    --vocab_size 10000 \
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

#--use_wandb \ 不出现意味着false

# 打印进程 ID
echo "Training started with PID: $!"
echo "To monitor the log: tail -f $LOG_FILE"