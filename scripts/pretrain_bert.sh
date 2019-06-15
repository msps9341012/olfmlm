#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 -m Megatron-LM.pretrain_bert \
    --batch-size 2 \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir cache_dir \
    --tokenizer-model-type bert-base-uncased \
    --vocab-size 30522 \
    --train-data wikipedia \
    --presplit-sentences \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 128 \
    --train-iters 10000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --lr-decay-iters 990000 \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers 2 \
    --epochs 10 \
    --bert-config-file /h/stephaneao/trained_berts/config_file.json \
    --save /h/stephaneao/trained_berts/ 
