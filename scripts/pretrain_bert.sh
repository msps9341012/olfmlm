#!/bin/bash

RANK=0
WORLD_SIZE=1
CUDA_LAUNCH_BLOCKING=1

python3 -m olfmlm.pretrain_bert "$@"
