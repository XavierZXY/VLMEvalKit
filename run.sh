#!/bin/bash
# define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# 定义变量
# CLEVR CLEVR_SQ Open_MI Operator_Induction CHESS
DATASET='CHESS'
RETRY_RANGE=10
MODEL='GPT4V'

source .venv/bin/activate &&
# torchrun --nproc-per-node=8  run.py --data MMVet --model Qwen2-VL-7B-Instruct --verbose;
# torchrun --nproc-per-node=6  run.py --data MMBench_DEV_CN_V11 --model Qwen2-VL-7B-Instruct --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI --model InternVL2-8B --verbose --shots=0

# keep retry util success
for attempt in $(seq 0 $RETRY_RANGE); do
    export OMP_NUM_THREADS=24
    python run.py --data "$DATASET" --model "$MODEL" --verbose --shots="$attempt"
    # python run.py --data BLINK --model GPT4V --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data CLEVR CLEVR_SQ --model Qwen2-VL-2B-Instruct qwen_chat --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data CLEVR_SQ --model Qwen2-VL-7B-Instruct --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI Operator_Induction --model Qwen2-VL-2B-Instruct Qwen2-VL-7B-Instruct --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI --model Qwen2-VL-7B-Instruct --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Operator_Induction --model Qwen2-VL-7B-Instruct --verbose --shots=2
    # CUDA_VISIBLE_DEVICES=1 python run.py --data BLINK --model Qwen2-VL-7B-Instruct --verbose
    if [ $? -eq 0 ]; then
        echo -e "$GREEN Command executed  successfully. Exiting. Let's sleep 10s.$NC"
        sleep 10
    else
        echo -e "$RED Command  failed. Retrying in 10 seconds... $NC"
        sleep 10 
    fi
done

# post log upload
if [ "$attempt" -eq $RETRY_RANGE ]; then
    echo -e "$RED Maximum attempts reached. Exiting. $NC"
    python tools/post_log.py --data "$DATASET" --model "$MODEL" --n_shots "$attempt"
fi