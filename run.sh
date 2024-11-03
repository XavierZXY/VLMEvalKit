#!/bin/bash
# define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

source .venv/bin/activate &&
# torchrun --nproc-per-node=8  run.py --data MMVet --model Qwen2-VL-7B-Instruct --verbose;
# torchrun --nproc-per-node=6  run.py --data MMBench_DEV_CN_V11 --model Qwen2-VL-7B-Instruct --verbose


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI --model InternVL2-8B --verbose --shots=0

# keep retry util success
for attempt in {0..4}; do
    # python run.py --data MME --model Qwen2-VL-7B-Instruct --verbose
    # Open_MI, CLEVR, Operator_Induction 
    # qwen_chat Qwen2-VL-2B-Instruct Qwen2-VL-7B-Instruct InternVL2-8B idefics2_8b
    # 
    export OMP_NUM_THREADS=24
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI CLEVR Operator_Induction --model InternVL2-8B --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI Operator_Induction --model Qwen2-VL-2B-Instruct Qwen2-VL-7B-Instruct --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Open_MI --model Qwen2-VL-7B-Instruct --verbose --shots="$attempt"
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 run.py --data Operator_Induction --model Qwen2-VL-7B-Instruct --verbose --shots=2
    # CUDA_VISIBLE_DEVICES=1 python run.py --data BLINK --model Qwen2-VL-7B-Instruct --verbose
    if [ $? -eq 0 ]; then
        echo -e "$GREEN Command executed  successfully. Exiting. Let's sleep 60s.$NC"
        sleep 15
    else
        echo -e "$RED Command  failed. Retrying in 30 seconds... $NC"
        sleep 60 
    fi
done

# if [ "$attempt" -eq 2 ]; then
#     echo -e "$RED Maximum attempts reached. Exiting. $NC"
# fi