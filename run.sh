#!/bin/bash
# define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

source .venv/bin/activate &&
# torchrun --nproc-per-node=8  run.py --data MMVet --model Qwen2-VL-7B-Instruct --verbose;
# torchrun --nproc-per-node=6  run.py --data MMBench_DEV_CN_V11 --model Qwen2-VL-7B-Instruct --verbose



# keep retry util success
# for attempt in {1..2}; do
    # python run.py --data MME --model Qwen2-VL-7B-Instruct --verbose
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=4 run.py --data Open_MI --model Qwen2-VL-7B-Instruct --verbose --shots=4
    # CUDA_VISIBLE_DEVICES=1 python run.py --data BLINK --model Qwen2-VL-7B-Instruct --verbose
    if [ $? -eq 0 ]; then
        echo -e "$GREEN Command executed  successfully. Exiting.$NC"
        break  
    else
        echo -e "$RED Command  failed. Retrying in 30 seconds... $NC"
        # sleep 30 
    fi
# done

# if [ "$attempt" -eq 2 ]; then
#     echo -e "$RED Maximum attempts reached. Exiting. $NC"
# fi