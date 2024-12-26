import argparse
import json
import os

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CHESS")
    parser.add_argument("--model", type=str, default="GPT4V")
    parser.add_argument("--n_shots", type=int, default=0)
    return parser.parse_args()


def post_log(data, model, n_shots):
    for shot in range(0, n_shots + 1):
        file_name = f"./outputs/{model}/shots_{shot}/{model}_{data}_acc.json"
        with open(
            file_name,
            "r",
        ) as f:
            score = json.load(f)
            print(f"shots :{shot}, score: {score}")
        wandb.log(data={f"{data}": score})


def main():
    args = parse_args()
    wandb.init(project="ICLBoom")
    post_log(args.data, args.model, args.n_shots)


if __name__ == "__main__":
    os.chdir("/home/zxy/codes/working/ICLBoom/VLMEvalKit")
    main()
