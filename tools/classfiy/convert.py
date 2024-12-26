import base64
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main(folder_path, output_tsv1, output_tsv2):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            prefix_index = filename.split(".")[0]
            id_index = prefix_index.split("_")[-1]
            answer_index = prefix_index.split("_")[0]
            image_path = os.path.join(folder_path, filename)
            image_base64 = convert_image_to_base64(image_path)
            data.append(
                {
                    "index": f"chess_{id_index}",
                    "question": "what is it?",
                    "answer": answer_index,
                    "image": image_base64,
                }
            )

    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)

    train_df.to_csv(output_tsv1, sep="\t", index=False)
    test_df.to_csv(output_tsv2, sep="\t", index=False)


if __name__ == "__main__":
    folder_path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/datasets/animals-test"  # 替换为你的文件夹路径
    output_tsv1 = "./query.tsv"  # 替换为你想要输出的训练集文件名
    output_tsv2 = "./support.tsv"  # 替换为你想要输出的测试集文件名
    main(folder_path, output_tsv1, output_tsv2)
