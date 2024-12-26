import base64
import csv
import json
import os
import pickle
import random
from collections import defaultdict
from io import BytesIO

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_data(datasets_path, datasets_name):
    query = dict()
    images = dict()
    # open tsv file
    csv.field_size_limit(10000000)
    with open(
        datasets_path + datasets_name + ".tsv", "r", encoding="utf-8"
    ) as tsvfile:
        # 创建一个csv.reader对象，指定制表符为字段分隔符
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        headers = next(tsvreader)

        # 遍历文件中的每一行
        for row in tsvreader:
            query[row[headers.index("index")]] = row[headers.index("question")]
            images[row[headers.index("index")]] = row[headers.index("image")]
    return query, images


def retireve_icl_data(
    query,
    images,
    icl_query,
    icl_image,
    path,
    shots: int = 10,
    retrieval_method="SQ",
) -> dict | None:
    """_summary_

    Args:
        query (_type_): _description_
        images (_type_): _description_
        icl_query (_type_): _description_
        icl_image (_type_): _description_
        path (_type_): _description_
        shots (int, optional): _description_. Defaults to 10.
        retrieval_method (str, optional): Defaults to "SQ".
            SQ: Similar Query (Y)
            SI: Similar Image (N)
            herding: Herding  (Y)

    Returns:
        dict : {query_index: [retrieved_index1, retrieved_index2, ...]}
    """

    def _load_or_extract_features(file_path, data, extract_fn):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                features = pickle.load(f)
        else:
            features = extract_fn(data)
            with open(file_path, "wb") as f:
                pickle.dump(features, f)
        return features

    def _extract_text_features(data, model, device):
        features = dict()
        for key, text in tqdm(data.items()):
            tokenized_text = clip.tokenize([text]).to(device)
            with torch.no_grad():
                features[key] = model.encode_text(tokenized_text).cpu().numpy()
        return features

    def _extract_image_features(data, preprocess, model, device):
        features = dict()
        for key, img in tqdm(data.items()):
            image = Image.open(BytesIO(base64.b64decode(img)))
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features[key] = model.encode_image(image_input).cpu().numpy()
        return features

    # Define the file paths for storing features
    query_features_file = path + "query_features.pkl"
    images_features_file = path + "images_features.pkl"
    icl_query_features_file = path + "icl_query_features.pkl"
    icl_images_features_file = path + "icl_images_features.pkl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load or extract query features
    query_features = _load_or_extract_features(
        query_features_file,
        query,
        lambda data: _extract_text_features(data, model, device),
    )

    # Load or extract icl_query features
    icl_query_features = _load_or_extract_features(
        icl_query_features_file,
        icl_query,
        lambda data: _extract_text_features(data, model, device),
    )

    # Load or extract images features
    images_features = _load_or_extract_features(
        images_features_file,
        images,
        lambda data: _extract_image_features(data, preprocess, model, device),
    )

    # Load or extract icl_images features
    icl_images_features = _load_or_extract_features(
        icl_images_features_file,
        icl_image,
        lambda data: _extract_image_features(data, preprocess, model, device),
    )

    if retrieval_method == "SQ":
        # Compute cosine similarity for each query individually
        top_queries = {}
        for q_key, q_vector in query_features.items():
            q_vector = q_vector.squeeze()
            icl_vectors = np.array(list(icl_query_features.values())).squeeze()

            similarities = np.dot(icl_vectors, q_vector.T).squeeze()
            similarities /= np.linalg.norm(icl_vectors, axis=1) * np.linalg.norm(
                q_vector
            )

            # Get the indices of the top 'shots' similar queries
            top_indices = np.argsort(similarities)[-shots:][::-1]

            # Retrieve the top similar queries
            top_queries[q_key] = [list(icl_query.keys())[i] for i in top_indices]
        return top_queries
    elif retrieval_method == "SI":
        # Compute cosine similarity for each image individually
        top_queries = {}
        for q_key, q_vector in images_features.items():
            q_vector = q_vector.squeeze()
            icl_vectors = np.array(list(icl_images_features.values())).squeeze()

            similarities = np.dot(icl_vectors, q_vector.T).squeeze()
            similarities /= np.linalg.norm(icl_vectors, axis=1) * np.linalg.norm(
                q_vector
            )

            # Get the indices of the top 'shots' similar images
            top_indices = np.argsort(similarities)[-shots:][::-1]

            # Retrieve the top similar images
            top_queries[q_key] = [list(icl_image.keys())[i] for i in top_indices]
        return top_queries
    elif retrieval_method == "herding":
        top_queries = {}
        icl_vectors = np.array(list(icl_images_features.values())).squeeze()
        mean_vector = np.mean(icl_vectors, axis=0)
        selected_indices = []
        for _ in range(shots):
            valid_indices = np.isfinite(icl_vectors).all(axis=1)
            distances = np.linalg.norm(
                icl_vectors[valid_indices] - mean_vector, axis=1
            )
            max_index = np.argmax(distances)
            selected_indices.append(max_index)
            mean_vector = (
                mean_vector * len(selected_indices) + icl_vectors[max_index]
            ) / (len(selected_indices) + 1)
            icl_vectors[
                max_index
            ] = -np.inf  # Exclude this index from future selection
        for key, value in query_features.items():
            top_queries[key] = [
                list(icl_image.keys())[i] for i in selected_indices
            ]
        return top_queries
    elif retrieval_method == "random":
        top_queries = {}
        icl_vectors = np.array(list(icl_images_features.values())).squeeze()
        selected_indices = random.sample(range(len(icl_vectors)), shots)
        for key, value in query_features.items():
            top_queries[key] = [
                list(icl_image.keys())[i] for i in selected_indices
            ]
        return top_queries


def read_tsv_file(retireve_data):
    path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/icltools/datasets/clevr/"
    support_name = "support.tsv"
    with open(path + support_name, "r", encoding="utf-8") as tsvfile:
        # 创建一个csv.reader对象，指定制表符为字段分隔符
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        headers = next(tsvreader)
        # 读取数据
        # 将数据转换为字典，键为index列的值
        data_dict = {row[headers.index("index")]: row for row in tsvreader}

        for index in retireve_data.keys():
            print(f"The query is: {index} \n")
            for i in retireve_data[index]:
                if str(i) in data_dict:
                    print(data_dict[str(i)][headers.index("question")])
                    print("\n")


def generate_tsv_file(
    retireve_data, path, query_name, support_name, retireve_method
):
    # to store the full data of retireve data(index,question, answer, image)
    retireve_full_data = defaultdict(list)
    retireve_data_headers = []
    # open tsv fileds
    csv.field_size_limit(10000000)
    with open(path + support_name + ".tsv", "r", encoding="utf-8") as tsvfile:
        # 创建一个csv.reader对象，指定制表符为字段分隔符
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        headers = next(tsvreader)
        retireve_data_headers = headers
        # 读取数据
        data_dict = {row[headers.index("index")]: row for row in tsvreader}
        for index in retireve_data.keys():
            for i in retireve_data[index]:
                data_temp = dict()
                for key, value in zip(retireve_data_headers, data_dict[str(i)]):
                    data_temp[key] = value
                retireve_full_data[index].append(data_temp)

    # 读取 query 文件
    query_data = []
    with open(path + query_name + ".tsv", "r", encoding="utf-8") as query_file:
        tsvreader = csv.DictReader(query_file, delimiter="\t")
        query_data = list(tsvreader)

    output_tsv = path + query_name + retireve_method + "_retrieved.tsv"
    with open(output_tsv, "w", newline="", encoding="utf-8") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")

        # Step 5: Write the headers to the TSV file
        headers = ["index", "question", "answer", "image", "support"]
        writer.writerow(headers)

        # Step 6: Iterate over each item in the JSON data
        for idx, item in enumerate(query_data):
            # Step 7: Extract the required fields
            row = [
                item.get("index", ""),
                item.get("question", ""),
                item.get("answer", ""),
                item.get("image", ""),
                retireve_full_data.get(item.get("index", ""), ""),
            ]
            # Step 8: Write the extracted fields to the TSV file
            writer.writerow(row)


def main():
    # Change the current working directory to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/icltools/datasets/clevr/"
    # path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/icltools/datasets/clevr/"
    # datasets_name = "Open_MI.tsv"
    path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/icltools/datasets/chess/"
    query_name = "query"
    support_name = "support"

    # get the query and images data
    query = dict()
    images = dict()
    query, images = load_data(path, query_name)
    icl_query = dict()
    icl_images = dict()
    icl_query, icl_images = load_data(path, support_name)

    # get the index of different retireve methods.(SI, SQ, SQA ...)
    retrieval_method = "SI"
    retireve_data = retireve_icl_data(
        query,
        images,
        icl_query,
        icl_images,
        path,
        shots=10,
        retrieval_method=retrieval_method,
    )
    # read tsv file
    # read_tsv_file(retireve_data)
    # generate new tsv data file
    generate_tsv_file(
        retireve_data,
        path,
        query_name,
        support_name,
        retireve_method=f"_{retrieval_method}",
    )


if __name__ == "__main__":
    main()
