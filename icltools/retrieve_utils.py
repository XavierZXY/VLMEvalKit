import base64
import csv
import json
import os
import pickle
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
    with open(datasets_path + datasets_name + ".tsv", "r", encoding="utf-8") as tsvfile:
        # 创建一个csv.reader对象，指定制表符为字段分隔符
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        headers = next(tsvreader)

        # 遍历文件中的每一行
        for row in tsvreader:
            query[row[headers.index("index")]] = row[headers.index("question")]
            images[row[headers.index("index")]] = row[headers.index("image")]
    return query, images


def retireve_icl_data(query, images, icl_query, icl_image, shots: int = 10):
    # Define the file paths for storing features
    query_features_file = "query_features.pkl"
    icl_query_features_file = "icl_query_features.pkl"

    # Check if the query features file exists
    if os.path.exists(query_features_file):
        # Load the query features from the file
        with open(query_features_file, "rb") as f:
            query_features = pickle.load(f)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        # Extract query features and save to file
        query_features = dict()
        for key, q in tqdm(query.items()):
            text = clip.tokenize([q]).to(device)
            with torch.no_grad():
                query_features[key] = model.encode_text(text).cpu().numpy()

        # Save the query features to the file
        with open(query_features_file, "wb") as f:
            pickle.dump(query_features, f)

    # Check if the icl_query features file exists
    if os.path.exists(icl_query_features_file):
        # Load the icl_query features from the file
        with open(icl_query_features_file, "rb") as f:
            icl_query_features = pickle.load(f)
    else:
        # Extract icl_query features and save to file
        icl_query_features = dict()
        for key, q in tqdm(icl_query.items()):
            text = clip.tokenize([q]).to(device)
            with torch.no_grad():
                icl_query_features[key] = model.encode_text(text).cpu().numpy()

        # Save the icl_query features to the file
        with open(icl_query_features_file, "wb") as f:
            pickle.dump(icl_query_features, f)

    # Compute cosine similarity for each query individually
    top_queries = {}
    for q_key, q_vector in query_features.items():
        q_vector = q_vector.squeeze()
        icl_vectors = np.array(list(icl_query_features.values())).squeeze()

        similarities = np.dot(icl_vectors, q_vector.T).squeeze()
        similarities /= np.linalg.norm(icl_vectors, axis=1) * np.linalg.norm(q_vector)

        # Get the indices of the top 'shots' similar queries
        top_indices = np.argsort(similarities)[-shots:][::-1]

        # Retrieve the top similar queries
        top_queries[q_key] = [list(icl_query.keys())[i] for i in top_indices]

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


def generate_tsv_file(retireve_data, path, query_name, support_name):
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

    output_tsv = path + query_name + "_retrieved.tsv"
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
    # path = "./datasets/clevr/"
    # path = "./datasets/open_mi/"
    path = "/home/zxy/codes/working/ICLBoom/VLMEvalKit/icltools/datasets/clevr/"
    # datasets_name = "Open_MI.tsv"
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
    retireve_data = retireve_icl_data(query, images, icl_query, icl_images, shots=10)
    # read tsv file
    # read_tsv_file(retireve_data)
    # generate new tsv data file
    generate_tsv_file(retireve_data, path, query_name, support_name)


if __name__ == "__main__":
    main()
