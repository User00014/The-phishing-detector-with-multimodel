import torch
from transformers import BertTokenizer, BertModel
from bs4 import BeautifulSoup
import csv
from urllib.parse import unquote
import os
import os.path as osp
import numpy as np

# 指定本地模型和分词器的路径（确保这是一个目录路径）
local_model_dir = r"C:\Users\13488\Desktop\大创用\数据\bert-base-uncased\bert-base-uncased"  # 使用原始字符串

# 从本地加载分词器
tokenizer = BertTokenizer.from_pretrained(local_model_dir)

# 从本地加载模型
model = BertModel.from_pretrained(local_model_dir)


def extract_features_from_URL(url_content):
    # Parse HTML content and extract text

    # 解码URL
    decoded_url = unquote(url_content)

    # Tokenize the text
    tokens = tokenizer.encode(decoded_url, add_special_tokens=True)
    tokens = tokens[:tokenizer.model_max_length]  # Ensure the input is within the model's max length

    # Convert tokens to PyTorch tensor
    input_ids = torch.tensor(tokens).unsqueeze(0)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the last hidden states
    last_hidden_states = outputs.last_hidden_state

    # 使用平均池化
    mean_pooling = last_hidden_states.mean(dim=1)  # (1, 768)

    return mean_pooling


# Example url content
filename = r"C:\Users\13488\Desktop\大创用\数据\URLfeatures.csv"
# 创建一个空列表来存储第一列内容
first_column = []
# 打开CSV文件进行读取
with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)

    # 跳过第一行
    next(csv_reader)

    # 逐行读取CSV文件的每一行
    for row in csv_reader:
        # 添加第一列内容到列表中
        first_column.append(row[0])


feature_arr = []
for url in first_column:
    features = extract_features_from_URL(url)
    # print('tmp feature:{}'.format(features.shape))
    feature_arr.append(features)

feature_arr = torch.cat(feature_arr)
print(feature_arr)
indices = [int(os.path.splitext(i)[0].split('_')[-1]) for i in range(1, len(feature_arr))]

# # 将特征向量和索引一起排序
# sorted_features = [f for _, f in sorted(zip(indices, feature_arr))]
#
# # 将排序后的特征向量堆叠成一个二维数组
# features_np = np.stack(sorted_features)
#
# # 保存排序后的特征数组和索引到 .npz 文件
# np.savez_compressed(
#     r'C:\Users\13488\Desktop\test\sorted_features.npz',
#     features=features_np,
#     indices=indices  # 这里添加索引的保存
# )
#
# print(f"特征向量和索引已保存到 'sorted_features.npz'，索引为: {indices}")

