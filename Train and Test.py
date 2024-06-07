import torch.nn as nn
import torch
import torch.nn.functional as F
import re
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
file_path = r'C:\Users\Administrator\Desktop\PhiUSIIL_Phishing_URL_Dataset.csv'


#导入提取出来的特征
def get_data_from_csv(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data_dict = {col: data[col].tolist() for col in data.columns}
    return data_dict
    #key分别为FILENAME,URL,URLLength,Domain,DomainLength,IsDomainIP,TLD,URLSimilarityIndex,CharContinuationRate,TLDLegitimateProb,URLCharProb,TLDLength,NoOfSubDomain,HasObfuscation,NoOfObfuscatedChar,ObfuscationRatio,NoOfLettersInURL,LetterRatioInURL,NoOfDegitsInURL,DegitRatioInURL,NoOfEqualsInURL,NoOfQMarkInURL,NoOfAmpersandInURL,NoOfOtherSpecialCharsInURL,SpacialCharRatioInURL,IsHTTPS,LineOfCode,LargestLineLength,HasTitle,Title,DomainTitleMatchScore,URLTitleMatchScore,HasFavicon,Robots,IsResponsive,NoOfURLRedirect,NoOfSelfRedirect,HasDescription,NoOfPopup,NoOfiFrame,HasExternalFormSubmit,HasSocialNet,HasSubmitButton,HasHiddenFields,HasPasswordField,Bank,Pay,Crypto,HasCopyrightInfo,NoOfImage,NoOfCSS,NoOfJS,NoOfSelfRef,NoOfEmptyRef,NoOfExternalRef,label


#html特征的提取与向量
class HTMLToEmbedding(nn.Module):
    def __init__(self, max_tokens=100, embedding_dim=100):
        super(HTMLToEmbedding, self).__init__()
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim

        # 定义文本向量化层
        self.tokenizer = re.compile(r"<[^>]+>|[\w]+")

        # 初始化词汇表
        self.vocab = {}
        self.vocab_size = 0

        # 定义嵌入层
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

    def build_vocab(self, texts):
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self.vocab_size = len(self.vocab)

        # 重新初始化嵌入层，确保嵌入层大小与词汇表大小匹配
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def preprocess_text(self, text):
        # 使用正则表达式去除 HTML 标签，并分词
        tokens = self.tokenizer.findall(text)
        return tokens[:self.max_tokens]

    def forward(self, html_features):
        # 预处理 HTML 特征
        tokens = self.preprocess_text(html_features)

        # 如果词汇表为空，建立词汇表
        if not self.vocab:
            self.build_vocab([html_features])

        # 将文本转换为对应的索引序列
        indexed_tokens = [self.vocab.get(token, 0) for token in tokens]

        # 进行长度填充
        indexed_tokens += [0] * (self.max_tokens - len(indexed_tokens))
        indexed_tokens = indexed_tokens[:self.max_tokens]

        # 转换为 PyTorch Tensor
        indexed_tokens = torch.tensor(indexed_tokens)

        # 将索引序列转换为对应的嵌入向量
        embedded_html = self.embedding(indexed_tokens.long())

        return embedded_html


# url特征的提取
class TextToEmbedding(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=60, max_tokens=60):
        super(TextToEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab = {}

    def preprocess_text(self, text):
        tokens = text.split()
        return tokens

    def build_vocab(self, texts):
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab) + 1

    def forward(self, text):
        tokens = self.preprocess_text(text)
        if not self.vocab:
            self.build_vocab([text])

        indexed_tokens = [self.vocab.get(token, 0) for token in tokens]
        indexed_tokens = indexed_tokens[:self.max_tokens] + [0] * (self.max_tokens - len(indexed_tokens))
        indexed_tokens = torch.tensor(indexed_tokens).long()
        embedded_text = self.embedding(indexed_tokens)
        return embedded_text

# 串联层
class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, input1, input2):
        # 拼接两个输入
        concatenated = torch.cat((input1, input2), dim=1)
        return concatenated


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, padding=4)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层1
        self.fc1 = nn.Linear(160, 256)  # 假设展平后的输入大小为1600
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层2
        self.fc2 = nn.Linear(256, 1)


    def forward(self, x):
        # 卷积层 -> ReLU激活函数 -> 池化层
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))

        # 展平操作，将特征张量展平为 [batch_size, features]
        # 注意：features 应该是池化层输出后的特征数量
        batch_size, _, features = x.size()
        x = x.view(batch_size, -1)

        # 全连接层1 -> ReLU激活函数 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # 全连接层2 -> Sigmoid激活函数
        # 这里使用 Sigmoid 假设是一个二分类问题
        x = torch.sigmoid(self.fc2(x))

        # 压缩输出到一个标量值
        x = x.squeeze()

        return x
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(50, 128)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=0.5)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 第一个全连接层，使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # Dropout 层
        x = self.dropout(x)
        # 第二个全连接层
        x = self.fc2(x)
        return x


def normalize(lists):
    new = []
    ma = max(lists)
    mi = min(lists)
    for val in lists:
        new.append((val - mi) / (ma - mi))
    return new

# num_data = []
data_all = get_data_from_csv(file_path)
# for key in data_all:
#     if key in ['URL', 'FILENAME', 'Domain', 'Title', 'TLD', 'label']:
#         continue
#     tmp_list = data_all[key]
#     num_data.append(tmp_list)
# float_list = [[float(x) for x in row] for row in num_data]
# float_list = [normalize(x) for x in float_list]
# X_train = torch.tensor(float_list).T                                   #形状为235795*50
# Y_train = torch.tensor([float(x) for x in data_all['label']])          #形状为235795

# X_train = X_train.unsqueeze(0)  # 在第一维度添加一个维度
# X_train = X_train.unsqueeze(0)  # 在第二维度添加一个维度
# 创建一个 TensorDataset

# 将数据转换为 TensorDataset
# train_dataset = TensorDataset(X_train, Y_train)
# # 创建 DataLoader
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




# 创建模型实例
model_cov = ConvNet()
model_simple = LinearModel()


# 假设 model_cov 已经被定义为一个接受嵌入向量作为输入的模型
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 初始化优化器
optimizer_cov = optim.Adam(model_cov.parameters(), lr=0.01)  # Adam优化器

# 初始化 TextToEmbedding 实例
vocab_size = 20000000  # 示例词汇表大小
embedding_dim = 10  # 示例嵌入维度大小
max_tokens = 50  # 示例最大 token 数量
text_to_embedding = TextToEmbedding(vocab_size, embedding_dim, max_tokens)

texts = data_all['URL']
text_to_embedding.build_vocab(texts)


num_epochs = 2
for epoch in range(num_epochs):
    cnt = 0
    running_loss = 0.0
    for inputs, labels in zip(data_all['URL'][:1000], data_all['label'][:1000]):
        labels = torch.tensor(labels)
        # 将梯度参数置零
        optimizer_cov.zero_grad()

        # 使用 TextToEmbedding 将文本转换为嵌入向量
        mat = text_to_embedding.forward(inputs)   #max_token * embedding_dim
        # 前向传播
        outputs = model_cov(mat)

        if labels.dtype == torch.long:
            labels.to(dtype=torch.float32)

        if outputs.dtype == torch.long:
            outputs.to(dtype=torch.float32)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer_cov.step()

        running_loss += loss.item()

        cnt += 1
        if cnt % 10 == 0:        #10个3秒
            print(f'训练进度：{(cnt / 10):.2f}%')

    # 打印每个epoch的平均损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_all['label'])}")

print("Finished Training")

