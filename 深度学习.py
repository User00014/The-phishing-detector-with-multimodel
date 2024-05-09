import torch.nn as nn
import torch
import torch.nn.functional as F
import re
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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


# 示例用法
html_features = "<html><body>This is an example</body></html>"
model = HTMLToEmbedding()
embedded_vector = model(html_features)


# url特征的提取
class TextToEmbedding(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=100, max_tokens=100):
        super(TextToEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens

        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 定义词汇表
        self.vocab = None

    def preprocess_text(self, text):
        # 在这里执行文本预处理操作，例如分词、去除标点等
        tokens = text.split()  # 这里简单地按空格进行分词

        return tokens

    def build_vocab(self, texts):
        # 构建词汇表
        vocab = {}
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab) + 1  # 0 保留给未登录词
        self.vocab = vocab

    def forward(self, text):
        # 预处理文本特征
        tokens = self.preprocess_text(text)

        # 如果词汇表为空，建立词汇表
        if not self.vocab:
            self.build_vocab([text])

        # 将文本转换为对应的索引序列
        indexed_tokens = [self.vocab.get(token, 0) for token in tokens]

        # 进行长度填充
        indexed_tokens += [0] * (self.max_tokens - len(indexed_tokens))
        indexed_tokens = indexed_tokens[:self.max_tokens]

        # 转换为 PyTorch Tensor
        indexed_tokens = torch.tensor(indexed_tokens)

        # 将索引序列转换为对应的嵌入向量
        embedded_text = self.embedding(indexed_tokens.long())

        return embedded_text

# 使用该类进行测试
model = TextToEmbedding()
text = "This is a test sentence."
embedded_vector1 = model(text)

#串联层
class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, input1, input2):
        # 拼接两个输入
        concatenated = torch.cat((input1, input2), dim=1)
        return concatenated

model = ConcatenateLayer()
new_vector = model(embedded_vector, embedded_vector1)
print(new_vector.shape)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层1
        self.fc1 = nn.Linear(32 * 96, 10)  # 修改全连接层的输入大小
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层2
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # 卷积层 -> ReLU激活函数 -> 池化层
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))
        # 将特征张量展平
        x = x.view(-1, 32 * 96)  # 修改展平后的大小
        # 全连接层1 -> ReLU激活函数 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 全连接层2 -> Sigmoid激活函数
        x = torch.sigmoid(self.fc2(x))
        return x



X_train = torch.randn(1000, 200)  # 1000个样本，每个样本有200个特征
y_train = torch.randint(0, 2, (1000,))  # 1000个二分类标签，0或1

# 转换数据为张量数据集
train_dataset = TensorDataset(X_train, y_train)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型实例
model = ConvNet()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 将梯度参数置零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs.squeeze(), labels.float())
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        running_loss += loss.item()

    # 打印每个epoch的平均损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Finished Training")

