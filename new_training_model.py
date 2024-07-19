import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义一维卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        # 最大池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层
        self.fc1 = nn.Linear(32 * 46, 10)  # 更新展平操作后的大小
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)
        # 输出层
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 46)  # 将特征展平成一维向量
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 创建模型实例
model = Net()

# 打印模型结构
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建虚拟训练数据
# 假设我们有10个样本，每个样本有1个特征，长度为100
num_samples = 10
input_data = torch.randn(num_samples, 1, 100)  # 生成正态分布的随机数据作为输入
targets = torch.randn(num_samples, 1)  # 生成与每个输入对应的目标值，这里假设是回归任务，目标值也是随机的

# 模型训练
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
test_input = torch.randn(1, 1, 100)  # 生成一个测试样例，长度为100的正态分布随机数列
print(test_input)
with torch.no_grad():
    output = model(test_input)
    print("预测结果:", output.item())

