import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        x = self.sigmoid(x)  # 使用Sigmoid激活函数
        return x

################################################################
# 检查GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU可用")
else:
    device = torch.device("cpu")
    print("GPU不可用，将使用CPU")
################################################################

# 创建模型和损失函数
model = SimpleNN()

criterion = nn.BCELoss()  # 二进制交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

# 模拟输入数据和目标标签
input_data = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
target = torch.tensor([[1.0]], dtype=torch.float32)

# 前向传播
output = model(input_data)
loss = criterion(output, target)

# 反向传播
optimizer.zero_grad()  # 清零梯度
loss.backward()  # 反向传播计算梯度

# 更新模型参数
optimizer.step()

# 打印模型的权重和偏置
print("Model parameters after one step of optimization:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")
