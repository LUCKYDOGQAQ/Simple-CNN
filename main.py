import torch
import torch.nn as nn
import torch.optim as optim

import os

# 导入model
from models.model import Net
# 导入config
import configs.config as config
# 导入logger
from utils.logger import logger
# 导入data
import datasets.data_loader as data_loader
# 结果可视化
import matplotlib.pyplot as plt


################################################################
# 检查GPU是否可用
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     logger.info("GPU可用")
# else:
#     device = torch.device("cpu")
#     logger.info("GPU不可用，将使用CPU")
################################################################

class NNTask:
    def __init__(self, args, train_loader, test_loader, model, device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.model = model
        self.device = device

    def train(self):
        pass
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

        # 训练配置
        num_epochs = 2
        train_losses = []

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0
            for i, data in enumerate(self.train_loader, 0):
                # 获取inputs和labels
                inputs, labels = data

                # 将梯度置为0
                optimizer.zero_grad()

                # 向前转播
                outputs = self.model(inputs)

                # 求loss
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()

                # 更新权重
                optimizer.step()

                # 记录训练损失
                running_loss += loss.item()

                # 打印训练信息
                print(f'Epoch [{epoch + 1}/{num_epochs}], iteration [{i}], Loss: {loss.item()}')

            # 计算平均loss
            train_losses.append(round(running_loss / len(train_loader), 3))

        # # 绘制损失曲线
        plt.plot(range(num_epochs), train_losses, label='Training Loss')
        # 添加坐标标签
        for i, (xi, yi) in enumerate(zip(range(num_epochs), train_losses)):
            plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 保存训练好的模型
        model_name = 'haha.pth'
        save_path = os.path.join(self.args.output_dir, model_name)
        torch.save(model.state_dict(), save_path)

    def test(self, model_path):
        # 加载模型的状态字典
        model_state_dict = torch.load(model_path)

        # 将模型的状态字典加载到模型中
        self.model.load_state_dict(model_state_dict)

        # 访问模型的权重
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Weights: {param.data}")
        pass


if __name__ == '__main__':
    # 定义任务所需的args，train_loader, test_loader, model, device
    args = config.Args().get_parser()
    print(args)

    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader

    model = Net()
    device = torch.device("cuda")

    # 创建任务
    task = NNTask(args, train_loader, test_loader, model, device)
    # 训练模型
    # task.train()
    # 测试模型
    model_name = 'haha.pth'
    model_path = os.path.join(args.output_dir, model_name)
    # print(model_path)
    task.test(model_path)
