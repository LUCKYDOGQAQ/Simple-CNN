import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

import os

# 导入model
from models.model import Net, CNN
# 导入config
import configs.config as config
# 导入logger
from utils.logger import logger
# 导入data
from datasets.data_loader import train_loader, val_loader, test_loader
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

class Cifar10Task:
    def __init__(self, args, train_loader, val_loader, test_loader, model, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.model = model
        self.device = device

    def train(self):
        # 模型转成训练模式
        self.model.train()
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

        # 训练配置
        num_epochs = 5
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

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

                # 每100个iteration打印一次训练信息
                if i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Iteration : {i}')

            # 保存训练loss和验证loss
            train_loss = round(running_loss / len(train_loader), 3)
            val_loss = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 保存训练accuracy和验证accuracy
            train_accuracy = self.compute_accuracy(self.train_loader)
            val_accuracy = self.compute_accuracy(self.val_loader)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # 打印每个epoch训练信息
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Loss: {train_loss}, Accuracy: {train_accuracy} ,  '
                  f'Validate Loss: {val_loss}, Validate Accuracy: {val_accuracy}')

        # 完成训练后
        # 绘制损失曲线
        plt.figure(figsize=(8, 4))  # 创建一个宽度为8英寸，高度为4英寸的新图形
        plt.plot(range(num_epochs), train_losses, label='Training Loss')
        plt.plot(range(num_epochs), val_losses, label='Validating Loss')

        # # 添加坐标标签
        # for i, (xi, yi) in enumerate(zip(range(num_epochs), train_losses)):
        #     plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')
        # for i, (xi, yi) in enumerate(zip(range(num_epochs), val_losses)):
        #     plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 绘制准确度曲线
        plt.figure(figsize=(8, 4))  # 创建一个宽度为8英寸，高度为4英寸的新图形
        plt.plot(range(num_epochs), train_accuracies, label='Training Accuracy')
        plt.plot(range(num_epochs), val_accuracies, label='Validating Accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
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

        # # 访问模型的权重
        # for name, param in self.model.named_parameters():
        #     print(f"Parameter name: {name}")
        #     print(f"Weights: {param.data}")

        self.model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # 统计正确预测的数量
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Accuracy on the test set: {:.2f}%'.format(accuracy))

    def validate(self):
        self.model.eval()  # 设置模型为评估模式
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        running_loss = 0

        # 在验证集上进行验证
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data

                # 向前传播
                outputs = self.model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)

                # 记录损失
                running_loss += loss.item()

        # 计算并返回平均验证损失
        val_loss = round(running_loss / len(self.val_loader), 3)
        self.model.train()  # 设置模型为训练模式
        return val_loss

    # 定义函数来计算准确率
    def compute_accuracy(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = round((correct / total), 3)
        self.model.train()
        return accuracy


if __name__ == '__main__':
    # 定义任务所需的args，train_loader, test_loader, model, device
    args = config.Args().get_parser()

    model = Net()

    device = torch.device("cuda")

    # 创建任务
    task = Cifar10Task(args, train_loader, val_loader, test_loader, model, device)
    # 训练模型
    task.train()
    # 测试模型
    model_name = 'haha.pth'
    model_path = os.path.join(args.output_dir, model_name)
    # print(model_path)
    task.test(model_path)
