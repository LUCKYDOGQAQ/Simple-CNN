from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# 定义数据
# 数据预处理，1.转为tensor，2.归一化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集
train_set = CIFAR10(root='./data',
                    train=True,
                    download=True,
                    transform=transform)
train_loader = DataLoader(train_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)
# 验证集
test_set = CIFAR10(root='./data',
                   train=False,
                   download=True,
                   transform=transform)
test_loader = DataLoader(test_set,
                         batch_size=4,
                         shuffle=False,
                         num_workers=2)
