from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

# 定义数据
# 数据预处理，1.数据增强 2.转为tensor 3.归一化
transform = transforms.Compose([
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),

    # 随机旋转（-10到+10度之间的随机旋转）
    transforms.RandomRotation(degrees=(-10, 10)),

    # 随机大小缩放和裁剪
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),

    # PIL图像转tensor
    transforms.ToTensor(),

    # 数据归一化
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 训练集
train_set = CIFAR10(root='./data',
                    train=True,
                    download=False,
                    transform=transform)
# 测试集
test_set = CIFAR10(root='./data',
                   train=False,
                   download=False,
                   transform=transform)

# 定义训练集和验证集的划分比例
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size

# 随机分割划分训练集和验证集
train_set, val_dataset = random_split(train_set, [train_size, val_size])

BATCH_SIZE = 32
NUM_WORKERS = 4
# 数据加载器
train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)
val_loader = DataLoader(train_set,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)
