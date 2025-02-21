import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 定义分类模型
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.net(x)

# 随机图像数据集
class RandomImageDataset(Dataset):
    def __init__(self, size=1024, img_size=224):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (
            torch.randn(3, self.img_size, self.img_size),  # 随机图像
            torch.randint(0, 10, (1,)).squeeze()           # 类别标签
        )



def example_runing_gpu():
        # 初始化双GPU环境
    assert torch.cuda.device_count() >= 2, "需要至少2块GPU"
    devices = [0, 1]
    
    # 创建并行模型
    model = nn.DataParallel(
        ClassificationModel(), 
        device_ids=devices
    ).cuda(devices[0])
    
    # 数据加载器
    dataset = RandomImageDataset(size=2048)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    
    # 训练配置
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(5):
        total_loss = 0.0
        for inputs, targets in loader:
            # 数据迁移到主GPU
            inputs = inputs.cuda(devices[0], non_blocking=True)
            targets = targets.cuda(devices[0], non_blocking=True)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")


def example_tensorflow_gpu():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)



if __name__ == "__main__":
    example_tensorflow_gpu()
