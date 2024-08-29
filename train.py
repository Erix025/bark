import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from argparse import ArgumentParser
from omegaconf import OmegaConf
import sys
from utils.data import MyDataLoader, TrainDataset, TestDataset
from utils.model import Linear_nn as Model
from models import ViTModel
import os
from datetime import datetime

def train(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    valid_loader, 
    epochs, 
    device,
    checkpoint_path='checkpoints'):
    # create checkpoints folder
    os.makedirs(checkpoint_path, exist_ok=True)
    
    try:
        for epoch in range(epochs):
            model.train()  # 设置模型为训练模式
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    with open('log.txt', 'a') as f:
                        f.write(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}\n')
            # 每隔几个 epoch 进行一次验证（例如每个 epoch 验证一次）
            if (epoch + 1) % 1 == 0:
                model.eval()  # 设置模型为评估模式
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():  # 关闭梯度计算
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss /= len(valid_loader)
                val_accuracy = correct / total

                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
                os.makedirs('log', exist_ok=True)
                with open(f'log/{datetime.now()}.log', 'a') as f:
                    f.write(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')
                # 保存最新的模型检查点
                checkpoint_filename = f'{checkpoint_path}/model_epoch_{epoch + 1}.ckpt'
                torch.save(model.state_dict(), checkpoint_filename)
                print(f'Saved checkpoint: {checkpoint_filename}')

    except KeyboardInterrupt:
        print('Training interrupted, saving checkpoint...')
        torch.save(model.state_dict(), 'checkpoints/model_interrupt.ckpt')

    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/model_final.ckpt')
                
def evaluate(model, dataloader):
    for i, data in enumerate(dataloader):
        id, data = data
        output = model(data)
        output = torch.softmax(output, dim=1)
        # TODO: Save the output to a file
        print(id, output)
        break

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    args = parser.parse_args(sys.argv[1:])
    config = OmegaConf.load(args.config)
    # Train
    data_cfg = config.Data
    opt_cfg = config.Optimization

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model = ViTModel(num_classes=120, device=device, pretrained=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    dataset = datasets.ImageFolder(root=data_cfg.dataset, transform=transform)
    # 计算训练集和验证集的大小
    train_size = int(0.9 * len(dataset))  # 80% 用于训练
    valid_size = len(dataset) - train_size   # 20% 用于验证
    # 随机划分数据集
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=opt_cfg.epochs,
        device=device
    )