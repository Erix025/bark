import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from omegaconf import OmegaConf
import sys
from models import get_model
from train import train


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    args = parser.parse_args(sys.argv[1:])
    # Load config
    config = OmegaConf.load(args.config)
    data_cfg = config.Data
    train_cfg = config.Train

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_model(model_name=train_cfg.model, num_classes=120, device=device, pretrained=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    dataset = datasets.ImageFolder(root=data_cfg.dataset, transform=transform)
    # 计算训练集和验证集的大小
    train_size = int(data_cfg.train_size_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    # 随机划分数据集
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=train_cfg.epochs,
        device=device,
        checkpoint_path=train_cfg.checkpoint_dir
    )