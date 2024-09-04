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
from evaluate import evaluate, InferLoader
import os
from datetime import datetime

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help="Path to the config file", required=False, default="train.yaml")
    args = parser.parse_args(sys.argv[1:])
    # Load config
    config = OmegaConf.load(args.config)
    data_cfg = config.Data
    train_cfg = config.Train
    test_cfg = config.Test
    # backup config
    backup_dir = os.path.join(train_cfg.checkpoint_dir, str(datetime.now()))
    os.makedirs(backup_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(backup_dir, "config.yaml"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_model(model_name=train_cfg.model, num_classes=120, device=device, pretrained=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    train_path = os.path.join(data_cfg.train_dataset)
    infer_path = os.path.join(data_cfg.infer_dataset)

    if test_cfg.inference_only:
        dataset = InferLoader(root=infer_path, transform=transform)
    else:
        dataset = datasets.ImageFolder(root=train_path, transform=transform)

    # 计算训练集和验证集的大小
    train_size = int(data_cfg.train_size_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    # 随机划分数据集
    if test_cfg.inference_only:
        valid_loader = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=False)
    else:
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if test_cfg.inference_only or test_cfg.validate_only:
        print("Using checkpoint: ", test_cfg.checkpoint_path)
        model.load_state_dict(torch.load(test_cfg.checkpoint_path, weights_only=True))

    if test_cfg.inference_only:
        evaluate(model, valid_loader, device=device)
    else:
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=train_cfg.epochs,
            device=device,
            checkpoint_path=train_cfg.checkpoint_dir,
            validate_only=test_cfg.validate_only,
            backup_dir=backup_dir
        )