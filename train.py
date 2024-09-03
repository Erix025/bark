import os
from datetime import datetime
import torch

def train(
    model, 
    optimizer, 
    criterion, 
    train_loader, 
    valid_loader, 
    epochs, 
    device,
    checkpoint_path='checkpoints',
    validate_only=False):
    # create checkpoints folder
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs('log', exist_ok=True)
    log_file = f'log/{datetime.now()}.log'
    try:
        for epoch in range(epochs):
            if not validate_only:
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
                        with open(log_file, 'a') as f:
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
                with open(log_file, 'a') as f:
                    f.write(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')
                # 保存最新的模型检查点
                if not validate_only:
                    checkpoint_filename = f'{checkpoint_path}/model_epoch_{epoch + 1}.ckpt'
                    torch.save(model.state_dict(), checkpoint_filename)
                    print(f'Saved checkpoint: {checkpoint_filename}')
            if validate_only:
                break

    except KeyboardInterrupt:
        print('Training interrupted, saving checkpoint...')
        torch.save(model.state_dict(), 'checkpoints/model_interrupt.ckpt')

    # 保存最终模型
    if not validate_only:
        torch.save(model.state_dict(), 'checkpoints/model_final.ckpt')
        print('Finished Training')