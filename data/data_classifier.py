import numpy as np
import pandas as pd
import os

csv = pd.read_csv('labels.csv')
labels = np.array(csv['breed'].values.tolist())
labels = np.unique(labels)
#rank by alphabet
labels.sort()
labels2num = dict(zip(labels, range(len(labels))))
print(labels)

for i in range(len(csv)):
    line = csv.iloc[i]
    breed = labels2num[line['breed']]
    frompath = os.path.join('train/train',line['id']+'.jpg')
    topath = os.path.join('ImageFolder','class'+str(breed),line['id']+'.jpg')
    if not os.path.exists(os.path.join('ImageFolder','class'+str(breed))):
        os.makedirs(os.path.join('ImageFolder','class'+str(breed)))
    os.system('cp '+frompath+' '+topath)


# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # 定义数据预处理和增强
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图像大小
#     transforms.ToTensor(),          # 将图像转换为Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
# ])

# # 使用 ImageFolder 加载数据集
# dataset = datasets.ImageFolder(root='ImageFolder', transform=transform)

# # 使用 DataLoader 来创建一个数据加载器
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
