import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm
from torch import nn

def resize_image(image : Image.Image, height, width):
    rate = min(height / image.size[1], width / image.size[0])
    new_size = (int(image.size[0] * rate), int(image.size[1] * rate))
    # pad the image to the target size
    image = image.resize(new_size)
    new_image = Image.new("RGB", (width, height))
    new_image.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return new_image

class MyDataSet(Dataset):
    def __init__(self, data_path, label_path, num_classes=None):
        # load label map from csv file, skip the first line
        self.label_map = {}
        with open(label_path, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split(',')
                self.label_map[line[0]] = line[1]
                
        # get all unique labels and map them to index
        label_index_map = {}
        for label in self.label_map.values():
            if label not in label_index_map:
                label_index_map[label] = len(label_index_map)

        self.label_map = {k: label_index_map[v] for k, v in self.label_map.items()}
        
        # load data
        data = []
        label = []

        num_classes = len(os.listdir(data_path)) if num_classes is None else num_classes
        # Iterate over all files in the data_path directory
        for file_name in tqdm(os.listdir(data_path)[:num_classes]):
            # Check if the file is a jpg file
            if file_name.endswith(".jpg"):
                # Open the image file
                image = Image.open(os.path.join(data_path, file_name))
                image = resize_image(image, 400, 400)
                # Convert the image to a tensor
                tensor = torch.Tensor(image.getdata()).view(image.size[1], image.size[0], 3)
                # Append the tensor to the data list
                data.append(tensor)
                # Append a label for this image
                label.append(self.label_map[file_name[:-4]])

        # Convert the data and label lists to tensors
        data = torch.stack(data)
        label = torch.tensor(label)

        # Assign the data and label tensors to the corresponding attributes of the dataset
        self.data = data
        self.label = label
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
class MyDataLoader(DataLoader):
    def __init__(self, data_path, label_path, batch_size, shuffle, num_classes=None):
        self.dataset = MyDataSet(data_path, label_path, num_classes)
        super(MyDataLoader, self).__init__(self.dataset, batch_size, shuffle)
        
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(400*400*3, 1200)
        self.linear2 = nn.Linear(1200, 120)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
def train(epochs, batch_size, lr, data_path, label_path, shuffle, num_classes=None):
    train_loader = MyDataLoader(data_path, label_path, batch_size, shuffle, num_classes)
    model = MyModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                
if __name__ == '__main__':
    train(epochs=10, batch_size=32, lr=0.001, data_path='data/train/train', label_path='data/labels.csv', shuffle=True, num_classes=10)