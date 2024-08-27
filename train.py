import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm

class MyDataSet(Dataset):
    def __init__(self, data_path, label_path):
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
        # mapping label to index
        self.label_map = {k: label_index_map[v] for k, v in self.label_map.items()}
        print(self.label_map)
        
        # load data
        data = []
        label = []

        # Iterate over all files in the data_path directory
        for file_name in tqdm(os.listdir(data_path)[:10]):
            # Check if the file is a jpg file
            if file_name.endswith(".jpg"):
                # Open the image file
                image = Image.open(os.path.join(data_path, file_name))
                # Convert the image to a tensor
                tensor = torch.Tensor(image.getdata()).view(image.size[1], image.size[0], 3)
                # Append the tensor to the data list
                data.append(tensor)
                # Append a label for this image (you can modify this according to your needs)
                label.append(self.label_map[file_name[:-4]])

        # Convert the data and label lists to tensors
        data = torch.stack(data)
        label = torch.tensor(label)

        # Assign the data and label tensors to the corresponding attributes of the dataset
        self.data = data
        self.label = label
        print('Data shape: {}, Label shape: {}'.format(data.shape, label.shape))
        print(data[0], label[0])
        exit()
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
class MyDataLoader(DataLoader):
    def __init__(self, data_path, label_path, batch_size, shuffle):
        self.dataset = MyDataSet(data_path, label_path)
        super(MyDataLoader, self).__init__(self.dataset, batch_size, shuffle)
        
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # TODO: define forward
        return x
    
def train(epochs, batch_size, lr, data_path, label_path, shuffle):
    train_loader = MyDataLoader(data_path, label_path, batch_size, shuffle)
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
    train(epochs=10, batch_size=32, lr=0.001, data_path='data/train/train', label_path='data/labels.csv', shuffle=True)