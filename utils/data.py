from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
from tqdm import tqdm
from utils.image_utils import resize_image

class TrainDataset(Dataset):
    def __init__(self, data_path, label_path, num_samples=None):
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

        num_samples = len(os.listdir(data_path)) if num_samples is None else num_samples
        # Iterate over all files in the data_path directory
        for file_name in tqdm(os.listdir(data_path)[:num_samples]):
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

class TestDataset(Dataset):
    def __init__(self, data_path, num_samples=None):
        data = []
        id = []
        num_samples = len(os.listdir(data_path)) if num_samples is None else num_samples
        for file_name in tqdm(os.listdir(data_path)[:num_samples]):
            if file_name.endswith(".jpg"):
                image = Image.open(os.path.join(data_path, file_name))
                image = resize_image(image, 400, 400)
                tensor = torch.Tensor(image.getdata()).view(image.size[1], image.size[0], 3)
                data.append(tensor)
                id.append(file_name[:-4])
        self.data = torch.stack(data)
        self.id = id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.id[idx], self.data[idx]

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        super(MyDataLoader, self).__init__(self.dataset, batch_size, shuffle)