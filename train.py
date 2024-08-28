import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
import sys
from utils.data import MyDataLoader
from utils.model import Linear_nn as Model

def train(epochs, batch_size, lr, data_path, label_path, shuffle, num_classes=None):
    train_loader = MyDataLoader(data_path, label_path, batch_size, shuffle, num_classes)
    model = Model()
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
    # Parse arguments
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args(sys.argv[1:])
    config = OmegaConf.load(args.config)
    # Train
    data_cfg = config.Data
    opt_cfg = config.Optimization

    train(opt_cfg.epochs, opt_cfg.batch_size, opt_cfg.lr, 
        data_cfg.data_path, data_cfg.label_path, data_cfg.shuffle, data_cfg.num_classes)