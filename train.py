import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
import sys
from utils.data import MyDataLoader, TrainDataset, TestDataset
from utils.model import Linear_nn as Model

def train(model, dataloader, lr, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, (data, label) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Epoch: {}, Iter: {}, Loss: {}'.format(epoch, i, loss.item()))
                
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

    train_loader = MyDataLoader(TrainDataset(data_path=data_cfg.train_path, label_path=data_cfg.label_path, num_samples=data_cfg.train_samples), batch_size=opt_cfg.batch_size, shuffle=data_cfg.shuffle)
    test_loader = MyDataLoader(TestDataset(data_path=data_cfg.test_path, num_samples=data_cfg.test_samples), batch_size=opt_cfg.batch_size, shuffle=False)
    model = Model()
    
    train(model, train_loader, lr=opt_cfg.lr, epochs=opt_cfg.epochs)
    
    evaluate(model, test_loader)