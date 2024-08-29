import torch        
def evaluate(model, dataloader):
    for i, data in enumerate(dataloader):
        id, data = data
        output = model(data)
        output = torch.softmax(output, dim=1)
        # TODO: Save the output to a file
        print(id, output)
        break