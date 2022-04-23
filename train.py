import torch
from model.mot_net import MOTNet

from torch_geometric.loader import DataLoader

def train(model,optimizer,train_dataset, train_loader,device, crit):

    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

def start_training(train_dataset, batch_size, num_epochs):
    device = torch.device('cuda')
    model = MOTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = torch.nn.BCELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    for epoch in range(num_epochs):
        train(model,optimizer,train_dataset, train_loader,device, crit)

from sacred import Experiment
ex = Experiment('hello_config')

@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)