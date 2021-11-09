import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)

# model =  nn.Sequential(nn.Linear(2,1),nn.Sigmoid())

class Data(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
    
    # Getter
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    
    # Get length
    def __len__(self):
        return self.len

data =  Data()

# custom modules

class logistic(nn.Module):
    def __init__(self, input_size, output_size):
        super(logistic, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

logistic_reg =  logistic(1,1)
optimizer = optim.SGD(logistic_reg.parameters(), lr=0.001)
trainloader =  DataLoader(dataset=data, batch_size = 4)
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()

def train(model, loss_fn, lr, epochs, plot_loss =True, figname = ""):
    Loss_list = []
    optimizer = optim.SGD(logistic_reg.parameters(), lr=lr)
    for epoch in range(epochs):
        for x,y in trainloader:
            yhat = model(x)
            loss = loss_fn(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_list.append(loss.item())
    print(Loss_list)

    if plot_loss:
        plt.plot(Loss_list)
        figname = "loss_" + figname
        plt.savefig(figname)
        plt.clf()
        plt.close()

lr = 2
epochs = 100   
train(logistic_reg, criterion1, lr, epochs, plot_loss=True, figname="1")
train(logistic_reg, criterion2, lr, epochs, plot_loss=True, figname="2")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logistic_reg')
writer.add_graph(logistic_reg)