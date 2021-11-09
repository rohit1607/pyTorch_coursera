import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(1)

# Multiple Linear Regression


# Create a Linear Regression Custom Class

class Lin_Reg(nn.Module):
    def __init__(self, input_size, output_size):
        super(Lin_Reg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class Data2D(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b    
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):          
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

data = Data2D()
model = Lin_Reg(2,1)

print(list(model.parameters()))
print(model.state_dict())

# create an optimizer
opt = optim.SGD(model.parameters(), lr=0.0001)
criterion =  nn.MSELoss()
train_loader = DataLoader(dataset=data, batch_size=1)


Loss_list = []
def train_model(epochs):
    for epoch in range(epochs):
        for x,y in train_loader:
            yhat = model(x) #compute yhat
            loss = criterion(yhat,y)    #compute loss
            Loss_list.append(loss)
            opt.zero_grad()         #make gradiens zero
            loss.backward()         #compute derivatives
            opt.step()              #update parameters

train_model(1000)
print("params after training:  ",model.state_dict())
