import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(1)

class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3,3,0.2).view(-1,1)
        self.f = self.x - 1
        self.y = self.f + 0.1*torch.randn(self.x.size())
        self.len = self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i,0], self.y[i,0]

    def __len__(self):
        return self.len

data = Data()
plt.scatter(data.x, data.y)
plt.savefig('test.png')

trainloader = DataLoader(dataset = data, batch_size = 1)
print(trainloader)

for x,y in trainloader:
    print("yo: ",x.data,y.data)