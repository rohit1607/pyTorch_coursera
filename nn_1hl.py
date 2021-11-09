import torch
from torch import nn, optim
import matplotlib.pyplot as plt

def PlotStuff(X, Y, model, epoch, leg=True):
    
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg == True:
        plt.legend()
    else:
        pass


class Net(nn.Module):
    def __init__(self, ip_dim, h_dim, op_dim):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(ip_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, op_dim)

    def forward(self, x):
        return self.linear2(torch.sigmoid(self.linear1(x)))

ip_dim = 1
h_dim = 2
op_dim = 1

model = Net(ip_dim, h_dim, op_dim)

X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0
print("X, Yshapes= ", X.size(), Y.size())
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# def criterion_cross(outputs, labels):
#     out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
#     return out


def train(epochs):
    cost_list = []
    for epoch in range(epochs):
        cost=0
        for x,y in zip(X,Y):
            # print("x,y = ",x,y)
            yhat = model(x)
            # print("yhat = ", yhat)
            # print("shapes= ", yhat.size(), y.size())
            # loss = criterion_cross(yhat,y)
            loss = criterion(yhat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost += loss.item()
        cost_list.append(cost)

        if epoch % 300 == 0:    
            PlotStuff(X, Y, model, epoch, leg=True)
            # plt.show()
            # model(X)
            # plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('activations')
            fname = "nn" + str(epoch)
            plt.savefig(fname)
            plt.clf()
    return cost_list
epochs = 1000
cost_list = train(epochs)
plt.plot(cost_list)
plt.savefig("nn_costlist.png")



        