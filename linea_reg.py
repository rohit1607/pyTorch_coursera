import torch
import matplotlib.pyplot as plt

X =  torch.linspace(-3,3,100)
# print(X)

Y = X + 0.2*torch.randn(X.size())
# print(Y)

plt.scatter(X,Y)
plt.savefig('test.png')

w = torch.tensor(-10.0, requires_grad=True)

def forward(w,X):
    return w*X
    

def compute_loss(Y, Yhat):
    return torch.mean((Yhat-Y)**2)


LOSS = []
iters = 100
lr = 0.01
for i in range(iters):
    Yhat = forward(w,X)
    loss = compute_loss(Y, Yhat)
    loss.backward() # accumalated the gradients
    w.data = w.data - lr*w.grad.data
    # print(w.grad.data)
    w.grad.data.zero_()
    # print(w.grad.data)
    LOSS.append(loss.data)
    # print("w=",w.data)

plt.plot(LOSS)
plt.savefig('LOSS.png')
print("solution: ", w.data)