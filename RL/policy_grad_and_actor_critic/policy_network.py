import numpy as np 
import torch
from torch import nn, optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, lr, ip_dim, h_dim, n_actions):
        super(PolicyNetwork,self).__init__()
        self.linear1 = nn.Linear(ip_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # for GPU
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyGradAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []

        h_dim = 128
        self.policy =  PolicyNetwork(self.lr, input_dims, h_dim, n_actions)

    def choose_action(self, obs):
        # state = torch.Tensor([obs])
        # For GPU
        state = torch.Tensor([obs]).to(self.policy.device)

        probs = F.softmax(self.policy.forward(state))
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        # print("rew_mem shape= ", self.reward_memory.shape)
        G = np.zeros_like(self.reward_memory, dtype=np.float32)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.gamma
            G[t]=G_sum
        # G = torch.tensor(G, dtype=torch.float)
        # For GPU
        G = torch.tensor(G, dtype=torch.float32).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g*logprob
        loss.backward()
        self.policy.optimizer.step()
        self.action_memory = []
        self.reward_memory = []

