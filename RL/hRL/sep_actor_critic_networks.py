import numpy as np
import torch as T
from torch import nn, optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, lr, ip_dims, n_actions, hl_1_dims = 256, hl_2_dims = 256):
        super(ActorNetwork, self).__init__()
        self.h1 = nn.Linear(ip_dims, hl_1_dims)
        self.h2 = nn.Linear(hl_1_dims, hl_2_dims)
        self.pi = nn.Linear(hl_2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print("self.params: ", self.parameters())
        # GPU Commands
        # self.device = T.device(0)
        # self.to(self.device)

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        return self.pi(x)


class CriticNetwork(nn.Module):
    def __init__(self, lr, ip_dims, hl_1_dims = 256, hl_2_dims = 256):
        super(CriticNetwork, self).__init__()
        self.h1 = nn.Linear(ip_dims, hl_1_dims)
        self.h2 = nn.Linear(hl_1_dims, hl_2_dims)
        self.v = nn.Linear(hl_2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        print("self.params: ", self.parameters())
        # GPU Commands
        # self.device = T.device(0)
        # self.to(self.device)

    def forward(self, state):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        return self.v(x)

class sepAgent():
    def __init__(self, lr, ip_dims, hl1_dims, hl2_dims, n_actions, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.hl1_dims = hl1_dims
        self.hl2_dims = hl2_dims
        self.actor = ActorNetwork(lr, ip_dims, n_actions, hl_1_dims=hl1_dims, hl_2_dims=hl2_dims)
        self.critic = CriticNetwork(lr, ip_dims, hl_1_dims=hl1_dims, hl_2_dims=hl2_dims)
        self.log_probs = None

    def choose_action(self, obs):
        state = T.tensor([obs])
        # GPU
        # state = T.tensor([obs]).to(self.actor_critic.device)
        state = T.tensor([obs])

        probs = self.actor.forward(state) #not probs yet
        probs = F.softmax(probs, dim=1)
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        state = T.tensor([state])
        state_ = T.tensor([state_])
        reward = T.tensor(reward)
        vs = self.critic.forward(state)
        vs_ = self.critic.forward(state_)
        delta = reward + self.gamma*vs_*(1-int(done)) - vs
       
        actor_loss = -self.log_prob*delta.item()
        critic_loss = delta**2
        actor_loss.backward()
        critic_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return actor_loss.item(), critic_loss.item()

    def save_model(self, fname='model'):
        critic_fname = "sep_critic_" + fname
        actor_fname = "sep_actor_" + fname
        T.save(self.actor, actor_fname)
        T.save(self.critic, critic_fname)

    def load_model(self, fname):
        critic_fname = "sep_critic_" + fname
        actor_fname = "sep_actor_" + fname
        self.actor = T.load(actor_fname)
        self.critic = T.load(critic_fname)
        print("--- Models loaded successfully ----")

