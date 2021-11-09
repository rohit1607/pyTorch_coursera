import numpy as np
import torch as T 
from torch import nn, optim
import torch.nn.functional as F 
import os

class CriticNetwork(nn.Module):
    def __init__(self, lr, ip_dims, hl1_dims, hl2_dims, n_actions, name, chkpt_dir = 'tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.ip_dims = ip_dims
        self.hl1_dims = hl1_dims
        self.hl2_dims = hl2_dims
        self.n_actions =  n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_ddpg')
        print("***n_actions=",n_actions)
        print("ip_dims=", ip_dims)
        print("hl1_dims=",hl1_dims)
        print("hl2_dims=",hl2_dims)

        self.linear1 = nn.Linear(ip_dims,hl1_dims)
        self.linear2 = nn.Linear(hl1_dims,hl2_dims)
        # Batch normaliztion (Layer NOrmalization)
        self.bn1 = nn.LayerNorm(self.hl1_dims)
        self.bn2 = nn.LayerNorm(self.hl2_dims)

        self.ac_val = nn.Linear(self.n_actions, self.hl2_dims)
        self.q = nn.Linear(hl2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        # print("Check1:", self.linear1.weight.data, '\n', self.linear1.weight.data.size())
        f1  = 1./np.sqrt(self.linear1.weight.data.size()[0])
        self.linear1.weight.data.uniform_(-f1,f1)
        self.linear1.bias.data.uniform_(-f1,f1)

        f2  = 1./np.sqrt(self.linear2.weight.data.size()[0])
        self.linear2.weight.data.uniform_(-f2,f2)
        self.linear2.bias.data.uniform_(-f2,f2)    

        f3 = 0.003
        self.q.weight.data.uniform_(-f3,f3)
        self.q.bias.data.uniform_(-f3,f3)    

        f4  = 1./np.sqrt(self.ac_val.weight.data.size()[0])
        self.ac_val.weight.data.uniform_(-f3,f3)
        self.ac_val.bias.data.uniform_(-f3,f3)           
        
    def forward(self, state, action):
        state_val = F.relu(self.bn1(self.linear1(state)))
        state_val = self.bn2(self.linear2(state_val))
        action_val = self.ac_val(action)
        state_action_val = F.relu(T.add(state_val, action_val))
        state_action_val = self.q(state_action_val)
        return state_action_val

    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))



class ActorNetwork(nn.Module):
    def __init__(self, lr, ip_dims, hl1_dims, hl2_dims, n_actions, name, chkpt_dir = 'tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.ip_dims = ip_dims
        self.hl1_dims = hl1_dims
        self.hl2_dims = hl2_dims
        self.n_actions =  n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name + '_ddpg')

        self.linear1 = nn.Linear(ip_dims,hl1_dims)
        self.linear2 = nn.Linear(hl1_dims,hl2_dims)
        # Batch normaliztion (Layer NOrmalization)
        self.bn1 = nn.LayerNorm(self.hl1_dims)
        self.bn2 = nn.LayerNorm(self.hl2_dims)
        self.linear3 = nn.Linear(hl2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # print("Check2:", self.linear1.weight.data, '\n', self.linear1.weight.data.size())
        f1  = 1./np.sqrt(self.linear1.weight.data.size()[0])
        self.linear1.weight.data.uniform_(-f1,f1)
        self.linear1.bias.data.uniform_(-f1,f1)

        f2  = 1./np.sqrt(self.linear2.weight.data.size()[0])
        self.linear2.weight.data.uniform_(-f2,f2)
        self.linear2.bias.data.uniform_(-f2,f2)    

        f3 = 0.003
        self.linear3.weight.data.uniform_(-f3,f3)
        self.linear3.bias.data.uniform_(-f3,f3)    
         
        
    def forward(self, state):
        x = F.relu(self.bn1(self.linear1(state)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = T.tanh(self.linear3(x))
        # x = T.tanh(self.linear3(x))
        return x

    def save_checkpoint(self):
        print("...saving checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))


