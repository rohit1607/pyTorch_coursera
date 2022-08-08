from turtle import forward
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np
import math
from os.path import join
import warnings

class PPOMemory:
    def __init__(self, batch_size):
        self.states=[]
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) #[0, 3, 6, 9]
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
        np.array(self.actions),\
        np.array(self.probs),\
        np.array(self.vals),\
        np.array(self.rewards),\
        np.array(self.dones),\
        batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):   
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = join(chkpt_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent:

    def __init__(self,
        env,
        learning_rate = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip= 0.2,
        clip_range_vf = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl = None,
        tensorboard_log = None,
        create_eval_env: bool = False,
        policy_kwargs = None,
        verbose: int = 0,
        seed = None,
        device = "auto",
        _init_setup_model: bool = True
    ):

        assert(batch_size > 1), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
        #TODO: self.num_envs significance
        # self.num_envs = 1
        # self.buffer_size = n_steps * self.num_envs 
        # assert(
        #         self.buffer_size > 1
        #     ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
        # untruncated_batches = self.buffer_size // batch_size
        # if self.buffer_size % batch_size > 0:
        #     warnings.warn(
        #         f"You have specified a mini-batch size of {batch_size},"
        #         f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
        #         f" after every {untruncated_batches} untruncated mini-batches,"
        #         f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
        #         f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
        #         f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
        #     )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.gamma = gamma
        self.n_steps =n_steps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate

        self.input_dims = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.actor = ActorNetwork(self.n_actions, self.input_dims, self.learning_rate)
        self.critic = CriticNetwork(self.input_dims, self.learning_rate)
        self.memory = PPOMemory(batch_size)


    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)    
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
  
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()  
  
        return action, probs, value

    def learn(self):
        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()            
  
        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        # compute advantages
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t = discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t

        advantage = T.tensor(advantage).to(self.actor.device)
        values = T.tensor(values).to(self.actor.device)

        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
            old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
            actions = T.tensor(action_arr[batch]).to(self.actor.device)            

            dist = self.actor(states)
            critic_value = self.critic(states)

            critic_value = T.squeeze(critic_value)

            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()
            #prob_ratio = (new_probs - old_probs).exp()
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                    1+self.policy_clip)*advantage[batch]
            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

            returns = advantage[batch] + values[batch]
            
            self.actor.optimizer.zero_grad()
            actor_loss.mean().backward()
            # T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            self.actor.optimizer.step()
            # print(f"actor_loss.grad={actor_loss.grad}")
            print("----- actor layer 1 -----")
            print(self.actor.actor[0].weight.grad)
            print("----- actor layer 2 -----")
            print(self.actor.actor[2].weight.grad)


            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # total_loss = actor_loss + self.vf_coef*critic_loss
            # self.actor.optimizer.zero_grad()
            # self.critic.optimizer.zero_grad()
            # total_loss.backward()
            # self.actor.optimizer.step()
            # self.critic.optimizer.zero_grad()

        self.memory.clear_memory() 







    
        
        
        
