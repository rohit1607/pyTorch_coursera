import numpy as np
import gym
from buffer import ReplayBuffer
from OU_action_noise import OUActionNoise
from DDPG_network import CriticNetwork, ActorNetwork
import torch as T 
import torch.nn.functional as F

def plot_learning_curve(scores,x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i]= np.mean(scores[max(0,i-100):i+1])
    plt.plot(x, running_avg)
    plt.title("running avg of prev 100 scores")
    plt.savefig(figure_file)

class Agent:
    def __init__(self,lr_actor, lr_critic, ip_dims, n_actions, ac_hl1_dims, ac_hl2_dims, cr_hl1_dims, cr_hl2_dims, tau, gamma=0.99,batch_size=64, max_size=1e6):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma =gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.memory = ReplayBuffer(max_size, ip_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(lr_actor, ip_dims, ac_hl1_dims, ac_hl2_dims, n_actions, 'actor')
        self.critic = CriticNetwork(lr_critic, ip_dims, cr_hl1_dims, cr_hl2_dims, n_actions, 'critic')
        self.target_actor = ActorNetwork(lr_actor, ip_dims, ac_hl1_dims, ac_hl2_dims, n_actions, 'target_actor')
        self.target_critic = CriticNetwork(lr_critic, ip_dims, cr_hl1_dims, cr_hl2_dims, n_actions, 'target_critic')
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        self.actor.eval() # becuase we are using LayerNorm
        state = T.Tensor([obs])
        # For GPU
        # state = torch.Tensor([obs]).to(self.policy.device)
        mu = self.actor.forward(state)
        mu_prime =  mu + T.tensor(self.noise()) #for exploration
        self.actor.train() #bring actor to train mode
        return mu_prime.detach().numpy()[0]
        
    def remember(self, state, action ,reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    
    def learn(self):
        # let the memory fill up until batch_size so we can start sampling aproptiately
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer((self.batch_size))
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        states_ = T.tensor(states_, dtype=T.float)
        dones = T.tensor(dones)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)
    
        critic_values_[dones]=0.0 #using dones as a mask

        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size,1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        sefl.critic.optimizer.step()

        self.actor.optimizer().zero_grad()
        actor_loss = self.critic.forward(state, self.actor.forward((states)))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        print("check named_params: ", actor_params)
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        print("check dict: ", actor_state_dict, type(actor_state_dict))
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        # What does clone do?
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)