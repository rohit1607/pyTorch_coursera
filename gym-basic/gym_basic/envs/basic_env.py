import gym

class BasicEnv(gym.Env):
    
    def __init__(self, n_states=2, n_actions=5):
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Discrete(n_states)
        print(self.observation_space, self.action_space)

    def step(self, action):
        if action == 0:
            self.state = 1
            self.reward = 1
        else:
            self.state = 0
            self.reward = -1

        if self.state == 1:
            self.done = True

        info = {} # what do i put here
        return self.state, self.reward, self.done, info

    def reset(self):
        self.state = 0
        self.done = False
        return self.state