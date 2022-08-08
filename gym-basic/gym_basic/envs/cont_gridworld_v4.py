import gym
import numpy as np
import math

VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30

# Contnuous states. n Discrete Actions; multiple possible starting points.
class ContGridWorld_v4(gym.Env):
    
    def __init__(self, state_dim=2, action_dim=1, n_actions=8, grid_dim=[10.,10.],start_pos=[[5.0,5.0]], target_pos=[8.0,8.0], target_rad=1, F=1):
        super(ContGridWorld_v4, self).__init__()

        return 

    def setup(self, cfg, start_pos):
        """
        To be called expicitly to retreive params
        cfg: dict laoded from a yaml file
        start_pos: list of lists(starting positions); e.g start_pos=[[5.0,2.0]] 
        """
  
        self.cfg = cfg
        self.params = self.cfg["grid_params"]
        self.state_dim = int(self.params["state_dim"])
        self.action_dim = int(self.params["action_dim"])
        self.n_actions = int(self.params["n_actions"])
        self.xlim, self.ylim = self.params["grid_dims"]

        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=self.xlim, shape=(self.state_dim,)) #square domain

        # print(self.observation_space, self.action_space)
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.target_pos = np.array(self.params["target_pos"])
        self.target_rad = self.params["target_rad"]
        print(f"start_pos.shape={self.start_pos.shape}")
        self.F = self.params["F"]
        # state = self.reset()
        # self.state = np.array(self.start_pos, dtype=np.float32).reshape(2,).copy()
        self.state = self.reset()

        self.target_state = np.array(self.target_pos, dtype=np.float32).reshape(2,).copy()
        print("init: ",self.start_pos, type(self.start_pos))
        print("xlim: ", self.xlim)


    def transition(self, action, add_noise=False):
        action *= (2*np.pi/self.n_actions)
        self.state[0] += self.F*math.cos(action)
        self.state[1] += self.F*math.sin(action)
        # add noise
        # self.state += 0.05*np.random.randint(-3,4)


    def is_outbound(self, check_state = [float('inf'), float('inf')]):
        status = False
        # if no argument, check status for self.state
        if check_state[0] == np.float('inf') and check_state[1] == np.float('inf'):
            check_state = self.state
        lims = [self.xlim, self.ylim]
        for i in range(self.state_dim):
            if check_state[i] >= lims[i] or check_state[i]<0:
                status = True
                break
        return status

    def has_reached_target(self):
        status = False
        if np.linalg.norm(self.state - self.target_state) <= self.target_rad:
            status = True
        return status

    def step(self, action):
        old_s = self.state  #to restore position in case of outbound
        self.transition(action)
        self.reward = -1
        has_reached_target = self.has_reached_target()
        is_outbound = self.is_outbound()
        if has_reached_target:
            self.reward = self.xlim
            self.done = True
        elif is_outbound:
            self.reward = -self.xlim
            self.done = True
            self.state =  old_s
        info = {"is_outbound": is_outbound, "has_reached_target":has_reached_target} # what do i put here
        return self.state, self.reward, self.done, info

    def reset(self, reset_state=[float('inf'), float('inf')]):
        # self.state = [x,y]
        # print("in_reset:", self.start_pos)
        reset_state = np.array(reset_state, dtype=np.float32)
        if reset_state[0] == np.float('inf') and reset_state[1] == np.float('inf'):
            idx = np.random.randint(0, len(self.start_pos))
            reset_state = self.start_pos[idx,:].copy()
        self.state = reset_state
        self.done = False
        self.reward = 0
        # self.target_state = reset_state
        return self.state

    def render(self):
        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        #     self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)
        
        pass

    def seed(self, seed=None):
        pass