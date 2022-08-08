import gym
import numpy as np

VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 30

# Contnuous states. 4 Discrete Actions
class ContGridWorld(gym.Env):
    
    def __init__(self, state_dim=2, action_dim=5, grid_dim=[10.,10.],start_pos=[5.0,2.0], target_pos=[8.0,8.0], target_rad=1):
        super(ContGridWorld, self).__init__()
        self.xlim, self.ylim = grid_dim
        self.action_space = gym.spaces.Discrete(action_dim)
        self.observation_space = gym.spaces.Box(low=0, high=self.xlim, shape=(state_dim,)) #square domain

        # print(self.observation_space, self.action_space)
        self.start_pos = np.array(start_pos, dtype=np.float32).reshape(2,)
        self.target_pos = np.array(target_pos)
        self.target_rad = target_rad
        self.state_dim = state_dim
        self.action_dim = action_dim
        # state = self.reset()
        self.state = np.array(start_pos, dtype=np.float32).reshape(2,).copy()
        self.target_state = np.array(target_pos, dtype=np.float32).reshape(2,).copy()
        print("init: ",self.start_pos, type(self.start_pos))
        print("xlim: ", self.xlim)

        return 

    def transition(self, action):
        if action == 0:
            pass
        if action == 1:
            self.state[1] += 1
        if action == 2:
            self.state[0] += 1
        if action == 3:
            self.state[1] -= 1
        if action == 4:
            self.state[0] -= 1
        # add noise
        self.state += 0.05*np.random.randint(-3,4)


    def is_outbound(self, check_state = [float('inf'), float('inf')]):
        status = False
        # if no argument, check statuf for self.state
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
            self.reward = 10
            self.done = True
        elif is_outbound:
            self.reward = -50
            self.done = True
            self.state =  old_s
        info = {"is_outbound": is_outbound, "has_reached_target":has_reached_target} # what do i put here
        return self.state, self.reward, self.done, info

    def reset(self, reset_state=[float('inf'), float('inf')]):
        # self.state = [x,y]
        # print("in_reset:", self.start_pos)
        reset_state = np.array(reset_state, dtype=np.float32)
        if reset_state[0] == np.float('inf') and reset_state[1] == np.float('inf'):
            reset_state = self.start_pos.copy()
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