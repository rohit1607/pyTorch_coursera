from utils import make_dir,read_cfg_file
import os
import numpy as np


current_filename = os.path.basename(__file__)
print(current_filename)
# mkdir= make_dir(dir="experiments", common_subdir="exp")
# print(mkdir())
# print(i, new_dir)

""" --------make contours-------"""

from utils import get_optimal_path
start_pos=(10,25)
target_pos=[40,25]
u=1
F=1
dims=(50,50)

# U=np.zeros((1000,dims[0],dims[1]))
# V=np.zeros((1000,dims[0],dims[1]))
# U[:,20:30,:]=u
# V[:,:,:]=0

# np.save("env_fields/velocity/test_hw_g50/u.npy", U)
# np.save("env_fields/velocity/test_hw_g50/v.npy", V)

U = np.load("env_fields/velocity/test_hw_g50/u.npy")
V = np.load("env_fields/velocity/test_hw_g50/v.npy")
print(U[0,19:35,0])
vel = (U,V)
path, conts = get_optimal_path(start_pos, target_pos, vel, F, dims, save_fig=True, fname='figs/hjb_path' )
print(len(path))
print(path)
print(len(conts))
sample_ids = list(np.arange(1,len(path)-1, 2))
sample_ids.insert(0,0)
sample_ids.append(len(path)-1)
print(sample_ids)
print(path[sample_ids])
print(type(conts), len(conts), len(conts[5]))

# """----------test Contgrid_v5-----------------"""
# import gym
# env_name = "gym_basic:contGrid-v5"
# cfg_fname = "cfg/contGrid_v5_PPO.yaml"
# cfg = read_cfg_file(cfg_fname, print_dict=True)
# env = gym.make(env_name)

# start_loc = np.array(cfg["grid_params"]['start_pos'], dtype=np.float32)
# target_pos = np.array(cfg["grid_params"]['target_pos'], dtype=np.float32)
# env.setup(cfg,[start_loc])

