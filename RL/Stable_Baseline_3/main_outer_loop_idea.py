from cgitb import reset
from operator import mod
import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from os.path import join
import matplotlib.pyplot as plt
import yaml
from utils import customPlot, get_levelsets_dummy
import sys


def plot_traj(traj, model_fname, env, savefig=False):
    traj = np.array(traj)
    print(traj.shape, traj)

    plt.plot(traj[:, 0], traj[:, 1])
    plt.scatter(traj[:, 0], traj[:, 1], color="r")
    plt.xlim(0, env.xlim)
    plt.ylim(0, env.ylim)
    if savefig:
        plot_fname = join("figs", model_fname)
        plt.savefig(plot_fname, bbox_inches="tight")


def read_cfg_file(cfg_name, print_dict=False):
    with open(cfg_name, "r") as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # if print_dict:
    #     for item in cfg.keys():
    #         print()
    #         for subitem in item.keys():
    #             print(f"{subitem}: \t {item[subitem]}")
    return cfg


# cfg_fname = "cfg/contGrid_v2_PPO.yaml"
# env = gym.make("gym_basic:contGrid-v2")
cfg_fname = "cfg/contGrid_v4_PPO.yaml"
env = gym.make("gym_basic:contGrid-v4")
cfg = read_cfg_file(cfg_fname, print_dict=True)
start_loc = np.array(cfg["grid_params"]['start_pos'], dtype=np.float32)
target_pos = np.array(cfg["grid_params"]['target_pos'], dtype=np.float32)
env.setup(cfg,[start_loc])
level_sets = get_levelsets_dummy(env, start_loc, target_pos, reverse=True)
level_sets.reverse()
n_level_sets = len(level_sets)
print([len(level_set) for level_set in level_sets])

obs = env.reset()
# model_fname = "contGrid_v2_PPO"
model_fname = "contGrid_v4_PPO"

rl_params = cfg["RL_params"]

cplot =customPlot(env)
cplot.plot_contours(level_sets, fname="level_sets", save_fig=True)

# for i in range(n_level_sets):
#     print(f"****** levelset {i} ***********")
#     env.setup(cfg,level_sets[i])
#     for j in range(5):
#         reset_state = env.reset()
#         current_state = env.state
#         print(f"reset_state= {reset_state} \n \
#                 current_state = {current_state} \n\n")

# sys.exit("sys exit")

if rl_params["learn"] == True:

    for i in range(n_level_sets):
        print(f"****** levelset {i} ***********")
        env.setup(cfg,level_sets[i])
        if i == 0:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v4_PPO/")
        # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v2_A2C/")
        model.learn(total_timesteps=rl_params["total_timesteps"])
        model_path = join("saved_models", model_fname)
        model.save(model_path)


if rl_params["predict"] == True:
    env.setup(cfg,[start_loc])
    model_path = join("saved_models", model_fname)
    model = PPO.load(model_path, env=env)
    obs = env.reset()

    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print("modelstats:", mean_reward, std_reward)

    params = model.get_parameters()
    print(params.keys())

    startpos = env.start_pos
    print("startpos=", startpos)
    print("starting at ", obs)
    max_episode_len = int(rl_params["max_episode_len"])
    n_trials = int(rl_params["n_trials"])
    print(f"params=\n{params.keys()}")

    for j in range(n_trials):
        print(f"trial {j} \n\n")
        sum_r = 0
        traj = []
        traj.append(tuple(obs))
        for i in range(max_episode_len):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            sum_r += reward
            traj.append(tuple(obs))
            print(i, obs, reward, info)
            if done:
                obs = env.reset()
                break
            if i % 10 == 0:
                print("iter: ", i)
        print(traj)
        plot_traj(traj, model_fname, env, savefig=True)


    ValPlot_path = "figs"
    ValPlot_fname = model_fname+"Vplot"
    ValPlot_fname = join(ValPlot_path,ValPlot_fname)
    cplot.plot_V(model.policy, vmin=-10, vmax=50, fname=ValPlot_fname, save_fig=True)
env.close()
