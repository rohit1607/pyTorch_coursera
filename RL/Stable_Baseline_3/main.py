import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import os
from os.path import join
import matplotlib.pyplot as plt
import yaml
from utils import get_levelsets_dummy, customPlot,make_dir, write_params_to_log, build_input_param_dict, read_cfg_file
import sys
import csv


def plot_traj(traj, plot_fname, env, savefig=False):
    traj = np.array(traj)
    print(traj.shape, traj)

    plt.plot(traj[:, 0], traj[:, 1])
    plt.scatter(traj[:, 0], traj[:, 1], color="r")
    plt.xlim(0, env.xlim)
    plt.ylim(0, env.ylim)
    if savefig:
        plt.savefig(plot_fname, bbox_inches="tight")


# def read_cfg_file(cfg_name, print_dict=False):
#     with open(cfg_name, "r") as file:
#         try:
#             cfg = yaml.safe_load(file)
#         except yaml.YAMLError as exc:
#             print(exc)
#     # if print_dict:
#     #     for item in cfg.keys():
#     #         print()
#     #         for subitem in item.keys():
#     #             print(f"{subitem}: \t {item[subitem]}")
#     return cfg


def test_dummy_level_sets(env):
    level_sets = get_levelsets_dummy(env, tang_spac=1, radial_spac=1, reverse=True)
    plots = customPlot(env)
    plots.setup_fig()
    plots.plot_contours(level_sets, fname="level_set_test", save_fig=True)
    lens = [len(contour) for contour in level_sets]
    print(f"contour_lens = {lens}")
    return level_sets


# cfg_fname = "cfg/contGrid_v2_PPO.yaml"
# env = gym.make("gym_basic:contGrid-v2")
env_name = "gym_basic:contGrid-v3"
cfg_fname = "cfg/contGrid_v3_PPO.yaml"
current_filename = os.path.basename(__file__)

mkdir= make_dir(dir="experiments", common_subdir="exp", make_subdirs=["figs","logs"])
exp_num, exp_dir = mkdir()
exp_local_log_name = join(exp_dir,"local_param_log.csv")
input_param_dict = build_input_param_dict(exp_num, current_filename, env_name, cfg_fname, print_test=True)
write_params_to_log(input_param_dict, logfile=exp_local_log_name, header_row=True, append=True)
# sys.exit("Stopping")

env = gym.make("gym_basic:contGrid-v3")
cfg = read_cfg_file(cfg_fname, print_dict=True)
env.setup(cfg)


obs = env.reset()
# model_fname = "contGrid_v2_PPO"
model_fname = "contGrid_v3_PPO"
rl_params = cfg["RL_params"]
model_path = join(exp_dir, model_fname)

start_loc = np.array(cfg["grid_params"]['start_pos'], dtype=np.float32)
target_pos = np.array(cfg["grid_params"]['target_pos'], dtype=np.float32)
# level_sets = get_levelsets_dummy(env, start_loc, target_pos, reverse=True)
# contour_plot_fname = join(join(exp_dir, "figs"), "level_sets")
cplot =customPlot(env)
# cplot.plot_contours(level_sets, fname="level_sets", save_fig=True)

if rl_params["learn"] == True:
    eval_env = gym.make(env_name)
    cfg = read_cfg_file(cfg_fname, print_dict=True)
    eval_env.setup(cfg)
    log_path = join(exp_dir,"logs")
    eval_env = Monitor(eval_env, log_path)

 
    class CustomCallback(BaseCallback):
        def __init__(self, check_freq, eval_env, model, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.eval_env = eval_env
            self.model = model
            self.mean_reward_list = []


        def _on_step(self) -> None:
            if self.n_calls % self.check_freq == 0:

                mean_reward, std_reward = evaluate_policy(
                        self.model, self.eval_env, n_eval_episodes=10)
                self.mean_reward_list.append(mean_reward)
                print(f"cb: mean_reward = {mean_reward}\n")
            
        
        # def _on_rollout_end(self) -> None:
        #     """
        #     This event is triggered before updating the policy.
        #     """
        #     mean_reward, std_reward = evaluate_policy(
        #         self.model, self.eval_env, n_eval_episodes=100)
        #     self.mean_reward_list.append(mean_reward)
        #     print(f"cb: mean_reward = {mean_reward}\n")

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v2_A2C/")
    callback = CustomCallback(check_freq=100, eval_env=eval_env, model=model)
    model.learn(total_timesteps=rl_params["total_timesteps"])
    model.save(model_path)
    # # callback
    # print(f"len(callback.mean_reward_list) = {len(callback.mean_reward_list)}")
    # reward_plot_name = join(join(exp_dir,"figs"),"reward_plot")
    # cplot.plot_series(callback.mean_reward_list, label="mean_reward",
    #                  fname=reward_plot_name, save_fig=True)


if rl_params["predict"] == True:
    model = PPO.load(model_fname, env=env)
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

    traj_plot_name = join(join(exp_dir,"figs"), "traj_plot")
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
        plot_traj(traj, traj_plot_name, env, savefig=True)

    ValPlot_path = join(exp_dir,"figs")
    ValPlot_fname = model_fname+"Vplot"
    ValPlot_fname = join(ValPlot_path,ValPlot_fname)
    cplot.plot_V(model.policy, vmin=-10, vmax=60, fname=ValPlot_fname, save_fig=True)
env.close()

exp_summary_log = join("experiments","exp_summary.csv")
write_params_to_log(param_dict=input_param_dict, logfile=exp_summary_log, 
                        header_row=False, append=True)