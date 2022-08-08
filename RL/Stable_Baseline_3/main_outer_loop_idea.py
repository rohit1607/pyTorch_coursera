import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from os.path import join
import matplotlib.pyplot as plt
import yaml
from utils import customPlot, get_optimal_path, get_levelsets_dummy, make_dir, write_params_to_log, build_input_param_dict, read_cfg_file
import sys
import csv
from cgitb import reset
from operator import mod


def plot_traj(traj, plot_fname, env, savefig=False):
    traj = np.array(traj)
    print(traj.shape, traj)

    plt.plot(traj[:, 0], traj[:, 1])
    plt.scatter(traj[:, 0], traj[:, 1], color="r")
    plt.xlim(0, env.xlim)
    plt.ylim(0, env.ylim)
    if savefig:
        plt.savefig(plot_fname, bbox_inches="tight")


# cfg_fname = "cfg/contGrid_v2_PPO.yaml"
# env = gym.make("gym_basic:contGrid-v2")
env_name = "gym_basic:contGrid-v5"
cfg_fname = "cfg/contGrid_v5_PPO.yaml"
current_filename = os.path.basename(__file__)


mkdir= make_dir(dir="experiments", common_subdir="exp", make_subdirs=["figs","logs"])
exp_num, exp_dir = mkdir()
exp_local_log_name = join(exp_dir,"local_param_log.csv")
input_param_dict = build_input_param_dict(exp_num, current_filename, env_name, cfg_fname, print_test=True)
write_params_to_log(input_param_dict, logfile=exp_local_log_name, header_row=True, append=True)
# sys.exit("Stopping")

env = gym.make(env_name)
cfg = read_cfg_file(cfg_fname, print_dict=True)
start_loc = np.array(cfg["grid_params"]['start_pos'], dtype=np.float32)
target_pos = np.array(cfg["grid_params"]['target_pos'], dtype=np.float32)
env.setup(cfg,[start_loc])
# env = Monitor(env, "log/" )
radial_spac = cfg["grid_params"]["radial_spac_level_sets"]

# Get levelset and paths
sample_freq = 2
vel = (env.U, env.V)
dims = (env.vel_shape[1],env.vel_shape[2])
# dims = (50,50)
# target_pos=[40,25]  #TODO: fix hard code

opt_path_,level_sets_ = get_optimal_path(list(start_loc), list(target_pos), vel, env.F, dims, save_fig=True, fname='figs/hjb_path1' )

# opt_path_ contains many points because of fine discrtezn of PDE soln. We dont need such fine desc in our RL soln
sample_ids = list(np.arange(1,len(opt_path_)-1, sample_freq))
sample_ids.append(len(opt_path_)-1)
n_samples = len(sample_ids)
opt_path = opt_path_[sample_ids]
# print(opt_path)
# sys.exit("stopping")
# level_sets = level_sets_[sample_ids]

# level_sets.reverse()
# n_level_sets = len(level_sets)
# print([len(level_set) for level_set in level_sets])

obs = env.reset()
model_fname = "contGrid_v5_PPO"
rl_params = cfg["RL_params"]

contour_plot_fname = join(join(exp_dir, "figs"), "level_sets")
cplot =customPlot(env)
# cplot.plot_contours(level_sets, fname=contour_plot_fname, save_fig=True)

ValPlot_path = join(exp_dir,"figs")
ValPlot_fname = model_fname+"Vplot"
ValPlot_fname = join(ValPlot_path,ValPlot_fname)
# for i in range(n_level_sets):
#     print(f"****** levelset {i} ***********")
#     env.setup(cfg,level_sets[i])
#     for j in range(5):
#         reset_state = env.reset()
#         current_state = env.state
#         print(f"reset_state= {reset_state} \n \
#                 current_state = {current_state} \n\n")

# sys.exit("sys exit")

model_path = join(exp_dir, model_fname)


if rl_params["learn"] == True:
    #setup evaluation callback
    eval_env = gym.make(env_name)
    cfg = read_cfg_file(cfg_fname, print_dict=True)
    eval_env.setup(cfg,[start_loc])
    log_path = join(exp_dir,"logs")
    eval_env = Monitor(eval_env, log_path)
    # eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
    #                             log_path='./log/', eval_freq=500,
    #                             deterministic=True, render=False )

    class CustomCallback(BaseCallback):
        def __init__(self, check_freq, eval_env, model, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.eval_env = eval_env
            self.model = model
            self.mean_reward_list = []

        def _on_step(self) -> None:
            pass

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=100)
            self.mean_reward_list.append(mean_reward)
            print(f"cb: mean_reward = {mean_reward}\n")

    # n_samples is effectively the no. of sub problems.
    overall_total_timesteps = rl_params["total_timesteps"]
    # TODO: may have to use smarter heuristic
    total_timesteps_per_level_set = overall_total_timesteps//n_samples

    # sub-problem loop
    for i in range(n_samples):
        # if len(level_sets[i])>0:
        print(f"****** path sample {i} ***********")
        env.setup(cfg,[opt_path[i]])
        # env = Monitor(env, "log/" )

        if i == 0:
            # log_path = join(exp_dir,"logs")
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
            # callback = CustomCallback(check_freq=100, eval_env=eval_env, model=model)

        # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v2_A2C/")
        # TODO: see doc for reset_num_timesteps
        model.learn(total_timesteps=total_timesteps_per_level_set, reset_num_timesteps=False)
        model.save(model_path)
        cplot.clf()
        ValPlot_fname_i = ValPlot_fname + str(i)
        cplot.plot_V(model.policy, vmin=-100, vmax=100, fname=ValPlot_fname_i, save_fig=True)



    # print(f"len(callback.mean_reward_list) = {len(callback.mean_reward_list)}")
    # reward_plot_name = join(join(exp_dir,"figs"),"reward_plot")
    # cplot.plot_series(callback.mean_reward_list, label="mean_reward",
    #                  fname=reward_plot_name, save_fig=True)
    
 

# def evaluate_policy_and_store_data(env, model, cfg, start_loc):
#     env.setup(cfg,[start_loc])
#     mean_reward, std_reward = evaluate_policy(
#     model, model.get_env(), n_eval_episodes=10)
#     return


if rl_params["predict"] == True:
    env.setup(cfg,[start_loc])
    model_path = join(exp_dir, model_fname)
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

    traj_plot_name = join(join(exp_dir,"figs"), "traj_plot")
    # episode loop / trial loop
    for j in range(n_trials):
        print(f"trial {j} \n\n")
        sum_r = 0
        traj = []
        traj.append(tuple(obs))
        #  steps within one episode
        for i in range(max_episode_len):
            action, _states = model.predict(obs, deterministic=True) #obs is cur state
            obs, reward, done, info = env.step(action) # this obs is the next state
            sum_r += reward
            traj.append(tuple(obs))
            print(i, obs, reward, info)
            if done:
                obs = env.reset()
                break
            if i % 10 == 0:
                print("iter: ", i)
        # print(traj)
        plot_traj(traj, traj_plot_name, env, savefig=True)


    cplot.plot_V(model.policy, vmin=-100, vmax=100, fname=ValPlot_fname, save_fig=True)

env.close()

exp_summary_log = join("experiments","exp_summary.csv")
write_params_to_log(param_dict=input_param_dict, logfile=exp_summary_log, 
                        header_row=False, append=True)
