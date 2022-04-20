import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from os.path import join
import matplotlib.pyplot as plt
import yaml
from utils import get_levelsets_dummy, customPlot


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
cfg_fname = "cfg/contGrid_v3_PPO.yaml"
env = gym.make("gym_basic:contGrid-v3")
cfg = read_cfg_file(cfg_fname, print_dict=True)
env.setup(cfg)


obs = env.reset()
# model_fname = "contGrid_v2_PPO"
model_fname = "contGrid_v3_PPO"

rl_params = cfg["RL_params"]

start_loc = np.array(cfg["grid_params"]['start_pos'], dtype=np.float32)
target_pos = np.array(cfg["grid_params"]['target_pos'], dtype=np.float32)
level_sets = get_levelsets_dummy(env, start_loc, target_pos, reverse=True)
cplot =customPlot(env)
cplot.plot_contours(level_sets, fname="level_sets", save_fig=True)

if rl_params["learn"] == True:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v3_PPO/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/contGrid_v2_A2C/")
    model.learn(total_timesteps=rl_params["total_timesteps"])
    model.save(model_fname)


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
    cplot.plot_V(model.policy, vmin=-10, vmax=60, fname=ValPlot_fname, save_fig=True)
env.close()
