import gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("gym_basic:contGrid-v0")
obs = env.reset()
# print("obs=", obs)
# # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log/")
# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./log/")
# model.learn(total_timesteps=60000)
# model.save("stoch_contGrid_A2C")
# del model


model = A2C.load("stoch_contGrid_A2C", env=env)
obs = env.reset()

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("modelstats:", mean_reward, std_reward)

params = model.get_parameters()
print(params.keys())

startpos = env.start_pos
print("startpos=", startpos)
print("starting at ", obs)
sum_r = 0
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    sum_r += reward
    print(obs, reward, info)
    if done:
        obs = env.reset()
        break
    if i%10 == 0:
        print("iter: ", i)

env.close()
