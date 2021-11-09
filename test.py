import gym
env = gym.make("gym_basic:contGrid-v0")
obs_ = env.reset()

print(obs_)

# from stable_baselines3.common.env_checker import check_env
# check_env(env)
sum_r = 0
done = False
while obs_[1]<8:
    action = 1
    obs_, reward, done, info = env.step(action)
    sum_r += reward
    print(obs_, sum_r, info)
    if done:
        break


while obs_[0]<9:
    action = 2
    obs_, reward, done, info = env.step(action)
    sum_r += reward
    print(obs_, sum_r, info)
    if done:
        break
# import numpy as np
# for i in range(10):
#     print(np.random.randint(0,3))