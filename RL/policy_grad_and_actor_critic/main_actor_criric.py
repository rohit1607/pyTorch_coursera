import gym
import numpy as np
from actor_crit_network import Agent
from main_lunar_lander import plot_learning_curve
import time
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma = 0.9, lr = 5e-6, ip_dims=8, n_actions=4, hl1_dims=2048, hl2_dims=1536)

    n_games = 3000
    fname = 'acotr_critic_ lunar_lr_' + str(lr) + 'n_games_' + str(n_games)+ '.png'
    figure_file = join('figs',fname)
    start = time.time()

    for i in range(n_games):
        done = False
        score=0
        obs = env.reset()
        scores = []
        while not done:
            action = Agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, reward, obs_, done)
            obs = obs_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if i%10==0:
            print('episode ',i , 'score=', score, 'avg_score=', avg_score)
    end =  time.time()
    print("*** time for ", n_games, "iters= ", end-start, "s")
    x = [i for i in range(n_games)]
    plot_learning_curve(scores, x, figure_file)