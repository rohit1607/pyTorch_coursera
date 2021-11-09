import  gym
import numpy as np
from policy_network import PolicyGradAgent
import matplotlib.pyplot as plt 
from os.path import join
import time

def plot_learning_curve(scores,x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i]= np.mean(scores[max(0,i-100):i+1])
    plt.plot(x, running_avg)
    plt.title("running avg of prev 100 scores")
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    lr = 0.0005
    agent = PolicyGradAgent(lr= lr, input_dims=8)
    fname = 'lunar_lr_' + str(lr) + 'n_games_' + str(n_games)+ '.png'
    figure_file = join('figs',fname)


    scores=[]

    start = time.time()
    for i in range(n_games):
        done=False
        observation =  env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = obs_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        if i%10==0:
            print('episode ',i , 'score=', score, 'avg_score=', avg_score)
    end =  time.time()
    print("*** time for ", n_games, "iters= ", end-start, "s")
    x = [i for i in range(n_games)]
    plot_learning_curve(scores, x, figure_file)