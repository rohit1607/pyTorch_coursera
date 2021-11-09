import gym
import numpy as np
from agent import Agent
from os.path import join

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    # print("env.observation_space.shape=", env.observation_space.shape)
    print("###n_actions=", env.action_space.shape)
    agent = Agent( lr_actor=0.0005, lr_critic=0.001, ip_dims=env.observation_space.shape[0], n_actions=env.action_space.shape[0],
                    ac_hl1_dims=400, ac_hl2_dims=300, cr_hl1_dims=400, cr_hl2_dims=300, 
                    tau = 0.001, gamma=0.99, batch_size=64, max_size=1000000)
              

    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.lr_actor) + '_beta_' + \
                str(agent.lr_critic) +'_ngames_' + str(n_games) + '.png'
    figure_file = join('figs',filename)
    
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, obs_, done)
            agent.learn()
            score += reward
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score>best_score:
            best_score =avg_score
            agent.save_models()
        if i%10==0:
            print('episode ',i , 'score=', score, 'avg_score=', avg_score)

    x = [i for i in range(n_games)]
    plot_learning_curve(score_history, x, figure_file)
        