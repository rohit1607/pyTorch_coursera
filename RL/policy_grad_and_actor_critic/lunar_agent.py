import gym

if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    print(env.action_space)
    print(env.observation_space)
    
    n_episodes = 10
    for i in range(n_episodes):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            print("obs=",obs_)
            print("r=",reward)
            print("d=",done)
            print("info",info)
            print()
            score+= reward
        print("episode", i, "\tscore=", score )