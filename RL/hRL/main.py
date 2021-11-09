import gym
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from actor_crit_network import Agent

def get_levelsets_dummy(env, vel_field_data=None, tang_spac=1, radial_spac=1):
    start_pos = env.start_pos
    target_pos = env.target_pos
    level_sets = []
    if (target_pos[0]- start_pos[0]) != 0:
        slope = (target_pos[1] - start_pos[1])/(target_pos[0]- start_pos[0])
    dist = np.linalg.norm(target_pos-start_pos)
    print("check dist, slope:", dist, slope)
    n_contours = int(dist/radial_spac) + 2
    print("check n_contours:", n_contours)
    for i in range(n_contours-1,-1,-1):
        rad = (i+1)*radial_spac
        del_theta = radial_spac/rad
        n_theta = int(0.67*np.pi/del_theta) + 1
        contour = []
        for j in range(n_theta):
            theta = j*del_theta
            p = np.array(start_pos) + rad*np.array([np.cos(theta), np.sin(theta)])
            # print("check p:", p)
            if not env.is_outbound(check_state=p):
                contour.append(p)
            # else: 
                # print(" is outbound \n\n")
        level_sets.append(contour)
    return level_sets

class customPlot():
    def __init__(self, env):
        self.env = env
        self.setup_fig()

    def setup_fig(self):
        fig = plt.figure(figsize=(10,10))
        plt.scatter(env.start_pos[0],env.start_pos[1],marker='o')
        plt.scatter(env.target_pos[0],env.target_pos[1],marker='*')
        plt.xlim(0,env.xlim)
        plt.ylim(0,env.ylim)
        plt.gca().set_aspect('equal', adjustable='box')


    def plot_contours(self, level_sets, fname = '', save_fig=False):
        for contour in level_sets:
            contour =  np.array(contour).reshape(len(contour),2)
            plt.plot(contour[:,0], contour[:,1])
        if save_fig:
            plt.savefig(fname, dpi=300)

    def plot_traj(self, traj, fname = '', save_fig=False):
        traj = np.array(traj).reshape(len(traj),2)
        plt.plot(traj[:,0], traj[:,1])
        if save_fig:
            plt.savefig(fname, dpi=300)
    
    def clf(self):
        plt.cla()

    def plot_V(self, agent, fname = '', save_fig=False):
        xs = np.linspace(0, self.env.xlim, 100)
        ys = np.linspace(0, self.env.ylim, 100)
        X,Y = np.meshgrid(xs,ys)
        V = np.empty_like(X)
        print("V.shape:", V.shape)
        for i in range(len(xs)):
            for j in range(len(ys)):
                x = X[i,j]
                y = Y[i,j]
                state = np.array([x,y], dtype=np.float32).reshape(2,)
                state = T.tensor([state])
                _, V[i][j] = agent.actor_critic.forward(state)
        plt.contourf(X,Y,V,zorder=-100)
        plt.colorbar()
        if save_fig:
            plt.savefig(fname, dpi=300)

env = gym.make("gym_basic:contGrid-v0", state_dim=2, action_dim=5, 
                grid_dim=[10.,10.],start_pos=[5.0,2.0], target_pos=[8.0,8.0], target_rad=1)
obs = env.reset()
# print("check obs:", (obs[0], obs[1]), obs[0])

level_sets =  get_levelsets_dummy(env)
pl = customPlot(env)
pl.plot_contours(level_sets, fname='contours.png',save_fig=True)

agent = Agent(gamma = 0.99, lr = 5e-6, ip_dims=2, n_actions=5, hl1_dims=512, hl2_dims=512)

def method1_forward_aproach(env, agent, n_iters):
    obs = env.reset()
    scores = []
    for i in range(n_iters):
        done = False
        score=0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, reward, obs_, done)
            obs = obs_
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if i%100==0:
            print('episode ',i , 'score=', score, 'avg_score=', avg_score)
    fname = 'model' +  str(n_iters)
    agent.save_model(fname=fname)

def method2_a(env,level_sets, iters_per_contour=100):
    # iters_per_contour = 200
    for contour in level_sets:
        for i in range(iters_per_contour):
            done = False
            score=0
            start_pt = contour[i%len(contour)]
            obs = env.reset(reset_state=start_pt)
            scores = []
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                score += reward
                agent.learn(obs, reward, obs_, done)
                obs = obs_
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            if i%100==0:
                print('episode ',i , 'score=', score, 'avg_score=', avg_score)
    fname = 'model_method_2_' +  str(iters_per_contour)
    agent.save_model(fname=fname)

def rollout(env, agent, load_file=None):
    obs = env.reset()
    if load_file != None:
        agent.load_model(load_file)
    done = False
    score=0
    traj = []
    traj.append((obs[0], obs[1]))
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        score += reward
        # print("s:", obs, "\t a:", action, "\t s':", obs_, "\t done:", done, "\t info: ", info)
        obs = obs_
        traj.append((obs[0], obs[1]))
        print("s: ", obs)
    return traj

# method1_forward_aproach(env, agent, 10000)
# traj=rollout(env, agent)
# print("traj", traj)
# pl.plot_traj(traj,fname='traj.png', save_fig=True)

method2_a(env,level_sets,iters_per_contour=400)
traj=rollout(env, agent)
print("traj", traj)
pl.plot_traj(traj,fname='m2a_traj.png', save_fig=True)

# print( "Now running loaded model")
traj2 = rollout(env, agent, load_file='model_method_2_400')
print("traj2", traj2)
pl.plot_traj(traj2,fname='m2a_traj2.png', save_fig=True)

pl.clf()
pl.setup_fig()
pl.plot_V(agent, fname='m2a_value_fn.png', save_fig=True)


