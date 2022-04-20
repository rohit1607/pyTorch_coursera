import gym
import numpy as np
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
import torch as T
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import sys


def get_levelsets_dummy(
    env, start_pos, target_pos, vel_field_data=None, tang_spac=1, radial_spac=1, reverse=False
):

    alpha = 0
    if reverse:
        temp = start_pos
        start_pos = target_pos
        target_pos = temp
        alpha = np.pi
    level_sets = []
    if (target_pos[0] - start_pos[0]) != 0:
        slope = (target_pos[1] - start_pos[1]) / (target_pos[0] - start_pos[0])
    dist = np.linalg.norm(target_pos - start_pos)
    print("check dist, slope:", dist, slope)
    n_contours = int(dist / radial_spac) + 2
    print("check n_contours:", n_contours)
    for i in range(n_contours - 1, -1, -1):
        rad = (i + 1) * radial_spac
        del_theta = radial_spac / rad
        n_theta = int(0.67 * np.pi / del_theta) + 1
        contour = []
        for j in range(n_theta):
            theta = j * del_theta
            p = np.array(start_pos) + rad * np.array(
                [np.cos(theta + alpha), np.sin(theta + alpha)]
            )
            # print("check p:", p)
            if not env.is_outbound(check_state=p):
                contour.append(p)
            # else:
            # print(" is outbound \n\n")
        level_sets.append(contour)
    return level_sets


class customPlot:
    def __init__(self, env):
        self.env = env
        self.setup_fig()

    def setup_fig(self):
        fig = plt.figure(figsize=(10, 10))
        try: #for contGridworld_v3
            plt.scatter(self.env.start_pos[0], self.env.start_pos[1], marker="o")
        except: #for contGridworld_v4
            for start_pos in self.env.start_pos:
                plt.scatter(start_pos[0], start_pos[1], marker="o")
        plt.scatter(self.env.target_pos[0], self.env.target_pos[1], marker="*")
        plt.xlim(0, self.env.xlim)
        plt.ylim(0, self.env.ylim)
        plt.gca().set_aspect("equal", adjustable="box")

    def plot_contours(self, level_sets, fname="", save_fig=False):
        for contour in level_sets:
            contour = np.array(contour).reshape(len(contour), 2)
            plt.plot(contour[:, 0], contour[:, 1])
        if save_fig:
            plt.savefig(fname, dpi=300)

    def plot_traj(self, traj, fname="", save_fig=False):
        traj = np.array(traj).reshape(len(traj), 2)
        plt.plot(traj[:, 0], traj[:, 1])
        if save_fig:
            plt.savefig(fname, dpi=300)

    def clf(self):
        plt.cla()

    def plot_V(self, model, vmin, vmax, fname="", save_fig=False):
        xs = np.linspace(0, self.env.xlim, 100)
        ys = np.linspace(0, self.env.ylim, 100)
        X, Y = np.meshgrid(xs, ys)
        shape = X.shape
 
        X = X.flatten()
        Y = Y.flatten()
        states = T.tensor(np.array([X,Y]))
        states = states.t().to('cuda:0')

        V = model.predict_values(states)
        V_cpu = V.to('cpu').detach().numpy()

        X = X.reshape(shape)
        Y = Y.reshape(shape)
        V_cpu = V_cpu.reshape(shape)
        
        plt.contourf(X, Y, V_cpu, zorder=-100, vmin=vmin, vmax=vmax)
        plt.colorbar()
        if save_fig:
            plt.savefig(fname, dpi=300)

    def plot_policy_arrows(self, probs, x, y):
        dx = np.array([0, 0, 1, 0, -1])
        dy = np.array([0, 1, 0, -1, 0])
        dx = np.multiply(probs[0, :], dx)
        dy = np.multiply(probs[0, :], dy)
        for i in range(5):
            plt.arrow(x, y, dx[i], dy[i], head_width=0.1)

    def plot_policy(self, agent, fname="", save_fig=False):
        xs = np.linspace(0, self.env.xlim, 6)
        ys = np.linspace(0, self.env.ylim, 6)
        X, Y = np.meshgrid(xs, ys)
        pi = {}
        # print("V.shape:", V.shape)
        for i in range(len(xs)):
            for j in range(len(ys)):
                x = X[i, j]
                y = Y[i, j]
                state = np.array([x, y], dtype=np.float32).reshape(
                    2,
                )
                state = T.tensor([state])
                # probs, V[i][j] = agent.actor_critic.forward(state)
                probs = agent.actor.forward(state)
                probs = F.softmax(probs, dim=1)
                probs = probs.detach().numpy()
                pi[(i, j)] = probs
                self.plot_policy_arrows(probs, x, y)
        print("check pi", pi[(2, 2)])
        if save_fig:
            plt.savefig(fname, dpi=300)

    def plot_series(self, series, label="", title="", fname="", save_fig=False):
        plt.plot(series, label=label)
        plt.title(title)
        plt.legend()
        if save_fig:
            plt.savefig(fname, dpi=300)
