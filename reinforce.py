import torch

from simulator.core.network import Network
from simulator.core.reward_manager import RewardManager

from simulator.model.connection import *
from simulator.model.group import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.cuda

import random

use_cuda = False and torch.cuda.is_available()


def main():
    net = Network(log_limit=500)

    rm = RewardManager(net, tau_d=20.0)

    g1 = LeakyIFGroup(
        network=net,
        name="g1",
        size=40,
        cuda=use_cuda,
        excitatory=True
        # threshold=25.0,
        # a=0.02,
        # b=0.2,
        # c=-65.0,
        # d=2.0
    )

    g2 = LeakyIFGroup(
        network=net,
        name="inh",
        size=10,
        cuda=use_cuda,
        excitatory=False
        # a=0.1,
        # d=2.0,
        # c=-60.0,
        # reset=-45.0,
        # threshold=-40.0
    )

    g1r = RSTDP(g1, g1, p=1, cuda=use_cuda, nu=1e-1)
    g1g2 = MSTDP(g1, g2, p=1, cuda=use_cuda, nu=1e-1)
    g2g1 = MSTDP(g2, g1, p=1, cuda=use_cuda, nu=1e-1)

    fig = plt.figure(num=10)
    fig.set_size_inches(w=5, h=5)
    ax = fig.add_subplot(111)
    ax.matshow(g1r.w.cpu().numpy())
    ax.set_title('Group 1 connection weights matrix')
    ax.set_xlabel("Weight")
    ax.set_ylabel("Neuron")
    fig.savefig('plots/g1r_rp_initial.png')
    fig.clf()

    net.step()

    while net.time < 1000 * 25:
        if use_cuda:
            input = torch.cuda.FloatTensor(g1.size).uniform_() * 4
        else:
            input = torch.rand(g1.size) * 4

        input[21] += 10
        input[22] += 10
        input[23] += 10
        input[24] += 10
        input[25] += 10

        net.step({'g1': input})

        if g1.cache[-1][19] > 0:
            net.reinforce(-1)
        # elif g1.cache[-1][20] > 0 and any(g1.cache[-2][x] > 0 for x in [18, 19]):
        #     net.reinforce(-0.01)
        if g1.cache[-1][20] > 0:
            net.reinforce(1)

        if int(net.time) % 1000 == 0:
            group_1_activity = torch.stack(g1.cache).cpu().numpy().transpose()
            fig = plt.figure(num=10)
            fig.set_size_inches(w=15, h=3)
            ax = fig.add_subplot(111)
            ax.matshow(group_1_activity)
            ax.set_title('Group 1 activity')
            fig.savefig('plots/group_1_activity.png')
            fig.clf()

            if int(net.time) % 1000 == 0:
                x = g1r.w.cpu().numpy().transpose()
                x[0, 20] = 1
                fig = plt.figure(num=10)
                fig.set_size_inches(w=5, h=5)
                ax = fig.add_subplot(111)
                ax.matshow(x)
                ax.set_title('Group 1 connection weights matrix')
                ax.set_xlabel("Weight")
                ax.set_ylabel("Neuron")
                fig.savefig('plots/rp_reinforced.png')
                fig.clf()

            print(int(net.time / 1000), "seconds of simulation")

    # plt.matshow(g1r.w.cpu().numpy())
    # plt.title('Group 1 connection weights matrix')
    # plt.savefig('plots/g1r.png')
    # plt.close()

    group_1_activity = torch.stack(g1.cache).cpu().numpy().transpose()
    plt.matshow(group_1_activity)
    plt.title('Group 1 activity')
    plt.savefig('plots/group_1_activity.png')
    plt.close()


if __name__ == '__main__':
    main()