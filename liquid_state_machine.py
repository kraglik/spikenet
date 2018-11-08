import ast

import torch
import pickle
import numpy as np

from simulator.core import Network, poisson_spike_train
from simulator.model.custom import LiquidStateMachine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas


load = False
use_cuda = True and torch.cuda.is_available()


def main():

    if load:
        net = torch.load('net.spiking_torch')
        lsm = net.lsm
    else:
        net = Network(log_limit=1000)
        lsm = LiquidStateMachine(
            net, "lsm",
            input_size=662,
            output_size=150,
            cuda=use_cuda
        )
        net.lsm = lsm

    vectors = pandas.read_csv("result.csv")

    net.time = 0
    # vectors = [vec for vec in vectors if len(vec) > 3]

    toggled = False
    toggle_time = 0
    readouts = []

    for s, vector in vectors.iterrows():
        spikes = torch.FloatTensor(poisson_spike_train(np.array(ast.literal_eval(vector[0])), 10, 50)).cuda()

        for i in range(spikes.shape[0]):
            input = spikes[i]

            net.step({'lsm': input})

        print(net.time / 1000, "seconds of simulation")

        if net.time > toggle_time and not toggled:
            lsm.toggle_learning(False)

        # if int(net.time) % 1000 == 0 and net.time > 1:
        #     group_1_activity = torch.stack(lsm.get_cache()).cpu().numpy().transpose()
        #     plt.matshow(group_1_activity)
        #     plt.title('Group 1 activity')
        #     plt.savefig('plots/lsm/%d.png' % int(net.time))
        #     plt.close()
        #
        #     del group_1_activity

        if int(net.time) % 50000 == 0:
            torch.save(net, 'net.spiking_torch')

        print("step ", s, " done")

        if net.time >= 400 and int(net.time) % 400 == 0:
            readout = lsm.get_readout_vector(400)

            readouts.append(readout)

        # if 10000 >= net.time > 200:
        #     readout = lsm.get_readout_vector(200)
        #     half_readout = lsm.get_readout_vector(100)
        #     avg_readout += torch.cat((readout, half_readout), dim=0)
        #
        #     readouts.append(torch.cat((readout, half_readout), dim=0))
        #
        # if net.time > 10000 and int(net.time) % 50 == 0:
        #     readout = lsm.get_readout_vector(100)
        #     half_readout = lsm.get_readout_vector(50)
        #
        #     readouts.append(torch.cat((readout, half_readout), dim=0))

            # distances.append((
            #     int(net.time / 25),
            #     distance(avg_readout, torch.cat((readout, half_readout), dim=0))
            # ))

        if int(net.time) % 50000 == 0 and net.time > 1:
            with open('readouts.pickle', 'wb') as f:
                pickle.dump(readouts, f)

    group_1_activity = torch.stack(lsm.get_cache()).cpu().numpy().transpose()
    # plt.figure(figsize=(40, 10))
    plt.matshow(group_1_activity)
    plt.title('Group 1 activity')
    plt.savefig('plots/liquid_state_machine_activity.png')
    plt.close()

    net.lsm = lsm
    torch.save(net, 'net.spiking_torch')

    with open('readouts.pickle', 'wb') as f:
        pickle.dump(readouts, f)


if __name__ == '__main__':
    main()