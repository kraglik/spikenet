import torch
import torch.random
import numpy as np
import matplotlib.pyplot as plt

from simulator.core import *
from simulator.model.group import *
from simulator.model.connection import *

NU = 1e-1

################### Network A ###################
net_a = Network(100)
net_a.reward_manager.tau_d = 1.0
net_a.reward_manager.d = 0.1

input_a = InputGroup(net_a, "input", 40)
hidden_a = LeakyIFGroup(net_a, "hidden", 30)
out_a = LeakyIFGroup(net_a, "out", 2)

StaticConnection(input_a, hidden_a)
conn = MESTDP(hidden_a, out_a, nu=NU)

################### Network B ###################
net_b = Network(100)
net_b.reward_manager.tau_d = 100.0
net_b.reward_manager.d = 0.1

input_b = InputGroup(net_b, "input", 40)
hidden_b = LeakyIFGroup(net_b, "hidden", 30)
out_b = LeakyIFGroup(net_b, "out", 2)

StaticConnection(input_b, hidden_b)
con2 = MESTDP(hidden_b, out_b, nu=NU)


STEP_SIZE = 200
inputs_all = poisson_spike_train(np.ones(40), rate=40, time=STEP_SIZE)
# input_rate_saver = poisson_spike_train(np.ones(2), rate=200, time=STEP_SIZE)


def main():
    epochs = 1000

    for epoch in range(epochs):
        for i in range(STEP_SIZE):
            net_a.step({
                'input': torch.FloatTensor(inputs_all[i]),
                # 'out': torch.FloatTensor(input_rate_saver[i])
            })
            net_b.step({
                'input': torch.FloatTensor(inputs_all[i]),
                # 'out': torch.FloatTensor(input_rate_saver[i])
            })

        rates_a = out_a.get_rates(STEP_SIZE)
        rates_b = out_b.get_rates(STEP_SIZE)

        skip = False

        # if rates_a[0] == rates_a[1]:
        #     skip = True
        #     net_a.reinforce(-1)
        #
        # if rates_b[0] == rates_b[1]:
        #     skip = True
        #     net_b.reinforce(-1)

        a = rates_a[1] >= rates_a[0]
        b = rates_b[1] >= rates_b[0]

        if not skip:
            if a and b:
                net_a.reinforce(-2)
                net_b.reinforce(-2)
                print("both are bad")
            elif not a and not b:
                net_a.reinforce(4)
                net_b.reinforce(4)
                print("both are good")
            elif a:
                net_a.reinforce(5)
                net_b.reinforce(-3)
                print("a is bad")
            else:
                net_a.reinforce(-3)
                net_b.reinforce(5)
                print("b is bad")

        print(rates_a, rates_b)

        for i in range(50):
            net_a.step()
            net_b.step()

        if epoch % 8 == 0:
            m = conn.w.numpy()
            m[0, 0] = 4.0
            plt.matshow(m)
            plt.title('Excitatory connection weights')
            plt.savefig('plots/c_w.png')
            plt.clf()
            plt.close()

            m = con2.w.numpy()
            m[0, 0] = 4.0
            plt.matshow(m)
            plt.title('Excitatory connection weights')
            plt.savefig('plots/c_w2.png')
            plt.clf()
            plt.close()


if __name__ == '__main__':
    main()
