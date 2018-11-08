import torch
import numpy as np
import matplotlib.pyplot as plt

from simulator.core import *
from simulator.model.connection import *
from simulator.model.group import *


SAMPLE_STEPS = 200
NU = 1e-3
EPOCHS = 3000

XOR = [
    ((1, 0), 1),
    ((0, 1), 1),
    ((1, 1), 0),
    ((0, 0), 0)
]

net = Network(SAMPLE_STEPS)
net.reward_manager.tau_d = 2

input_in_a = InputGroup(net, "inhibitory_a", 15)
input_in_b = InputGroup(net, "inhibitory_b", 15)
input_a = InputGroup(net, "input_a", 15)
input_b = InputGroup(net, "input_b", 15)

hidden = LeakyIFGroup(net, "hidden", 60, refractory=4.0)

out = LeakyIFGroup(net, "output", 1)

i1 = MESTDP(input_in_a, hidden, nu=NU, minimum=-5, maximum=0)
i2 = MESTDP(input_in_b, hidden, nu=NU, minimum=-5, maximum=0)
c = MESTDP(input_a, hidden, nu=NU, maximum=5)
c2 = MESTDP(input_b, hidden, nu=NU, maximum=5)
MESTDP(hidden, out, nu=NU, maximum=5)


def main():
    torch.no_grad()
    for epoch in range(EPOCHS):
        for (a, b), target in XOR:
            a_e = poisson_spike_train(np.ones(15), 40 * a, SAMPLE_STEPS)
            b_e = poisson_spike_train(np.ones(15), 40 * b, SAMPLE_STEPS)

            i_e_a = poisson_spike_train(np.ones(15), 50 * a, SAMPLE_STEPS)
            i_e_b = poisson_spike_train(np.ones(15), 50 * b, SAMPLE_STEPS)

            for i in range(SAMPLE_STEPS):
                net.step({
                    'inhibitory_a': torch.FloatTensor(i_e_a[i]),
                    'inhibitory_b': torch.FloatTensor(i_e_b[i]),
                    'input_a': torch.FloatTensor(a_e[i]),
                    'input_b': torch.FloatTensor(b_e[i])
                })

                if out.spikes[0] > 0:
                    if target == 1:
                        net.reinforce(5)
                    elif target == 0:
                        net.reinforce(-5)

            print("a:", a, "b:", b, "target:", target, "rates:", out.get_rates(SAMPLE_STEPS))
            print("reward:", net.reward_manager.d)

            for i in range(50):
                net.step()

        print("-" * 40)

        m = c.w.numpy()
        m[0, 0] = 4.0
        plt.matshow(m)
        plt.title('Excitatory connection weights')
        plt.savefig('plots/c_w.png')
        plt.clf()
        plt.close()

        m = c2.w.numpy()
        m[0, 0] = 4.0
        plt.matshow(m)
        plt.title('Excitatory connection weights')
        plt.savefig('plots/c_w2.png')
        plt.clf()
        plt.close()

        m = i1.w.numpy() * -1
        m[0, 0] = 4.0
        plt.matshow(m)
        plt.title('Excitatory connection weights')
        plt.savefig('plots/c_iw.png')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()
