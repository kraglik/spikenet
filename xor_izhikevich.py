import torch
import numpy as np
import matplotlib.pyplot as plt

from simulator.core import *
from simulator.model.connection import *
from simulator.model.group import *


SAMPLE_STEPS = 50
NU = 1
EPOCHS = 3000

XOR = [
    ((1, 0), 1),
    ((0, 1), 1),
    ((1, 1), 0),
    ((0, 0), 0)
]

net = Network(SAMPLE_STEPS)
net.reward_manager.tau_d = 10

input_in_a = LeakyIFGroup(net, "inhibitory_a", 15)
input_in_b = LeakyIFGroup(net, "inhibitory_b", 15)
input_a = LeakyIFGroup(net, "input_a", 15)
input_b = LeakyIFGroup(net, "input_b", 15)

hidden = LeakyIFGroup(net, "hidden", 60, refractory=4.0)

out_true = LeakyIFGroup(net, "output", 1, refractory=2.0)
out_false = LeakyIFGroup(net, "output", 1, refractory=2.0)

i1 = RSTDP(input_in_a, hidden, nu=NU, minimum=-5, maximum=0)
i2 = RSTDP(input_in_b, hidden, nu=NU, minimum=-5, maximum=0)
c = RSTDP(input_a, hidden, nu=NU, maximum=5)
c2 = RSTDP(input_b, hidden, nu=NU, maximum=5)
RSTDP(hidden, out_true, nu=NU, maximum=5)
RSTDP(hidden, out_false, nu=NU, maximum=5)


def main():
    torch.no_grad()
    for epoch in range(EPOCHS):
        answers = []

        for (a, b), target in XOR:
            a_e = np.ones(15) * a * 20
            b_e = np.ones(15) * b * 20

            i_e_a = np.ones(15) * a * 20
            i_e_b = np.ones(15) * b * 20

            watch_spikes = True

            print("a:", a, "b:", b)

            for i in range(SAMPLE_STEPS):
                net.step({
                    'inhibitory_a': torch.FloatTensor(i_e_a),
                    'inhibitory_b': torch.FloatTensor(i_e_b),
                    'input_a': torch.FloatTensor(a_e),
                    'input_b': torch.FloatTensor(b_e)
                })

                if (out_true.spikes[0] > 0 or out_false.spikes[0] > 0) and watch_spikes:
                    if target == 1 and out_true.spikes[0] > 0 and out_false.spikes[0] < 1:
                        net.reinforce(1)
                        print(True)
                        answers.append(True)
                        watch_spikes = False

                    elif target == 0 and out_true.spikes[0] > 0 and out_false.spikes[0] < 1:
                        print(False)
                        answers.append(False)
                        watch_spikes = False

                    elif target == 1 and out_true.spikes[0] < 1 and out_false.spikes[0] > 0:
                        print(False)
                        answers.append(False)
                        watch_spikes = False

                    elif target == 0 and out_true.spikes[0] < 1 and out_false.spikes[0] > 0:
                        net.reinforce(1)
                        print(True)
                        answers.append(True)
                        watch_spikes = False

            if watch_spikes:
                print('default:', False)

                if a != 0 or b != 0:
                    net.reinforce(-1)

                answers.append(False)

            for i in range(50):
                net.step()

        print(answers)
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
