import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from simulator.core import *
from simulator.model.connection import *
from simulator.model.group import *

from torchvision import datasets, transforms


SAMPLE_STEPS = 100
NU = 1
EPOCHS = 5

net = Network(SAMPLE_STEPS)
net.reward_manager.tau_d = 10

input_ex = LeakyIFGroup(net, "excitatory_input", 28 * 28, cuda=True)

hidden = LeakyIFGroup(net, "hidden", 50, refractory=2.0, cuda=True)

out = LeakyIFGroup(net, "output", 10, refractory=1.0, cuda=True)

e_c = RSTDP(input_ex, hidden, nu=NU, minimum=0, maximum=5, cuda=True)
h_c = RSTDP(hidden, out, nu=NU, minimum=0.001, maximum=5, cuda=True)


def main():
    torch.no_grad()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    for epoch in range(200):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 20:
                break

            data = data.view(-1).cuda()
            target = int(target)

            watch = True

            prediction = -1

            spiked = False

            for i in range(SAMPLE_STEPS):
                net.step({
                    'excitatory_input': data
                })

                if sum(out.spikes) > 0:
                    spiked = True

                if sum(out.spikes) == 1 and watch:
                    out_spikes = out.spikes.cpu().numpy()

                    prediction = np.argmax(out_spikes)
                    watch = False

            if prediction == target:
                net.reinforce(1)
            elif prediction == -1:
                net.reinforce(-0.5)
            else:
                net.reinforce(-0.25)

            if spiked is False:
                net.reinforce(-5)

            print('sample:', batch_idx,
                  'target:', target,
                  'prediction:', prediction,
                  'current reward:', net.reward_manager.d,
                  '-> match' if target == prediction else ''
                  )

            print('spiked at least once:', spiked)

            if batch_idx % 10 == 0:
                m = e_c.w.cpu().numpy()
                m[0, 0] = 4.0
                plt.matshow(m)
                plt.title('Excitatory connection weights')
                plt.savefig('plots/e_c.png')
                plt.clf()
                plt.close()

                m = h_c.w.cpu().numpy() * -1
                m[0, 0] = 4.0
                plt.matshow(m)
                plt.title('Hidden connection weights')
                plt.savefig('plots/h_c.png')
                plt.clf()
                plt.close()

            if batch_idx % 10 == 0:
                torch.save(net, 'mnist.spiking_torch')


if __name__ == '__main__':
    main()
