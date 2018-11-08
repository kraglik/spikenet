from simulator.core.group import Group

import torch
import torch.nn as nn


class IzhikevichGroup(Group):
    def __init__(self,
                 network, name: str,
                 size: int,
                 cuda=False,
                 a=0.02,
                 b=0.2,
                 d=8.0,
                 c=-65.0,
                 reset=-65.0,
                 threshold=25.0,
                 refrac=5.0,
                 excitatory=True):
        super(IzhikevichGroup, self).__init__(network, name, size, cuda, excitatory=excitatory)

        self.a = a
        self.b = b
        self.d = d
        self.c = c
        self.threshold = threshold
        self.reset=reset
        self.size = size
        self.t = 0.0
        self.refrac=refrac

        self.v = torch.zeros(size).float() + self.c
        self.u = torch.zeros(size).float() + self.c * self.b

        self.spikes = torch.zeros(size)
        self.refractory = torch.zeros(size)

        if cuda:
            self.v = self.v.cuda()
            self.u = self.u.cuda()
            self.spikes = self.spikes.cuda()
            self.refractory = self.refractory.cuda()

    def reinforce(self, reward: float):
        for connection in self.post:
            connection.reinforce(reward)

    def step(self):
        v, u, a, b, c, d, thr = self.v, self.u, self.a, self.b, self.c, self.d, self.threshold

        self.inputs *= 1 - torch.clamp(self.refractory, 0, 1)

        v += 0.5 * ((0.04 * v + 5.0) * v + 140 - u + self.inputs)
        u += 0.5 * a * (b * v - u)

        v += 0.5 * ((0.04 * v + 5.0) * v + 140 - u + self.inputs)
        u += 0.5 * a * (b * v - u)

        spikes = (v - thr).clamp(0.0, 1.0).ceil()
        non_spikes = (1.0 - spikes)

        self.refractory = self.refractory * non_spikes + spikes * self.refrac

        self.v = v * non_spikes + c * spikes  # Resetting membrane potential of fired neurons
        self.u = u + d * spikes

        self._update_traces(spikes)

        for connection in self.post:
            connection.step(spikes)

        self.t += 1

        self.spikes = spikes

        self.cache.append(spikes)

        self.refractory = torch.clamp(self.refractory - 1, 0, self.refrac)

        if len(self.cache) > self.network.log_limit:
            del self.cache[0]

    def _reset(self):
        super()._reset()
        self.v = torch.ones(self.size) * self.c
        self.u = self.v / 5