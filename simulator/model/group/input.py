from simulator.core.group import Group

import torch
import torch.nn as nn


class InputGroup(Group):
    def __init__(self, network, name: str, size: int, cuda=False, excitatory=True):
        super(InputGroup, self).__init__(network, name, size, cuda, excitatory=excitatory)
        self.cuda = cuda
        self.t = 0.0

    def reinforce(self, reward: float):
        for connection in self.post:
            connection.reinforce(reward)

    def step(self):

        spikes = self.inputs.clamp(0.0, 1.0).ceil()
        self.spikes = spikes

        if self.cuda:
            self.spikes = self.spikes.cuda()

        self._update_traces(spikes)

        for connection in self.post:
            connection.step(spikes)

        self.t += 1

        self.cache.append(spikes)
        if len(self.cache) > self.network.log_limit:
            self.cache.pop(0)

        return spikes