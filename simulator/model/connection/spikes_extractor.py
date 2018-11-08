import torch
from random import choice

from simulator.core.connection import Connection
from simulator.core.group import Group


class SpikesExtractor(Connection):
    def __init__(self, pre: Group, post: Group, cuda=False, limit=5.0):
        super(SpikesExtractor, self).__init__(pre, post)
        self.limit = limit

        self.pre = pre
        self.post = post

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * limit
        # self.w *= (self.limit / self.w.sum(0).unsqueeze(-1)).repeat(1, self.w.shape[0]).permute(1, 0)
        # self.w_mask = torch.zeros_like(self.w)
        #
        # watched_neurons = list(range(pre.size))
        #
        # for i in range(post.size):
        #     neuron = choice(watched_neurons)
        #     watched_neurons.remove(neuron)
        #
        #     self.w_mask[neuron, i] = 1

        if cuda:
            self.w = self.w.cuda()
            # self.w_mask = self.w_mask.cuda()

    def step(self, spikes):
        pre_spikes = self.pre.spikes

        self.post.add_next((self.w.t() * pre_spikes).sum(1))


