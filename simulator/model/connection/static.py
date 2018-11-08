import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class StaticConnection(Connection):
    def __init__(self, pre: Group, post: Group, cuda=False, delay = 0, p=0.5, limit=4.0):
        super(StaticConnection, self).__init__(pre, post)
        self.limit = limit

        self.pre = pre
        self.post = post

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * limit
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()

        self.spike_cache = torch.zeros_like(self.w)
        self.spikes_cache_t = torch.zeros_like(self.w.t() * self.pre.spikes.cpu())
        self.spikes = torch.zeros_like(self.post.spikes)

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()

            self.spikes = self.spikes.cuda()
            self.spike_cache = self.spike_cache.cuda()
            self.spikes_cache_t = self.spikes_cache_t.cuda()

    def step(self, spikes):
        torch.mul(self.w_mask, self.w, out=self.spike_cache)
        torch.mul(self.spike_cache.t(), self.pre.spikes, out=self.spikes_cache_t)
        torch.sum(self.spikes_cache_t, 1, out=self.spikes)
        torch.mul(self.spikes, self.pre.sign, out=self.spikes)

        self.post.add_next(self.spikes)


