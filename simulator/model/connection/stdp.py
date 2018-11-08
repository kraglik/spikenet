import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class STDP(Connection):
    def __init__(self,
                 pre: Group,
                 post: Group,
                 cuda=False,
                 nu=1e-2,
                 nu_pre=1e-2,
                 nu_post=1.02e-2,
                 p=1,
                 limit=4.0):
        super(STDP, self).__init__(pre, post)

        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.nu = nu

        self.limit = limit

        self.pre = pre
        self.post = post

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * 0.15 + 0.075
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()

    def step(self, spikes):
        pre_spikes = self.pre.spikes

        self.post.add_next(((self.w_mask.t() * self.w.t()) * pre_spikes).sum(1))

    def update(self):
        pre_traces = self.pre.traces
        pre_spikes = self.pre.spikes

        post_traces = self.post.get_traces()
        post_spikes = self.post.get_spikes()

        dw = self.nu_post * pre_traces.view(-1, 1) * post_spikes.view(1, -1)  # LTP
        dw -= self.nu_pre * pre_spikes.view(-1, 1) * post_traces.view(1, -1)  # LTD

        self.w = torch.clamp(self.w + self.nu * dw, 0, self.limit)

