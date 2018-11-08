import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class MSTDP(Connection):
    def __init__(self,
                 pre: Group,
                 post: Group,
                 cuda=False,
                 delay=0,
                 p=1,
                 nu=1e-3,
                 a_plus=1,
                 a_minus=-1,
                 reversed=False,
                 minimum=0,
                 maximum=4):
        super(MSTDP, self).__init__(pre, post)

        self.nu = nu
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.minimum = minimum
        self.maximum = maximum

        self.pre = pre
        self.post = post

        self.reversed = reversed

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * (maximum - minimum) + minimum
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()

    def step(self, spikes):
        pre_spikes = self.pre.spikes.unsqueeze(0)

        output = ((self.w_mask * self.w).t() * pre_spikes).sum(1)

        self.post.add_next(output * self.pre.sign)

    def update(self):
        d = self.post.network.reward_manager.d

        pre_traces = self.pre.traces.unsqueeze(-1)
        pre_spikes = self.pre.spikes.unsqueeze(-1)

        post_traces = self.post.get_traces().unsqueeze(0)
        post_spikes = self.post.get_spikes().unsqueeze(0)

        p_plus = self.a_plus * pre_traces
        p_minus = self.a_minus * post_traces

        c = p_plus * post_spikes + pre_spikes * p_minus

        self.w += self.nu * c * d
        self.w = torch.clamp(self.w, self.minimum, self.maximum)

