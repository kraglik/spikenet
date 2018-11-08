import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class MESTDP(Connection):
    def __init__(self,
                 pre: Group,
                 post: Group,
                 cuda=False,
                 nu=1e-2,
                 p=1,
                 a_plus=1,
                 a_minus=-1,
                 tau_c=0.5,
                 minimum=0,
                 maximum=5,
                 reversed=False):
        super(MESTDP, self).__init__(pre, post)
        self.maximum = maximum
        self.minimum = minimum

        self.pre = pre
        self.post = post

        self.tau_c_plus = 0.05
        self.tau_c_minus = 0.05

        self.a_plus = a_plus
        self.a_minus = a_minus

        self.tau_c_trace = tau_c

        self.p_plus = 0.0
        self.p_minus = 0.0
        self.reversed = reversed

        self.nu = nu

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * (maximum - minimum) + minimum
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()
        self.a = torch.zeros_like(self.w)
        self.b = torch.zeros_like(self.w)
        self.spike_cache = torch.zeros_like(self.w)
        self.spikes_cache_t = torch.zeros_like(self.w.t() * self.pre.spikes.cpu())
        self.spikes = torch.zeros_like(self.post.spikes)

        self.c = torch.zeros_like(self.w)

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()
            self.c = self.c.cuda()
            self.a = self.a.cuda()
            self.b = self.b.cuda()

            self.spikes = self.spikes.cuda()
            self.spike_cache = self.spike_cache.cuda()
            self.spikes_cache_t = self.spikes_cache_t.cuda()

    def step(self, spikes):
        torch.mul(self.w_mask, self.w, out=self.spike_cache)
        torch.mul(self.spike_cache.t(), self.pre.spikes, out=self.spikes_cache_t)
        torch.sum(self.spikes_cache_t, 1, out=self.spikes)
        torch.mul(self.spikes, self.pre.sign, out=self.spikes)
        self.post.add_next(self.spikes)

    def update(self):
        d = self.post.network.reward_manager.d

        pre_traces = self.pre.traces.unsqueeze(-1)
        pre_spikes = self.pre.spikes.unsqueeze(-1)

        post_traces = self.post.spikes.unsqueeze(0)
        post_spikes = self.post.traces.unsqueeze(0)

        self.p_plus = -(self.tau_c_plus * self.p_plus) + self.a_plus * pre_traces
        self.p_minus = -(self.tau_c_minus * self.p_minus) + self.a_minus * post_traces
        torch.mm(self.p_plus, post_spikes, out=self.a)
        torch.mm(pre_spikes, self.p_minus, out=self.b)

        self.c += -(self.tau_c_trace * self.c) + self.a + self.b

        self.w += self.nu * self.c * d
        self.w = torch.clamp(self.w, self.minimum, self.maximum)

