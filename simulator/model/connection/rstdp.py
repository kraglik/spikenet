import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class RSTDP(Connection):
    def __init__(self,
                 pre: Group,
                 post: Group,
                 cuda=False,
                 p=1,
                 nu_pre=1e-3,
                 nu_post=1.2e-3,
                 tau_c=250,
                 nu=1e-2,
                 minimum=0,
                 maximum=4,
                 inhibitory=False):
        super(RSTDP, self).__init__(pre, post)

        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.maximum = maximum
        self.minimum = minimum
        self.inhibitory = inhibitory
        self.nu = nu

        self.pre = pre
        self.post = post
        self.tau_c = tau_c

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * (maximum - minimum)
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()

        self.c_plus = torch.zeros_like(self.w)
        self.c_minus = torch.zeros_like(self.w)
        self.d = 0

        self.spike_cache = torch.zeros_like(self.w)
        self.spikes_cache_t = torch.zeros_like(self.w.t() * self.pre.spikes.cpu())
        self.spikes = torch.zeros_like(self.post.spikes)

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()
            self.c_plus = self.c_plus.cuda()
            self.c_minus = self.c_minus.cuda()

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
        pre_traces = self.pre.traces
        pre_spikes = self.pre.spikes

        post_traces = self.post.get_traces()
        post_spikes = self.post.get_spikes()

        dw_plus = self.nu_post * pre_traces.view(-1, 1) * post_spikes.view(1, -1)  # LTP
        dw_minus = self.nu_pre * pre_spikes.view(-1, 1) * post_traces.view(1, -1)  # LTD

        self.c_plus += dw_plus - self.c_plus / self.tau_c
        self.c_minus += dw_minus - self.c_minus / self.tau_c

        if self.inhibitory:
            self.w += self.nu * (self.c_minus - self.c_plus) * self.post.network.reward_manager.d
        else:
            self.w += self.nu * (self.c_plus - self.c_minus) * self.post.network.reward_manager.d

        self.w = torch.clamp(self.w, self.minimum, self.maximum)
