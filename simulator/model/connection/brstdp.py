import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.connection import Connection
from simulator.core.group import Group


class BRSTDP(Connection):
    def __init__(self,
                 pre: Group,
                 post: Group,
                 cuda=False,
                 delay=0,
                 p=0.5,
                 nu=1,
                 nu_pre=1e-3,
                 nu_post=1.02e-3,
                 tau_c=1000,
                 minimum=0,
                 reversed=False,
                 limit=3):
        super(BRSTDP, self).__init__(pre, post, delay)

        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.nu = nu
        self.limit = limit
        self.minimum = minimum

        self.pre = pre
        self.post = post
        self.tau_c = tau_c

        self.reversed = reversed

        pre.post.append(self)
        post.pre.append(self)

        self.w = torch.rand((pre.size, post.size)).float() * limit / 2
        self.w_mask = (torch.rand((pre.size, post.size)).float() - (1 - p)).ceil()

        self.c_plus = torch.zeros_like(self.w)
        self.c_minus = torch.zeros_like(self.w)
        self.d = 0

        if cuda:
            self.w = self.w.cuda()
            self.w_mask = self.w_mask.cuda()
            self.c = self.c.cuda()

    def step(self, spikes):
        self.spikes.append(spikes)
        self.traces.append(self.pre.get_traces())

        if len(self.spikes) < self.delay:
            return

        pre_traces = self.traces.pop()
        pre_spikes = self.spikes.pop()

        post_traces = self.post.get_traces()
        post_spikes = self.post.get_spikes()

        self.d = self.post.network.reward_manager.d

        if self.learning:
            self.update(pre_traces, post_traces, pre_spikes, post_spikes)

        output = ((self.w_mask * self.w).t() * pre_spikes).sum(1)

        self.post.add_next(output * self.pre.sign)

    def update(self, pre_traces, post_traces, pre_spikes, post_spikes):
        dw_plus = self.nu_post * pre_traces.view(-1, 1) * post_spikes.view(1, -1)   # LTP
        dw_minus = self.nu_pre * pre_spikes.view(-1, 1) * post_traces.view(1, -1)   # LTD

        self.c_plus += dw_plus - self.c_plus / self.tau_c
        self.c_minus += dw_minus - self.c_minus / self.tau_c

        if not self.reversed:
            self.w += self.nu * (self.c_plus - self.c_minus) * self.d
        else:
            self.w += self.nu * (self.c_minus - self.c_plus) * self.d
        self.w = torch.clamp(self.w, self.minimum, self.limit)

