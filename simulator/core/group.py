import torch
import torch.nn as nn
from abc import abstractmethod


class Group(nn.Module):
    def __init__(self, network, name: str, size: int, cuda=False, tau_traces=20.0, excitatory=True):
        super(Group, self).__init__()
        self.network = network
        network.groups[name] = self
        network.ordered_groups.append(self)
        self.name = name
        self.size = size
        self.tau_traces = tau_traces
        self.traces = torch.zeros(size).float()
        self.spikes = torch.zeros(size).float()
        self.inputs = torch.zeros(size).float()
        self.next_inputs = torch.zeros(size).float()
        self.sign = 1.0 if excitatory else -1.0

        if cuda:
            self.traces = self.traces.cuda()
            self.inputs = self.inputs.cuda()
            self.spikes = self.spikes.cuda()
            self.next_inputs = self.next_inputs.cuda()

        self.pre = []
        self.post = []

        self.cache = []

    def toggle_learning(self, enabled=True):
        for conn in self.post:
            conn.toggle_learning(enabled)

    def add_next(self, inputs: torch.FloatTensor):
        self.next_inputs += inputs

    def reinforce(self, reward):
        for conn in self.pre:
            conn.reinforce(reward)

    def swap(self):
        self.inputs, self.next_inputs = self.next_inputs, self.inputs
        self.next_inputs.zero_()

    def forward(self):
        self.step()

    def add_post(self, post):
        self.post.append(post)

    def add_pre(self, pre):
        self.pre.append(pre)

    def get_traces(self):
        return self.traces

    def get_spikes(self):
        return self.spikes

    def get_rates(self, period):
        return torch.sum(torch.stack(self.cache[-period:], dim=0), dim=0)

    def update(self):
        for conn in self.pre:
            conn.update()

    @abstractmethod
    def step(self):
        pass

    def _update_traces(self, spikes):
        self.traces *= (1 - spikes)
        self.traces -= self.traces / self.tau_traces
        self.traces += spikes

    def rates(self, period, maximum=100):
        return sum(self.cache[len(self.cache) - period:]) * (1000 / period) / maximum
