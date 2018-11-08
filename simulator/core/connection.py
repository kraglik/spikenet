import torch
import torch.nn as nn
from abc import abstractmethod

from simulator.core.group import Group


class Connection(nn.Module):
    def __init__(self, pre: Group, post: Group):
        super(Connection, self).__init__()

        self.pre = pre
        self.post = post

        pre.add_post(self)
        post.add_pre(self)

        self.reward = 0.0
        self.learning = True

    def toggle_learning(self, enabled=True):
        self.learning = enabled

    def push(self, inputs: torch.FloatTensor):
        self.inputs.append(inputs)

    def reinforce(self, reward):
        self.reward = reward

    @abstractmethod
    def step(self, spikes):
        pass

    @abstractmethod
    def update(self):
        pass

