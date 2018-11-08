import math

import torch

from simulator.core.group import Group
from simulator.model.connection import *
from simulator.model.connection.spikes_extractor import SpikesExtractor
from simulator.model.connection.static import StaticConnection
from simulator.model.group import IzhikevichGroup, InputGroup
from simulator.model.group.leaky_if import LeakyIFGroup


class DimensionReductor(Group):
    def __init__(self, network, name, input_size=625, hidden_size=625, output_size=125, cuda=False):
        super(DimensionReductor, self).__init__(network, name, input_size, cuda)

        self.input = InputGroup(network, name + "_input", self.size, cuda=cuda)
        self.hidden = LeakyIFGroup(network, name + "_hidden", hidden_size, cuda=cuda, refractory=5.0)
        self.outputs = LeakyIFGroup(network, name + "_outputs", output_size, cuda=cuda, refractory=4.0)

        self.x_to_h = StaticConnection(self.input, self.outputs, p=30 / input_size, limit=1.0, cuda=cuda)
        self.h_to_h = StaticConnection(self.hidden, self.hidden, p=12 / hidden_size, limit=1.0, cuda=cuda)
        self.h_to_o = StaticConnection(self.hidden, self.outputs, p=30 / hidden_size, limit=2.7, cuda=cuda)
        self.o_to_o = StaticConnection(self.outputs, self.outputs, p=12 / output_size, limit=1.0, cuda=cuda)

        self.cache = self.outputs.cache

    def add_next(self, inputs):
        self.input.add_next(inputs)

    def add_post(self, post):
        self.outputs.add_post(post)

    def add_pre(self, pre):
        self.neurons.add_pre(pre)

    def get_traces(self):
        return self.neurons.traces

    def get_spikes(self):
        return self.neurons.spikes

    def get_cache(self):
        return self.outputs.cache

    def get_readout_vector(self, steps_count=25):
        return sum(self.outputs.cache[len(self.outputs.cache) - steps_count:]) / steps_count
