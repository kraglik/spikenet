from simulator.core import Group
import torch


class LeakyIFGroup(Group):
    def __init__(self,
                 network, name: str,
                 size: int,
                 cuda=False,
                 rest=-65.0,
                 reset=-65.0,
                 threshold=-52.0,
                 refractory=5.0,
                 voltage_decay=1e-2,
                 tau_trace=5e-2,
                 excitatory=True):
        super(LeakyIFGroup, self).__init__(network, name, size, cuda, excitatory=excitatory)

        self.threshold = threshold
        self.rest = rest
        self.reset=reset
        self.size = size
        self.t = 0.0
        self.refrac = refractory

        self.voltage_decay = voltage_decay
        self.trace_tau = tau_trace

        self.v = torch.zeros(size).float() + self.rest

        self.spikes = torch.zeros(size)
        self.refractory = torch.zeros(size)

        if cuda:
            self.v = self.v.cuda()
            self.spikes = self.spikes.cuda()
            self.refractory = self.refractory.cuda()

    def reinforce(self, reward: float):
        for connection in self.post:
            connection.reinforce(reward)

    def step(self):
        v, rest, reset, thr = self.v, self.rest, self.reset, self.threshold

        self.inputs *= 1 - torch.clamp(self.refractory, 0, 1)

        self.v -= self.voltage_decay * (self.v - self.rest)

        spikes = (v - thr).clamp(0.0, 1.0).ceil()
        non_spikes = (1.0 - spikes)

        self.refractory = self.refractory * non_spikes + spikes * self.refrac

        self.v = v * non_spikes + reset * spikes  # Resetting membrane potential of fired neurons

        self._update_traces(spikes)

        for connection in self.post:
            connection.step(spikes)

        self.t += 1

        self.spikes = spikes

        self.cache.append(spikes)

        torch.clamp(self.refractory - 1, 0, self.refrac, out=self.refractory)

        self.v += self.inputs

        if len(self.cache) > self.network.log_limit:
            del self.cache[0]

    def _reset(self):
        super()._reset()

        self.v = self.rest * torch.ones(self.size)
        self.refractory = torch.zeros(self.size)


