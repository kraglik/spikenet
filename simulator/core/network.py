from simulator.core.reward_manager import RewardManager


class Network:

    def __init__(self, log_limit=1000):
        self.groups = {}
        self.ordered_groups = []
        self.log = {}
        self.log_limit = log_limit
        self.time = 0.0
        self.reward_manager = RewardManager(self, tau_d=200)

    def make_observable(self, group_name):
        self.log[group_name] = []

    def set_reward_manager(self, rm):
        self.reward_manager = rm

    def reinforce(self, reward):
        self.reward_manager.reinforce(reward)

    def toggle_learning(self, value=True):
        for layer in self.ordered_groups:
            layer.toggle_learning(value)

    def rates(self, group_name, period, maximum=100):
        return sum(self.log[group_name][len(self.log[group_name]) - period:]) * (1000 / period) / maximum

    def step(self, inputs: dict = None):
        if inputs is not None:
            for name, input in inputs.items():
                self.groups[name].add_next(input)

        for group in self.ordered_groups:
            group.swap()
            group.step()

        for group in self.ordered_groups:
            group.update()

        self.reward_manager.step()
        self.time += 1
