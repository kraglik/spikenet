class RewardManager:
    def __init__(self, net, tau_d):
        self.net = net
        self.d = 0.01
        self.tau_d = tau_d
        self.reward = 0
        net.set_reward_manager(self)

    def step(self):
        self.d -= self.d / self.tau_d
        self.d += self.reward
        self.reward = 0

    def reinforce(self, reward):
        self.reward += reward
