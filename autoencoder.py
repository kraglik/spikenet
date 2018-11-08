import pickle
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.ec1 = nn.Conv2d(1, 50, kernel_size=(3, 5))
        self.ep1 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
        self.ec2 = nn.Conv2d(50, 100, kernel_size=(2, 5))
        self.ep2 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
        self.ec3 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.ep3 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
        self.ec4 = nn.Conv2d(200, 100, kernel_size=(1, 3))

        self.el1 = nn.Linear(400, 200)
        self.el2 = nn.Linear(200, 100)

        self.dl1 = nn.Linear(100, 200)
        self.dl2 = nn.Linear(200, 400)

        self.dc1 = nn.ConvTranspose2d(100, 200, kernel_size=(1, 3))
        self.dp1 = nn.MaxUnpool2d(kernel_size=(1, 2))
        self.dc2 = nn.ConvTranspose2d(200, 100, kernel_size=(1, 5))
        self.dp2 = nn.MaxUnpool2d(kernel_size=(1, 2))
        self.dc3 = nn.ConvTranspose2d(100, 50, kernel_size=(2, 5))
        self.dp3 = nn.MaxUnpool2d(kernel_size=(1, 2))
        self.dc4 = nn.ConvTranspose2d(50, 1, kernel_size=(3, 5))

        self.train()

    def encode(self, readouts, return_indices=True):
        x = F.relu(self.ec1(readouts))
        x, p1 = self.ep1(x)
        x = F.relu(self.ec2(x))
        x, p2 = self.ep2(x)
        x = F.relu(self.ec3(x))
        x, p3 = self.ep3(x)
        x = F.relu(self.ec4(x))

        x = F.tanh(self.el1(x.view(-1)))
        x = F.tanh(self.el2(x))

        if return_indices:
            return x, (p3, p2, p1)
        else:
            return x

    def decode(self, encoding, indices):
        x = F.tanh(self.dl1(encoding))
        x = F.tanh(self.dl2(x))

        x = x.view(1, 100, 2, 2)

        x = F.relu(self.dc1(x))
        x = self.dp1(x, indices[0])
        x = F.relu(self.dc2(x))
        x = self.dp2(x, indices[1])
        x = F.relu(self.dc3(x))
        x = self.dp3(x, indices[2])
        x = self.dc4(x)

        return x

    def forward(self, readouts):
        encoding, indices = self.encode(readouts)
        return self.decode(encoding, indices)


def main():
    net = Autoencoder().cuda()
    net.optim = torch.optim.Adam(net.parameters())

    with open('readouts.pickle', 'rb') as f:
        readouts = pickle.load(f)

    train = [torch.stack(readouts[i - 5: i]) for i in range(5, 10000)]
    test = [torch.stack(readouts[i - 5: i]) for i in range(10005, len(readouts), 5)]

    for epoch in range(5):
        print('epoch %d' % epoch)

        random.shuffle(train)

        loss_f = nn.MSELoss()

        for i, sample in enumerate(train):
            net.optim.zero_grad()

            sample = sample.cuda().view(1, 1, 5, 60)
            restored = net.forward(sample)

            loss = loss_f(restored, sample)
            loss.backward()

            print('sample %d' % i, ', loss %f' % float(loss))

            net.optim.step()

        torch.save(net, 'net.torch')


if __name__ == '__main__':
    main()
