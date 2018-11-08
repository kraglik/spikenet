import matplotlib
import matplotlib.pyplot as plt
import pandas
from sklearn.manifold import TSNE
import pickle
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#
#         self.ec1 = nn.Conv2d(1, 50, kernel_size=(3, 5))
#         self.ep1 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
#         self.ec2 = nn.Conv2d(50, 100, kernel_size=(2, 5))
#         self.ep2 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
#         self.ec3 = nn.Conv2d(100, 200, kernel_size=(1, 5))
#         self.ep3 = nn.MaxPool2d(kernel_size=(1, 2), return_indices=True)
#         self.ec4 = nn.Conv2d(200, 100, kernel_size=(1, 3))
#
#         self.el1 = nn.Linear(400, 200)
#         self.el2 = nn.Linear(200, 100)
#
#         self.dl1 = nn.Linear(100, 200)
#         self.dl2 = nn.Linear(200, 400)
#
#         self.dc1 = nn.ConvTranspose2d(100, 200, kernel_size=(1, 3))
#         self.dp1 = nn.MaxUnpool2d(kernel_size=(1, 2))
#         self.dc2 = nn.ConvTranspose2d(200, 100, kernel_size=(1, 5))
#         self.dp2 = nn.MaxUnpool2d(kernel_size=(1, 2))
#         self.dc3 = nn.ConvTranspose2d(100, 50, kernel_size=(2, 5))
#         self.dp3 = nn.MaxUnpool2d(kernel_size=(1, 2))
#         self.dc4 = nn.ConvTranspose2d(50, 1, kernel_size=(3, 5))
#
#         self.train()
#
#     def encode(self, readouts, return_indices=True):
#         x = F.relu(self.ec1(readouts))
#         x, p1 = self.ep1(x)
#         x = F.relu(self.ec2(x))
#         x, p2 = self.ep2(x)
#         x = F.relu(self.ec3(x))
#         x, p3 = self.ep3(x)
#         x = F.relu(self.ec4(x))
#
#         x = F.tanh(self.el1(x.view(-1)))
#         x = F.tanh(self.el2(x))
#
#         if return_indices:
#             return x, (p3, p2, p1)
#         else:
#             return x
#
#     def decode(self, encoding, indices):
#         x = F.tanh(self.dl1(encoding))
#         x = F.tanh(self.dl2(x))
#
#         x = x.view(1, 100, 2, 2)
#
#         x = F.relu(self.dc1(x))
#         x = self.dp1(x, indices[0])
#         x = F.relu(self.dc2(x))
#         x = self.dp2(x, indices[1])
#         x = F.relu(self.dc3(x))
#         x = self.dp3(x, indices[2])
#         x = self.dc4(x)
#
#         return x
#
#     def forward(self, readouts):
#         encoding, indices = self.encode(readouts)
#         return self.decode(encoding, indices)
#
#
# net = torch.load('net.torch')
#
# with open('readouts.torch', 'rb') as f:
#     readouts = pickle.load(f)

with open('readouts.pickle', 'rb') as f:
    readouts = pickle.load(f)

with open('readouts.pickle', 'wb') as f:
    pickle.dump([x.cpu() for x in readouts], f)

with open('readouts.pickle', 'rb') as f:
    readouts = pickle.load(f)
    readouts = [x[0: 250] for x in readouts]

    p = 5

    readouts = [sum(readouts[i - p: i]) / p for i in range(p, len(readouts), p)]
# #
#     readouts = [net.encode(x.cuda().view(1, 1, 5, 60))[0].data.cpu() for x in readouts]
#
#     with open('readouts.torch', 'wb') as f:
#         pickle.dump(readouts, f)

#     print(len(readouts))

m = torch.stack(readouts, dim=0)

print(m.shape)

distance = torch.nn.CosineSimilarity()
x = torch.sum(m[0: 100], dim=0) / min(m.shape[0], 100)

print('averaging...')

print('calculating distances...')
distances = [1 - float(distance(readout.view(1, -1), x.view(1, -1))) for readout in readouts]

plt.title("Distances")
plt.plot(distances)
plt.xlabel("Time")
plt.ylabel("Cosine distance")
plt.savefig("plots/distances.png")
plt.close()

plt.title("readouts")
plt.title("Readouts")
plt.matshow(m.numpy().transpose())
plt.xlabel("Step")
plt.ylabel("Neuron")
plt.savefig("plots/readouts.png")
plt.close()

print('drawing plot...')
f = plt.figure(figsize=(9, 14))
s1 = f.add_subplot(3, 1, 1)
s1.set_title("Distances")
s1.plot(distances)
s1.set_xlabel("Time")
s1.set_ylabel("Cosine distance")

m = m.numpy().transpose()

s2 = f.add_subplot(3, 1, 2)
s2.set_title("Readouts")
s2.matshow(m)
s2.set_xlabel("Step")
s2.set_ylabel("Neuron")

s3 = f.add_subplot(3, 1, 3)
vs = pandas.read_csv('ac_mse_inputs.csv').as_matrix().transpose()
s3.set_title("Raw data")
s3.matshow(vs)

f.savefig("plots/all.png")
# embedded = TSNE(n_components=2, n_iter=1000, perplexity=50, early_exaggeration=45).fit_transform([x.numpy() for x in readouts])
#
# plt.figure(figsize=(15, 15))
# plt.title("t-SNE of readout vectors")
# plt.scatter(x=[x[0] for x in embedded], y=[x[1] for x in embedded])
# plt.savefig("plots/t-sne-readouts.png")
# plt.close()

