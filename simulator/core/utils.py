import torch
import numpy as np


def poisson_spike_train(vector, rate, time):
    n_input = vector.shape[0]

    vector = 1000 / (vector * rate)
    vector[np.isinf(vector)] = 0

    spike_times = np.random.poisson(vector, [time, n_input])
    spike_times = np.cumsum(spike_times, axis=0)

    spike_times[spike_times >= time] = 0.
    spikes = np.zeros([time, n_input])

    for idx in range(time):
        spikes[spike_times[idx, :], np.arange(n_input)] = 1
    spikes[0, :] = 0

    return spikes


def bernoulli_spike_train(data, time):
    max_prob = 1

    data = np.copy(data)
    shape, size = data.shape, data.size
    data = data.ravel()

    if data.max() > 1.0:
        data /= data.max()
    data *= max_prob

    s = np.random.binomial(1, data, [time, size])
    s = s.reshape([time, *shape])

    return torch.Tensor(s)


def get_im2col_indices(x_shape, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py.
    N, C, H, W = x_shape

    assert (H + 2 * padding[0] - field_height) % stride[0] == 0
    assert (W + 2 * padding[1] - field_height) % stride[1] == 0

    out_height = int((H + 2 * padding[0] - field_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - field_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=(1, 1), stride=(1, 1)):
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py

    p = padding

    x_padded = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return torch.Tensor(cols)


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]
