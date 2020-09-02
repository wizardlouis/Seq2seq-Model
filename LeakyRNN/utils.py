import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from configs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from gene_seq import rank_batch2input_batch


def pf(a1, a2): plt.figure(figsize=(a1, a2))


def tn(x): return x.cpu().detach().numpy()


def tt(x, dtype=torch.float, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)


# saving different types of files
def save(object, filepath, filename, type, *args, **kwargs):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if type == 'dict':
        np.save(filepath + '//' + filename, object)
    elif type == 'dictxt':
        # js = json.dumps(object)
        file = open(filepath + '//' + filename, 'w')
        file.write(str(object))
        file.close()
    elif type == 'npy':
        np.save(filepath + '//' + filename, object)
    pass


# read package from specific path
def load_path(filepath, seq='seq.npz', loss='loss.npz', model='model.pth', data_hps='data_hps.npy'):
    seq = np.load(filepath + '//' + seq)
    loss = np.load(filepath + '//' + loss)
    model = torch.load(filepath + '//' + model)['model']
    data_hps = np.load(filepath + '//' + data_hps).item()
    return seq, loss, model, data_hps


def get_hidden(model, rank_batch, hps):
    """

    :param model: network
    :param rank_batch: a batch of sequence in rank form
    :param hps: some hyper-parameters of input data
    :return: hidden state numpy.ndarray:(seq_len, batch_size, n_neuron) and a color map for visualization
    """
    # t_rest = hps['t_rest']
    # t_on = hps['t_on']
    # t_off = hps['t_off']
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']
    rank_size = hps['rank_size']

    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)

    h_0 = model.reset_hidden()
    hidden, _, _, _ = model(h_0, input_batch)
    hidden = hidden.detach().numpy()
    # make a color map
    traj = hidden[:, 0, :]
    c = np.zeros(len(traj))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(rank_size):
        c[-t_retrieve + t_cue + i * (t_ron + t_roff):-t_retrieve + t_cue + i * (t_ron + t_roff) + t_ron] = 0
    return hidden, c


def get_candidates(net, rank_batch, step_work, step_delay, hps):
    """
    get some candidates from trajectories
    :param net:
    :param rank_batch:
    :param step_work: the interval between points got from stimulus stage and retrieve stage
    :param step_delay: the interval between points got from delay stage
    :param hps:
    :return: torch.tensor, size:(candidates number, hidden_size)
    """
    t_rest = hps['t_rest']
    t_on = hps['t_on']
    t_off = hps['t_off']
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']
    rank_size = hps['rank_size']

    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)

    h_0 = net.reset_hidden()
    hidden, _, _, _ = net(h_0, input_batch)
    candidates = torch.tensor([])

    for i in range(hidden.size()[1]):
        hidden_t = hidden[:, i, :]  # the i st sequence in this batch (seq_len, hidden_size)
        # for t in range(t_rest, t_rest + (rank_size - 1) * (t_on + t_off) + t_on, step_work):
        #     candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)),
        #                            dim=0)
        # for t in range(-t_retrieve, -t_retrieve + t_cue + (rank_size - 1) * (t_ron + t_roff) + t_ron, step_work):
        #     candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)), dim=0)
        for t in range(-t_retrieve - t_delay, -t_retrieve, step_delay):
            candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)), dim=0)

    return candidates


def get_hidden_delay(net, rank_batch, step_delay, hps):
    """
    get hidden state each step_delay from delay stage
    :param net:
    :param rank_batch:
    :param step_delay: the interval between points got from delay stage
    :param hps:
    :return: torch.tensor, size:(t_delay, batch_size, hidden_size)
    """

    t_retrieve = hps['t_retrieve']
    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)
    h_0 = net.reset_hidden()
    hidden, _, _, _ = net(h_0, input_batch)
    points = hidden[-t_retrieve - t_delay: -t_retrieve, :, :]
    # points = torch.tensor([])
    # for i in range(hidden.size()[1]):
    #     hidden_t = hidden[:, i, :]
    #     for t in range(-t_retrieve - t_delay, -t_retrieve, step_delay):
    #         points = torch.cat((points, hidden_t[t, :].unsqueeze(0)),
    #                            dim=0)
    return points


def get_hidden_retrieve(net, rank_batch, rank, hps, step_retrieve=1):
    """
    get hidden state each step_retrieve from retrieve stage
    :param net:
    :param rank_batch:
    :param step_retrieve: the interval between points got from delay stage
    :param hps:
    :return: torch.tensor, size:(points number, hidden_size)
    """
    points = torch.tensor([])
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']

    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)

    h_0 = net.reset_hidden()
    hidden, _, _, _ = net(h_0, input_batch)
    for i in range(hidden.size()[1]):
        hidden_t = hidden[:, i, :]
        for t in range(-t_retrieve + (rank - 1) * (t_roff + t_ron), -t_retrieve + rank * (t_roff + t_ron),
                       step_retrieve):
            points = torch.cat((points, hidden_t[t, :].unsqueeze(0)),
                               dim=0)
    return points
