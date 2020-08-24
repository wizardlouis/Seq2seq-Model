import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


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
        js = json.dumps(object)
        file = open(filepath + '//' + filename, 'w')
        file.write(js)
        file.close()
    elif type == 'npy':
        np.save(filepath + '//' + filename, object)
    pass


# read package from specific path
def loadpath(filepath, seq='seq.npz', loss='loss.npz', model='model.pth'):
    seq = np.load(filepath + '//' + seq)
    loss = np.load(filepath + '//' + loss)
    model = torch.load(filepath + '//' + model)
    return seq, loss, model


# return hidden state ndarray:(T,Batch_Size, n_neuro)
def get_trajectories(model, batch_seq):
    h_0 = model.reset_hidden()
    hidden, geometry, _ = model(h_0, batch_seq)
    seq_length = len(batch_seq[0])
    hidden = hidden.detach().numpy()
    # make a color map
    traj = hidden[:, 0, :]
    c = np.zeros(len(traj))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(seq_length):
        c[-model.t_retrieve + model.t_cue + i *
          (model.t_ron + model.t_roff):-model.t_retrieve + model.t_cue + i *
                                       (model.t_ron + model.t_roff) + model.t_ron] = 0
    return hidden, c


def get_candidates(net, seq_set, t_rest, t_on, t_off, t_ron, t_roff, t_delay,
                   t_retrieve, t_cue, step_work, step_delay):
    candidates = torch.tensor([])
    h_0 = net.reset_hidden()
    hidden, _, _ = net(h_0, seq_set)
    n_item = len(seq_set[0])
    for i in range(hidden.size()[1]):
        hidden_t = hidden[:, i, :]
        # for t in range(t_rest, t_rest + (n_item - 1) * (t_on + t_off) + t_on,
        #                step_work):
        #     candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)),
        #                            dim=0)
        # for t in range(-t_retrieve,
        #                -t_retrieve + t_cue + (n_item - 1) * (t_ron + t_roff) + t_ron,
        #                step_work):
        #     candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)),
        #                            dim=0)
        for t in range(-t_retrieve - t_delay, -t_retrieve, step_delay):
            candidates = torch.cat((candidates, hidden_t[t, :].unsqueeze(0)),
                                   dim=0)

    return candidates


def get_delay_points(net, seq_set, t_rest, t_on, t_off, t_ron, t_roff, t_delay,
                     t_retrieve, t_cue, step_delay):
    points = torch.tensor([])
    h_0 = net.reset_hidden()
    hidden, _, _ = net(h_0, seq_set)
    n_item = len(seq_set[0])
    for i in range(hidden.size()[1]):
        hidden_t = hidden[:, i, :]
        for t in range(-t_retrieve - t_delay, -t_retrieve, step_delay):
            points = torch.cat((points, hidden_t[t, :].unsqueeze(0)),
                               dim=0)
    return points


def get_rank(net, seq_set, rank, t_rest, t_on, t_off, t_ron, t_roff, t_delay,
             t_retrieve, t_cue, step_retrieve=1):
    points = torch.tensor([])
    h_0 = net.reset_hidden()
    hidden, _, _ = net(h_0, seq_set)
    for i in range(hidden.size()[1]):
        hidden_t = hidden[:, i, :]
        for t in range(-t_retrieve + (rank - 1) * (t_roff + t_ron), -t_retrieve + rank * (t_roff + t_ron),
                       step_retrieve):
            points = torch.cat((points, hidden_t[t, :].unsqueeze(0)),
                               dim=0)
    return points
