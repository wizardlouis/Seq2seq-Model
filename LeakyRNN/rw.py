import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from network import Direction


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


# Batch_Seq(Batch_Size,seq_length)
def plot_trajectory(Net, Batch_Seq, row, column=5):
    Direct = np.array(Direction)
    h_0 = Net.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _ = Net(h_0, Batch_Seq)
    geometry = geometry.detach().numpy()
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(len(Batch_Seq[0])):
        c[-Net.t_retrieve + Net.t_cue + i * (Net.t_ron + Net.t_roff):-Net.t_retrieve + Net.t_cue + i * (
                    Net.t_ron + Net.t_roff) + Net.t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(Batch_Seq)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:, 0], Direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        plt.text(-2, -2, str(Batch_Seq[k]))


def plot_trajectory2(Net, Batch_Seq, row, column=5):
    Direct = np.array(Direction)
    h_0 = Net.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _ = Net(h_0, Batch_Seq)
    seq_length = len(Batch_Seq[0])
    geometry = geometry.detach().numpy()
    geometry = geometry[-Net.t_retrieve - Net.t_delay: -Net.t_retrieve + Net.t_cue + seq_length * Net.t_ron + (
                seq_length - 1) * Net.t_roff]
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(seq_length):
        c[Net.t_delay + Net.t_cue + i * (Net.t_ron + Net.t_roff):Net.t_delay + Net.t_cue + i * (
                    Net.t_ron + Net.t_roff) + Net.t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(Batch_Seq)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:, 0], Direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        # plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm, marker='-')
        plt.text(-2, -2, str(Batch_Seq[k]), fontsize=20)

