#! /usr/bin/env python3
#-*- coding: utf-8 -*-
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
import os
import math
from gene_seq import *
from seq_train import *

default_Vector = ([[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                   [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                   [math.cos(0), math.sin(0)],
                   [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                   [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                   [math.cos(math.pi), math.sin(math.pi)]
                   ])
default_colorset = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown'])


# save and load param dictionaries(if not needed in fast reading)
def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def pf(a1, a2): plt.figure(figsize=(a1, a2))


def tn(x): return x.cpu().detach().numpy()


def tt(x, dtype=torch.float, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)

#transform any data type to torch.tensor
def a2t(x):
    if type(x)==list:
        return torch.tensor(x)
    elif type(x)==np.ndarray:
        return torch.from_numpy(x)
    elif type(x)==torch.Tensor:
        return x.detach()
    else:
        print("x is not any known type!")
#transform any data type to numpy.ndarray
def a2n(x):
    if type(x)==list:
        return np.array(x)
    elif type(x)==np.ndarray:
        return x
    elif type(x)==torch.Tensor:
        return x.detach().numpy()
    else:
        print("x is not any known type!")

def w_dict(filepath, dict):
    f = open(filepath, 'w')
    f.write(str(dict))
    f.close()


# saving different types of files
def save(object, filepath, filename, type):
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


# Any point close to target point with distance<criteria is thought to be correst
def accuracy_matrix(out, Batch_Seq, P, Vector=default_Vector, criteria=0.15):
    # out (n_trial,T,2) Batch_Seq(n_trial,L_seq)
    L_seq = len(Batch_Seq[0])
    # target(Batch_Size,L_seq,2)
    target = torch.tensor([[Vector[i - 1] for i in Seq] for Seq in Batch_Seq])
    # target(Batch_Size,L_seq,t_ron,2)
    target = target.unsqueeze(dim=2).repeat(1, 1, P['t_ron'], 1)
    # out(Batch_Size,t_ron,2) to Slides(Batch_Size,L_seq,t_ron,2)
    Slides = torch.cat([out[:, P['r_b' + str(i)]:P['r_e' + str(i)]].unsqueeze(dim=1) for i in range(L_seq)], dim=1)
    # Deviation(Batch_Size,L_seq,t_ron)
    Deviation = torch.sqrt(((Slides - target) ** 2).sum(dim=3))
    Judge = Deviation < criteria
    Judge_ = [[any(item) for item in Seq] for Seq in Judge]
    AM = np.array([[float(item) for item in Seq] for Seq in Judge_])
    return AM


#returns a Closest_Matrix representing the most closest position in each item
#a Correct_Matrix entry of which represents match/mismatch in corresponding position
# def closest_matrix(out, Batch_Seq, P, Vector=default_Vector, criteria=0.15):
#     L_seq = len(Batch_Seq[0])
#     # out(Batch_Size,t_ron,2) to Slides(Batch_Size,L_seq,t_ron,2)
#     Slides = torch.cat([out[:, P['r_b' + str(i)]:P['r_e' + str(i)]].unsqueeze(dim=1) for i in range(L_seq)],
#                        dim=1).detach().numpy()
#     Vec = np.array(Vector.copy())
#     # Dist= (Batch_Size,L_seq,t_ron,6) which is distance from current point to every one of six points
#     Dist = np.sqrt(((np.array([[[t - Vec for t in item] for item in seq] for seq in Slides])) ** 2).sum(axis=4))
#     CM_discrete = Dist < criteria
#     # item with True means around some target point,with criteris<0.5,at most one True exists,if no True,simply readout 0
#     V = np.array([[1, 2, 3, 4, 5, 6]]).transpose((1, 0))
#     # (Batch_Size,L_seq,t_ron)dist<creteria corresponding position,else 0
#     CM_t = CM_discrete @ V
#     CM_t=CM_t.reshape(CM_t.shape[:-1])
#     # CM (Batch_Size,L_seq)
#     Closest_M = np.array([[np.argmax(np.bincount(item)) for item in seq] for seq in CM_t])
#     Target_M = np.array(Batch_Seq)
#     Correct_M=Target_M==Closest_M
#     return Closest_M, Correct_M


def plot_trajectory3(Net, Batch_Seq, row, column=5, colorset=default_colorset):
    P = Net.P
    L_seq = len(Batch_Seq[0])
    Direction = np.array(Net.Direction)
    colorset = list(colorset)
    h_0 = Net.reset_hidden()
    Input = Batch_Seq2Input(Batch_Seq, Net.Vectorrep, Net.Vec_embedding, Net.P)
    _, geometry = Net(h_0, Input)
    seq_length = len(Batch_Seq[0])
    geometry = geometry.detach().numpy()
    t_retrieve = P['t_rrest'] + (L_seq - 1) * P['t_rinterval'] + P['t_ron'] + P['n_windows'] + 10
    dt_t = P['t_on'] + P['t_off']
    dt_r = P['t_rinterval']
    for k in range(len(Batch_Seq)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.scatter(Direction[:, 0], Direction[:, 1], color=colorset, alpha=0.5, s=500, marker='o')
        k_geo = geometry[k]
        plt.plot(k_geo[:, 0], k_geo[:, 1], c='grey')
        for i in range(seq_length):
            plt.scatter(k_geo[P['t_rest'] + i * dt_t:P['t_rest'] + i * dt_t + P['t_on'], 0],
                        k_geo[P['t_rest'] + i * dt_t:P['t_rest'] + i * dt_t + P['t_on'], 1], s=200, marker='x',
                        c=colorset[Batch_Seq[k][i] - 1])
            plt.scatter(k_geo[
                        -t_retrieve + P['t_rrest'] + i * dt_r:-t_retrieve + P['t_rrest'] + i * dt_r + P['t_ron'] + P[
                            'n_windows'] - 1, 0], k_geo[-t_retrieve + P['t_rrest'] + i * dt_r:-t_retrieve + P[
                't_rrest'] + i * dt_r + P['t_ron'] + P['n_windows'] - 1, 1], s=200,
                        marker='.', c=colorset[Batch_Seq[k][i] - 1])
            plt.scatter(-1.3 + 0.2 * i, -1.2, s=200, marker='o', c=colorset[Batch_Seq[k][i] - 1])
        plt.scatter(k_geo[-t_retrieve:-t_retrieve + P['t_cue'], 0], k_geo[-t_retrieve:-t_retrieve + P['t_cue'], 1],
                    s=200, c='black', marker='x')
        plt.scatter(k_geo[-5:, 0], k_geo[-5:, 1], s=200, c='black', marker='.')
        plt.text(-1.4, -1.4, str(Batch_Seq[k]), fontsize=20)
    plt.show()


def plot_trajectory3_(Net, Batch_Seq, column=6, colorset=default_colorset, fig_length=10, delay=None):
    if Batch_Seq == []:
        print('This Batch has no sequences!')
    else:
        P = Net.P
        N_figs = len(Batch_Seq)
        row = math.ceil(N_figs / column)
        plt.figure(figsize=(fig_length * column, fig_length * row))
        L_seq = len(Batch_Seq[0])
        Direction = np.array(Net.Direction)
        colorset = list(colorset)
        h_0 = Net.reset_hidden()
        Input = Batch_Seq2Input(Batch_Seq, Net.Vectorrep, Net.Vec_embedding, Net.P, add_delay=delay)
        _, geometry = Net(h_0, Input)
        seq_length = len(Batch_Seq[0])
        geometry = geometry.detach().numpy()
        t_retrieve = P['t_rrest'] + (L_seq - 1) * P['t_rinterval'] + P['t_ron'] + P['n_windows'] + 10
        dt_t = P['t_on'] + P['t_off']
        dt_r = P['t_rinterval']
        for k in range(len(Batch_Seq)):
            plt.subplot(row, column, k + 1)
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.scatter(Direction[:, 0], Direction[:, 1], color=colorset, alpha=0.5, s=500, marker='o')
            k_geo = geometry[k]
            plt.plot(k_geo[:, 0], k_geo[:, 1], c='grey')
            for i in range(seq_length):
                plt.scatter(k_geo[P['t_rest'] + i * dt_t:P['t_rest'] + i * dt_t + P['t_on'], 0],
                            k_geo[P['t_rest'] + i * dt_t:P['t_rest'] + i * dt_t + P['t_on'], 1], s=200, marker='x',
                            c=colorset[Batch_Seq[k][i] - 1])
                plt.scatter(k_geo[
                            -t_retrieve + P['t_rrest'] + i * dt_r:-t_retrieve + P['t_rrest'] + i * dt_r + P['t_ron'] +
                                                                  P['n_windows'] - 1, 0], k_geo[-t_retrieve + P[
                    't_rrest'] + i * dt_r:-t_retrieve + P['t_rrest'] + i * dt_r + P['t_ron'] + P['n_windows'] - 1, 1],
                            s=200,
                            marker='.', c=colorset[Batch_Seq[k][i] - 1])
                plt.scatter(-1.3 + 0.2 * i, -1.2, s=200, marker='o', c=colorset[Batch_Seq[k][i] - 1])
            plt.scatter(k_geo[-t_retrieve:-t_retrieve + P['t_cue'], 0], k_geo[-t_retrieve:-t_retrieve + P['t_cue'], 1],
                        s=200, c='black', marker='x')
            plt.scatter(k_geo[-5:, 0], k_geo[-5:, 1], s=200, c='black', marker='.')
            plt.text(-1.4, -1.4, str(Batch_Seq[k]), fontsize=20)
        plt.show()


def plot_norm(Net, Batch_Seq):
    I = Batch_Seq2Input(Batch_Seq, Net.Vectorrep, Net.Vec_embedding, Net.P)
    hidden_0 = Net.reset_hidden()
    hidden, geo = Net(hidden_0, I)
    n2_hidden = torch.mean(torch.mean(hidden ** 2, dim=0), dim=1)
    plt.plot(list(range(len(n2_hidden))), n2_hidden.clone().detach().numpy())
    plt.show()


act_color = ['red', 'orange', 'deepskyblue', 'darkblue']


def plot_activity(Net, Batch_Seq, savepath, seq_id=0, N_id=None, x_axis='on', y_axis='on'):
    I = Batch_Seq2Input(Batch_Seq, Net.Vectorrep, Net.Vec_embedding, Net.P)
    hidden_0 = Net.reset_hidden()
    # hidden(Batch_Size,T,N)
    hidden, geo = Net(hidden_0, I)
    y = tn(hidden[seq_id].transpose(0, 1))
    x = list(range(len(y[0])))
    if N_id == None:
        for i in range(Net.N):
            plt.plot(x, y[i], color=random.sample(act_color, 1)[0])
    elif type(N_id) == int:
        Nlist = random.sample(list(range(Net.N)), N_id)
        for i in range(N_id):
            plt.plot(x, y[Nlist[i]], color=random.sample(act_color, 1)[0])
    else:
        for id in N_id:
            plt.plot(x, y[id], color=random.sample(act_color, 1)[0])
    if x_axis == 'off':
        plt.xticks([])
    if y_axis == 'off':
        plt.yticks([])
    plt.savefig(savepath + '.pdf', bbox_inches='tight')
    plt.show()


def closest_id(Net, Batch_Seq):
    I = Batch_Seq2Input(Batch_Seq, Net.Vectorrep, Net.Vec_embedding, Net.P)
    hidden_0 = Net.reset_hidden()
    hidden, geo = Net(hidden_0, I)
    L_seq = len(Batch_Seq[0])
    Direction = Net.Direction
    geo_r = geo[:, -Net.P['t_retrieve']:]
    # 将每个seq上item重复t_retrieve次，然后转换成二维坐标(Batch_Size,L_seq*t_retrieve,2)
    target = torch.tensor(Batch_Seq).reshape(-1, 1).repeat(1, geo_r.shape[1]).reshape(-1, L_seq * geo_r.shape[1])
    target = torch.tensor([[Direction[i - 1] for i in Seq] for Seq in target])
    # 每个rank的readout距离(Rank,Batch_Size,t_retrieve)
    D_geo = torch.tensor([])
    # 距离最近的位置(rank,Batch_Size)
    Index = []
    for i in range(L_seq):
        target_i = target[:, i * Net.P['t_retrieve']:(i + 1) * Net.P['t_retrieve']]
        # (Batch_Size,t_retrieve,2)-(Batch_Size)
        d_geo = torch.sqrt(((geo_r - target_i) ** 2).sum(dim=2))
        D_geo = torch.cat((D_geo, d_geo.unsqueeze(dim=0)), dim=0)
        d_geo_l = d_geo.clone().detach().numpy().tolist()
        index = [seq.index(min(seq)) for seq in d_geo_l]
        Index.append(index)
    return Index, D_geo


colorset = ['blue', 'orange', 'green']


def plot_closest_id(Net, Batch_Seq):
    # (rank,Batch_Size)&(Rank,Batch_Size,t_retrieve)
    Index, D_geo = closest_id(Net, Batch_Seq)
    L_seq = len(Index);
    Batch_Size = len(Index[0])
    m_Index = [math.floor(np.mean(L)) for L in np.array(Index)]
    d_Index = [[Index[i][j] - m_Index[i] for j in range(Batch_Size)] for i in range(L_seq)]
    totalwidth = 0.5
    gmin = min(min(d_Index))
    gmax = max(max(d_Index))
    plt.xlim(gmin - 1, gmax + 1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(L_seq):
        d_Index_i = d_Index[i]
        minimum = min(d_Index_i);
        maximum = max(d_Index_i)
        x = list(range(minimum, maximum + 1))
        countlist = []
        for xj in x:
            countlist.append(d_Index_i.count(xj))
        x = [xj + -0.5 * totalwidth + (i + 0.5) * totalwidth / L_seq for xj in x]
        ax1.bar(x, countlist, width=totalwidth / L_seq, label='rank' + str(i + 1) + ' ref:' + str(m_Index[i]),
                color=colorset[i])
        distance = (
            torch.mean(D_geo[i, :, m_Index[i] + gmin - 1:m_Index[i] + gmax + 2] ** 2, dim=0)).clone().detach().numpy()
        ax2.plot(list(range(gmin - 1, gmax + 2)), distance, color=colorset[i], marker='.')
    plt.title('Timing of each rank')
    ax1.xlabel('rank')
    ax1.set_ylabel('count')
    ax2.set_ylabel('distance')
    ax1.legend()
    plt.show()


def plot_loss(filepath, n, label):
    data = np.array([])
    for i in range(n):
        loss = np.load(filepath + '//loss_' + str(i) + '.npz', allow_pickle=True)[label]
        data = np.concatenate((data, loss))
    plt.plot(list(range(len(data))), data)
    plt.show()


def plot_test_loss(filepath, n, testset, delay=None):
    Loss_p = []
    for i in range(n):
        model = torch.load(filepath + '//model_' + str(i) + '.pth', map_location=torch.device('cpu'))
        hidden_0 = model.reset_hidden()
        Input = Batch_Seq2Input(testset, model.Vectorrep, model.Vec_embedding, model.P, add_delay=delay)
        hidden, geo = model(hidden_0, Input)
        L_p = pos_loss(model, geo, testset, model.Direction)
        Loss_p.append(L_p)
    plt.plot(list(range(len(Loss_p))), Loss_p)
    plt.show()
    L = np.array(Loss_p)
    print(np.argsort(L))


def plot_distance():
    pass


if __name__ == '__main__':
    # test = dict(a=1, b=[1, 2])
    # save_obj(test, 'test')
    test = load_obj('test')
    print(test)
