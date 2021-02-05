from rw import *
from gene_seq import *
from network import *
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import math
from functools import *

####################################################################################################
#                                   Define loss functions
#
#
####################################################################################################
# 一般来说考虑三种loss:第一个是regularization(embedding,W,linear readout)
# input:Net,geo,Batch_seq,Vectorrep()

default_Vector = ([[0., 0., 1.],
                   [math.cos(math.pi / 6), math.sin(math.pi / 6), 0.],
                   [math.cos(-math.pi / 6), math.sin(-math.pi / 6), 0.],
                   [math.cos(-math.pi / 2), math.sin(-math.pi / 2), 0.],
                   [math.cos(-math.pi * 5 / 6), math.sin(-math.pi * 5 / 6), 0.],
                   [math.cos(math.pi * 5 / 6), math.sin(math.pi * 5 / 6), 0.],
                   [math.cos(math.pi / 2), math.sin(math.pi / 2), 0.]
                   ])


def reg_loss(Net, w_reg=1, type='L2'):
    Loss = nn.MSELoss(reduction='mean')
    if type == 'L1':
        Loss = nn.L1Loss(reduction='mean')

    def MSE0(T):
        return Loss(T, torch.zeros_like(T))

    L_I = MSE0(Net.In.weight)
    L_O = MSE0(Net.Out.weight)
    L_W = MSE0(Net.W.weight)
    return w_reg * (L_I + L_O) + L_W


# fixation由三部分组成：go cue以前+item中间+t_final
def fixed_loss(P, out, L_seq):
    t_r = -P['t_final'] - L_seq * P['t_retrieve'] - P['t_cue']
    fix1 = out[:, :t_r]
    fix2 = torch.cat(([
        out[:, -P['t_final'] - (L_seq - 1) * P['t_retrieve'] - P['t_fix']:-P['t_final'] - (L_seq - 1) * P['t_retrieve']]
        for i in range(L_seq)]), dim=1)
    fix3 = out[:, -P['t_final']:]
    fix = torch.cat((fix1, fix2, fix3), dim=1)
    Loss = nn.MSELoss(reduction='sum')
    return Loss(fix, torch.zeros_like(fix))


# 固定窗口loss
# Bacth_Seq(Batch_Size,L_seq)
# Slides(Batch_Size,L_seq*t_ron,2)
def pos_loss_NOREVERSE(P, out, Batch_Seq, Vectorrep, device='cpu'):
    L_seq = len(Batch_Seq[0])
    # (Batch_Size,L_seq*t_ron)
    target = torch.tensor(Batch_Seq).reshape(-1, 1).repeat(1, P['t_ron']).reshape(-1, L_seq * P['t_ron']).to(device)
    # (Batch_Size,L_seq*t_ron,2)
    target = torch.tensor([[Vectorrep[i][:2] for i in Seq] for Seq in target]).to(device)
    t_r = -P['t_final'] - L_seq * P['t_retrieve'] + int((P['t_retrieve'] - P['t_fix'] - P['t_ron']) / 2)
    Slides = torch.cat([out[:, t_r + i * P['t_retrieve']:t_r + i * P['t_retrieve'] + P['t_ron']] for i in range(L_seq)],
                       dim=1)
    Loss = nn.MSELoss(reduction='sum')
    return Loss(target, Slides)


def pos_loss_REVERSE():
    pass


def pos_loss(P, out, Batch_Seq, Vectorrep, device='cpu', reverse=False):
    if not reverse:
        return pos_loss_NOREVERSE(P, out, Batch_Seq, Vectorrep, device=device)
    else:
        return pos_loss_REVERSE()


def train_NOREVERSE(Net, optimizer, scheduler, Batch_Seq, Vectorrep, w_reg=1, weight=[1, 1, 1], device='cpu',
                    regtype='L2'):
    optimizer.zero_grad()
    L_seq = len(Batch_Seq)
    Input = Batch_Seq2Input(Batch_Seq, Vectorrep, Net.P, type='reproduct', add_P=None)
    hidden, out = Net(Input, device=device)
    l_r = len(Batch_Seq) * reg_loss(Net, w_reg=w_reg, type=regtype)
    l_f = fixed_loss(Net.P, out, L_seq)
    l_p = pos_loss(Net.P, out, Batch_Seq, Vectorrep, device=device, reverse=False)
    tl = weight[0] * l_r + weight[1] * l_f + weight[2] * l_p
    tl.backward()
    torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return tl.data.item(), l_r.data.item(), l_f.data.item(), l_p.data.item()


def iter_train_NOREVERSE(Net, optimizer, scheduler, trainset, batch_size, repeat_size, Vectorrep, w_reg=1,
                         weight=[1, 1, 1], n_epochs=200, device="cpu", regtype='L2', freq_report=100):
    Etl, El_r, El_f, El_p = [], [], [], []
    for epoch in range(n_epochs):
        mixed_trainset = mixed_batchify(trainset, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Btl, Bl_r, Bl_f, Bl_p = 0, 0, 0, 0
        for Batch_seq in mixed_trainset:
            tl, l_r, l_f, l_p = train_NOREVERSE(Net, optimizer, scheduler, Batch_seq, Vectorrep, w_reg=w_reg,
                                                weight=weight, device=device, regtype=regtype)
            Btl += tl
            Bl_r += l_r
            Bl_f += l_f
            Bl_p += l_p
        Etl.append(Btl)
        El_r.append(Bl_r)
        El_f.append(Bl_f)
        El_p.append(Bl_p)
        if epoch % freq_report == freq_report - 1:
            print('Epoch {}:\nTotal Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
                  .format(str(epoch + 1), str(Btl), str(Bl_r), str(Bl_f), str(Bl_p)))
    return Etl, El_r, El_f, El_p
