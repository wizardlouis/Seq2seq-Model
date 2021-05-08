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

default_Vector = ([[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                   [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                   [math.cos(0), math.sin(0)],
                   [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                   [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                   [math.cos(math.pi), math.sin(math.pi)]
                   ])


def activity_reg_loss(hidden_r, type='L2'):
    if type == 'L2':
        Loss = nn.MSELoss(reduction='sum')
    elif type == 'L1':
        Loss = nn.L1Loss(reduction='sum')

    def Loss0(T):
        return Loss(T, torch.zeros_like(T))

    return Loss0(hidden_r)


def reg_loss(Net, w_reg=1, regtype='L2'):
    # regtype
    if regtype == 'L2':
        Loss = nn.MSELoss(reduction='mean')
    elif regtype == 'L1':
        Loss = nn.L1Loss(reduction='mean')

    # 0 loss
    def Loss0(T):
        return Loss(T, torch.zeros_like(T))

    # loss on W
    if type(Net) == fullRNN:
        L_W = Loss0(Net.W.weight)
    elif type(Net) == lowRNN:
        L_W = Loss0(Net.U.weight) + Loss0(Net.V.weight)
    else:
        L_W = 0
        print('network type does not match any type in regularization loss!!!')
    # W reg only if not train i-o
    if not Net.P['train_io']:
        return L_W
    # add In Out loss if train i-o
    else:
        L_I = Loss0(Net.In.weight)
        L_O = Loss0(Net.Out.weight)
        return w_reg * (L_I + L_O) + L_W


# fixation由2部分组成：go cue以前+t_final
def fixed_loss(P, out):
    Loss = nn.MSELoss(reduction='sum')

    def MSE0(T):
        return Loss(T, torch.zeros_like(T))

    fix = out[:, P['fix_b']:P['fix_e']]
    return MSE0(fix)


# 固定窗口loss
# Bacth_Seq(Batch_Size,L_seq)
# Slides(Batch_Size,L_seq*t_ron,2)
def pos_loss_REPRODUCT(P, out, Batch_Seq, Vectorrep, device='cpu'):
    # [[]]is blank sequence,while []means no sequence
    if Batch_Seq == [[]]:
        return 0
    else:
        L_seq = len(Batch_Seq[0])
        # (Batch_Size,L_seq,2)
        target = torch.tensor([[Vectorrep[i - 1] for i in Seq] for Seq in Batch_Seq]).to(device)
        # (Batch_Size,L_seq,t_ron,2)
        target=target.unsqueeze(dim=2).repeat(1,1,P['t_ron'],1)

        #out(Batch_Size,t_ron,2) to Slides(Batch_Size,L_seq,t_ron,2)
        Slides = torch.cat(
            [out[:, P['r_b' + str(i)]:P['r_e' + str(i)]].unsqueeze(dim=1) for i in range(L_seq)],
            dim=1)
        Loss = nn.MSELoss(reduction='sum')
        return Loss(target, Slides)


def pos_loss_REVERSE():
    pass


def pos_loss(P, out, Batch_Seq, Vectorrep, device='cpu', reverse=False):
    if not reverse:
        return pos_loss_REPRODUCT(P, out, Batch_Seq, Vectorrep, device=device)
    else:
        return pos_loss_REVERSE()


def train(Net, optimizer, scheduler, Batch_Seq, Vectorrep, input_type='reproduct', w_reg=1, weight=[1, 1, 1, 1], device='cpu',
          aregtype='L1', regtype='L2', strength=10):
    if type(Batch_Seq)==np.ndarray:
        Batch_Seq=Batch_Seq.copy().tolist()
    optimizer.zero_grad()
    L_seq = len(Batch_Seq[0])
    # randomly select delay period
    param = p2p(Net.P, L_seq, type=input_type)
    Input = Batch_Seq2Input(Batch_Seq, Vectorrep, param, type=input_type, strength=strength)
    hidden, out = Net(Input, device=device)
    l_ar = activity_reg_loss(Net.act_func(hidden), type=aregtype)
    l_r = len(Batch_Seq) * reg_loss(Net, w_reg=w_reg, regtype=regtype)
    l_f = fixed_loss(param, out)
    l_p = pos_loss(param, out, Batch_Seq, Vectorrep, device=device, reverse=False)
    tl = weight[0] * l_ar + weight[1] * l_r + weight[2] * l_f + weight[3] * l_p
    tl.backward()
    torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return tl.data.item(), l_ar.data.item(), l_r.data.item(), l_f.data.item(), l_p.data.item()


def iter_train_REPRODUCT(Net, optimizer, scheduler, trainset, batch_size, repeat_size, Vectorrep, input_type='reproduct',
                         w_reg=1, weight=[1, 1, 1, 1], n_epochs=200, device="cpu", aregtype='L1', regtype='L2',
                         freq_report=100):
    Etl, El_ar, El_r, El_f, El_p = [], [], [], [], []
    for epoch in range(n_epochs):
        mixed_trainset = mixed_batchify(trainset, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Btl, Bl_ar, Bl_r, Bl_f, Bl_p = 0, 0, 0, 0, 0
        for Batch_seq in mixed_trainset:
            tl, l_ar, l_r, l_f, l_p = train(Net, optimizer, scheduler, Batch_seq, Vectorrep, input_type=input_type, w_reg=w_reg,
                                            weight=weight, device=device, aregtype=aregtype, regtype=regtype)
            Btl += tl
            Bl_ar += l_ar
            Bl_r += l_r
            Bl_f += l_f
            Bl_p += l_p
        Etl.append(Btl)
        El_ar.append(Bl_ar)
        El_r.append(Bl_r)
        El_f.append(Bl_f)
        El_p.append(Bl_p)
        if epoch % freq_report == freq_report - 1:
            print('Epoch {}:\nTotal Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
                  .format(str(epoch + 1), str(Btl), str(Bl_r), str(Bl_f), str(Bl_p)))
    return Etl, El_ar, El_r, El_f, El_p
