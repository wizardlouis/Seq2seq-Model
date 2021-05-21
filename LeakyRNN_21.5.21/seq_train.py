# -*- coding: utf-8 -*-
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


# if training hidden_0 ,then we need hidden_0_loss:
def hidden_0_loss(Net, P, hidden_r, regtype='L2', device='cpu'):
    if regtype == 'L2':
        Loss = nn.MSELoss(reduction='mean')
    elif regtype == 'L1':
        Loss = nn.L1Loss(reduction='mean')

    def Loss0(T):
        return Loss(T, torch.zeros_like(T))

    if Net.P['train_hidden_0']:
        hidden_r_s = hidden_r[:, :P['t_b0']]
        hidden_r0_s = Net.act_func(Net.hidden_0)
        dev_hidden = hidden_r_s - hidden_r0_s
        return Loss0(dev_hidden)
    else:
        return torch.tensor(0).to(device)


# averaged over trials/neurons/T
def activity_reg_loss(hidden_r, regtype='L2'):
    if regtype == 'L2':
        Loss = nn.MSELoss(reduction='mean')
    elif regtype == 'L1':
        Loss = nn.L1Loss(reduction='mean')

    def Loss0(T):
        return Loss(T, torch.zeros_like(T))

    return Loss0(hidden_r)


# In continual learning,regularization loss with reference weights
def reg_loss_W(Net, w_reg=1, regtype='L2', **kwargs):
    kwarg = kwargs.copy()
    # regtype
    if regtype == 'L2':
        Loss = nn.MSELoss(reduction='mean')
    elif regtype == 'L1':
        Loss = nn.L1Loss(reduction='mean')

    if type(Net) == fullRNN:
        L_W = Loss(Net.W.weight, kwarg['W'])
    elif type(Net) == lowRNN:
        L_W = Loss(Net.U.weight, kwarg['U']) + Loss(Net.V.weight, kwarg['V'])
    else:
        L_W = 0
        print('network type does not match any type in regularization loss!!!')
    # W reg only if not train i-o
    if not Net.P['train_io']:
        return L_W
    # add In Out loss if train i-o
    else:
        L_I = Loss(Net.In.weight, kwarg['In'])
        L_O = Loss(Net.Out.weight, kwarg['Out'])
        return w_reg * (L_I + L_O) + L_W


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
        L_W = Loss0(torch.einsum('ij,jk->ik', Net.U.weight, Net.V.weight))
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

    fix1 = out[:, P['fixl_b1']:P['fixl_e1']]
    fix2 = out[:, P['fixl_b2']:P['fixl_e2']]
    ex_fix1 = out[:, P['fixl_b1']:P['fixl_b1'] + 5]
    ex_fix2 = out[:, P['fixl_b2']:P['fixl_b2'] + 5]
    return MSE0(fix1) + MSE0(fix2)+19*(MSE0(ex_fix1)+MSE0(ex_fix2))


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
        target = target.unsqueeze(dim=2).repeat(1, 1, P['t_ron'], 1)

        # out(Batch_Size,t_ron,2) to Slides(Batch_Size,L_seq,t_ron,2)
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


# **kwargs params:input_type,weight,hregtype.aregtype,regtype,
def train(Net, optimizer, scheduler, Batch_Seq, Vectorrep, device='cpu', **kwargs):
    # Manipulate parameters
    kwarg = kwargs.copy()
    default_param = {'input_type': 'reproduct', 'hregtype': 'L1', 'aregtype': 'L1', 'regtype': 'L2', 'strength': 10,
                     'w_reg': 1., 'init_hidden': 'default'}
    for key in default_param.keys():
        if key not in kwarg:
            kwarg[key] = default_param[key]

    if type(Batch_Seq) == np.ndarray:
        Batch_Seq = Batch_Seq.copy().tolist()
    optimizer.zero_grad()
    L_seq = len(Batch_Seq[0])
    # randomly select delay period
    param = p2p(Net.P, L_seq, type=kwarg['input_type'])
    Input = Batch_Seq2Input(Batch_Seq, Vectorrep, param, type=kwarg['input_type'], strength=kwarg['strength'])
    if kwarg['init_hidden'] == 'default':
        hidden, out = Net(Input, device=device)
    elif kwarg['init_hidden'] == 'rand':
        hidden, out = Net(Input, hidden_0=nn.Parameter(torch.randn(Net.N).to(device), requires_grad=False),
                          device=device)
    else:
        print('init_hidden keyword missed or wrong!!!')

    l_h0 = len(Batch_Seq) * hidden_0_loss(Net, param, Net.act_func(hidden), regtype=kwarg['hregtype'], device=device)
    l_ar = len(Batch_Seq) * activity_reg_loss(Net.act_func(hidden), regtype=kwarg['aregtype'])
    l_r = len(Batch_Seq) * reg_loss(Net, w_reg=kwarg['w_reg'], regtype=kwarg['regtype'])
    l_f = fixed_loss(param, out)
    l_p = pos_loss(param, out, Batch_Seq, Vectorrep, device=device, reverse=False)
    weight = kwarg['weight']
    tl = weight[0] * l_h0 + weight[1] * l_ar + weight[2] * l_r + weight[3] * l_f + weight[4] * l_p
    tl.backward()
    torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return [tl.data.item(), l_h0.data.item(), l_ar.data.item(), l_r.data.item(), l_f.data.item(), l_p.data.item()]


def train_w_reg(Net, optimizer, scheduler, Batch_Seq, Vectorrep, device='cpu', **kwargs):
    # Manipulate parameters
    kwarg = kwargs.copy()
    default_param = {'input_type': 'reproduct', 'hregtype': 'L1', 'aregtype': 'L1', 'regtype': 'L2', 'strength': 10,
                     'w_reg': 1., 'init_hidden': 'default'}
    for key in default_param.keys():
        if key not in kwarg:
            kwarg[key] = default_param[key]

    if type(Batch_Seq) == np.ndarray:
        Batch_Seq = Batch_Seq.copy().tolist()
    optimizer.zero_grad()
    L_seq = len(Batch_Seq[0])
    # randomly select delay period
    param = p2p(Net.P, L_seq, type=kwarg['input_type'])
    Input = Batch_Seq2Input(Batch_Seq, Vectorrep, param, type=kwarg['input_type'], strength=kwarg['strength'])
    if kwarg['init_hidden'] == 'default':
        hidden, out = Net(Input, device=device)
    elif kwarg['init_hidden'] == 'rand':
        hidden, out = Net(Input, hidden_0=nn.Parameter(torch.randn(Net.N).to(device), requires_grad=False),
                          device=device)
    else:
        print('init_hidden keyword missed or wrong!!!')

    l_h0 = len(Batch_Seq) * hidden_0_loss(Net, param, Net.act_func(hidden), regtype=kwarg['hregtype'])
    l_ar = len(Batch_Seq) * activity_reg_loss(Net.act_func(hidden), regtype=kwarg['aregtype'])
    l_r = len(Batch_Seq) * reg_loss(Net, w_reg=kwarg['w_reg'], regtype=kwarg['regtype'])
    l_rW = len(Batch_Seq) * reg_loss_W(Net, **kwarg)
    l_f = fixed_loss(param, out)
    l_p = pos_loss(param, out, Batch_Seq, Vectorrep, device=device, reverse=False)
    weight = kwarg['weight']
    tl = weight[0] * l_h0 + weight[1] * l_ar + weight[2] * l_r + weight[3] * l_rW + weight[4] * l_f + weight[5] * l_p
    tl.backward()
    torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return [tl.data.item(), l_h0.data.item(), l_ar.data.item(), l_r.data.item(), l_rW.data.item(), l_f.data.item(),
            l_p.data.item()]

def train_ortho_proj(Net, optimizer, scheduler, Batch_Seq, Vectorrep, device='cpu', **kwargs):
    # Manipulate parameters
    kwarg = kwargs.copy()
    default_param = {'input_type': 'reproduct', 'hregtype': 'L1', 'aregtype': 'L1', 'regtype': 'L2', 'strength': 10,
                     'w_reg': 1., 'init_hidden': 'default'}
    for key in default_param.keys():
        if key not in kwarg:
            kwarg[key] = default_param[key]


    if type(Batch_Seq) == np.ndarray:
        Batch_Seq = Batch_Seq.copy().tolist()
    optimizer.zero_grad()
    L_seq = len(Batch_Seq[0])
    # randomly select delay period
    param = p2p(Net.P, L_seq, type=kwarg['input_type'])
    Input = Batch_Seq2Input(Batch_Seq, Vectorrep, param, type=kwarg['input_type'], strength=kwarg['strength'])
    if kwarg['init_hidden'] == 'default':
        hidden, out = Net(Input, device=device)
    elif kwarg['init_hidden'] == 'rand':
        hidden, out = Net(Input, hidden_0=nn.Parameter(torch.randn(Net.N).to(device), requires_grad=False),
                          device=device)
    else:
        print('init_hidden keyword missed or wrong!!!')

    l_h0 = len(Batch_Seq) * hidden_0_loss(Net, param, Net.act_func(hidden), regtype=kwarg['hregtype'])
    l_ar = len(Batch_Seq) * activity_reg_loss(Net.act_func(hidden), regtype=kwarg['aregtype'])
    l_r = len(Batch_Seq) * reg_loss(Net, w_reg=kwarg['w_reg'], regtype=kwarg['regtype'])
    l_f = fixed_loss(param, out)
    l_p = pos_loss(param, out, Batch_Seq, Vectorrep, device=device, reverse=False)
    weight = kwarg['weight']
    tl = weight[0] * l_h0 + weight[1] * l_ar + weight[2] * l_r + weight[3] * l_f + weight[4] * l_p
    tl.backward()
    #Use projection operator to manipulate gradient
    if 'train_io' not in kwarg.keys():
        print('Training process undefined io training mode!!!')
    else:
        if kwarg['train_io']:
            Zgrad=torch.cat((Net.W.weight.grad,Net.In.weight.grad),dim=1)
            Zgrad_=kwarg['P_Wz']@Zgrad@kwarg['P_z']
            Net.W.weight.grad=Zgrad[:,:Zgrad_.shape[0]]
            Net.In.weight.grad=Zgrad[:,Zgrad_.shape[0]:]
            Net.Out.weight.grad=kwarg['P_y']@Net.Out.weight.grad@kwarg['P_h']
        else:
            Net.W.weight.grad=kwarg['P_Wz']@Net.W.weight.grad@kwarg['P_z']

    torch.nn.utils.clip_grad_norm_(Net.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return [tl.data.item(), l_h0.data.item(), l_ar.data.item(), l_r.data.item(), l_f.data.item(), l_p.data.item()]

# def iter_train_REPRODUCT(Net, optimizer, scheduler, trainset, batch_size, repeat_size, Vectorrep,
#                          input_type='reproduct',
#                          w_reg=1, weight=[1, 1, 1, 1], n_epochs=200, device="cpu", aregtype='L1', regtype='L2',
#                          freq_report=100):
#     Etl, El_ar, El_r, El_f, El_p = [], [], [], [], []
#     for epoch in range(n_epochs):
#         mixed_trainset = mixed_batchify(trainset, batch_size, repeat_size)
#         random.shuffle(mixed_trainset)
#         Btl, Bl_ar, Bl_r, Bl_f, Bl_p = 0, 0, 0, 0, 0
#         for Batch_seq in mixed_trainset:
#             tl, l_ar, l_r, l_f, l_p = train(Net, optimizer, scheduler, Batch_seq, Vectorrep, input_type=input_type,
#                                             w_reg=w_reg,
#                                             weight=weight, device=device, aregtype=aregtype, regtype=regtype)
#             Btl += tl
#             Bl_ar += l_ar
#             Bl_r += l_r
#             Bl_f += l_f
#             Bl_p += l_p
#         Etl.append(Btl)
#         El_ar.append(Bl_ar)
#         El_r.append(Bl_r)
#         El_f.append(Bl_f)
#         El_p.append(Bl_p)
#         if epoch % freq_report == freq_report - 1:
#             print('Epoch {}:\nTotal Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
#                   .format(str(epoch + 1), str(Btl), str(Bl_r), str(Bl_f), str(Bl_p)))
#     return Etl, El_ar, El_r, El_f, El_p
