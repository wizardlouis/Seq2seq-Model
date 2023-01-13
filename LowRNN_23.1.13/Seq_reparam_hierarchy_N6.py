# -*- codeing = utf-8 -*-
# @time:2022/8/15 下午9:23
# Author:Xuewen Shen
# @File:Seq_reparam_hierarchy.py
# @Software:PyCharm

from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import random
import math
import os
import time
import pandas as pd
from rw import *
from network import Batch_onehot
from network_modulator import *
import Seq_reparam_module as SRM

# max length of sequences
L = 2
# time periods /ms
max_t = SRM.get_max_t(L)
mixed_max_t = [SRM.get_max_t(l) for l in range(1, L + 1)]

n_items = 6
item_angle = [2 * math.pi * i / n_items for i in range(n_items)]
Embedding_2d = [[math.cos(angle), math.sin(angle)] for angle in item_angle]

# training hyperparamaters
device = 'cuda:0'
Batch_Size = 24
mixed_Batch_Size = [12, 36]
# Learning rate is extremely important, for any learning rate higher than 1e-2 fell to converge! 3e-3 is recommended!!!
learning_rate = 3e-3
# loss weight list refers to weight of [positional loss, fixation loss, regularization loss, orthogonal loss]
loss_weight = [1., .3, 1e-6, 4e-3]
clip_gradient = True
g_in = 0.02
g_rec = 0.02
reg_type = 'L1'
fix_all = True

# traininig dataset
trainset = SRM.Gen_Seq_Mix_Length(list(range(n_items)), list(range(1, L + 1)), repeat=True)
mixed_trainset = [SRM.Gen_Seq(list(range(n_items)), l, repeat=True) for l in range(1, L + 1)]

# initialization of populations and ranks:
Embedding = Embedding_2d
n_dim = len(Embedding[0])
N_pop = 2
pop_expand = [3, 18]
N_I = 1 * n_dim
N_R = 3 * n_dim
N_O = L * n_dim
N_F = N_I + 2 * N_R + N_O

# setup gen_I mode
gen_I = 'R_to_I'


def init_n_pop_I(gen_I, pop_expand):
    assert gen_I in ['default', 'repeat', 'R_to_I']
    if gen_I == 'default':
        return sum(pop_expand)
    else:
        return len(pop_expand)


# Initialization of R_to_I if gen_I=='R_to_I'
R_to_I = nn.Parameter(torch.randn(len(pop_expand), N_I, 2 * N_R))
Mask_R_to_I = torch.ones(len(pop_expand), N_I, 2 * N_R)


# Initialization of layers
g = 0.01

w_mu_I = torch.zeros(init_n_pop_I(gen_I, pop_expand), N_I)
w_C_I = torch.zeros(init_n_pop_I(gen_I, pop_expand), N_I, N_F)
w_C_I[:, :, :N_I] = torch.eye(N_I)
w_mu_O = torch.zeros(N_pop, N_O)
w_C_O = torch.zeros(N_pop, N_O, N_F)
w_C_O[:, :, -N_O:] = torch.eye(N_O)
# w_mu_O = g * torch.randn(N_pop, N_O)
# w_C_O = g * torch.randn(N_pop, N_O, N_F)
# w_C_O[:, :, -N_O:] = g * torch.eye(N_O)
out_Amplification = torch.tensor([4., 6.]).repeat_interleave(n_dim, dim=0)


# Initialization of population modulation hierarchy
no_z = False
pop_mod1 = gen_rotation_symmetry_pops([C_3_group], [0, 0, 0, 0, 0, 0, 0, 0])
pop_mod2 = gen_rotation_symmetry_pops([C_3_group, C_6_group], [1, 0, 1, 1, 0, 1, 1, 0])
pop_mod = [pop_mod1, pop_mod2]

# Initialization of R_to_z if network is no_z for L=2
R_to_z = nn.Parameter(torch.randn(N_O, 2 * N_R))
Mask_R_to_z = torch.zeros(N_O, 2 * N_R)
Mask_R_to_z[:2, N_R:N_R + 2] = 1.
Mask_R_to_z[2:4, N_R + 2:N_R + 4] = 1.
Mask_R_to_z[:2, N_R + 4:N_R + 6] = 1.


train_G, train_I, train_R, train_O = True, True, True, True
ortho_type = dict(
    I=train_I,
    U=train_R,
    V=train_R,
    W=train_O,
    IU=train_R,
    IW=train_R,
    corr_only=False,
    d_I=1.,
    d_R=0.95,
    d_O=1.)
Init = dict(
    G_train=train_G, mu_I_train=train_I, mu_R_train=train_R, mu_O_train=train_O, C_I_train=train_I, C_R_train=train_R,
    C_O_train=train_O,
    w_mu_I=w_mu_I, g_mu_R=g, w_mu_O=w_mu_O, w_C_I=w_C_I, g_C_R=g, w_C_O=w_C_O,
    Out_Amplification=out_Amplification,
    R_to_z=R_to_z, Mask_R_to_z=Mask_R_to_z,
    pop_mod=pop_mod
)
# Init = dict(
#     G_train=True, mu_I_train=False, mu_R_train=True, mu_O_train=True, C_I_train=False, C_R_train=True,
#     C_O_train=True,
#     w_mu_I=w_mu_I, g_mu_R=g, g_mu_O=g, w_C_I=w_C_I, g_C_R=g, g_C_O=g,
#     Out_Amplification=out_Amplification,
#     R_to_z=R_to_z, Mask_R_to_z=Mask_R_to_z,
#     pop_mod=pop_mod
# )

# saving and date information
train_sign = [train_G, train_I, train_R, train_O]
train_string = ''
for i in range(len(train_sign)):
    train_string += str(int(train_sign[i]))
ortho_sign = [ortho_type[i] for i in ['I', 'U', 'V', 'W', 'IU', 'IW']]
ortho_string = ''
for i in range(len(ortho_sign)):
    ortho_string += str(int(ortho_sign[i]))
date = '20221101'
savepath = f'Seq2seq_reparameter_sample_sep_N6//date={date}_sep_hierarchy_L{L}N{n_items}_lr={learning_rate}_train={train_string}_genI={gen_I}_noz={no_z}_lateout_softlatefixall{loss_weight[1]}_ortho={ortho_string}_orthoweight={loss_weight[3]}'
if not os.path.exists(savepath):
    os.makedirs(savepath)
inherit = False
inherit_path = None
model_type = 'reparameter'

train_information = dict(device=device, Batch_Size=Batch_Size, learning_rate=learning_rate, loss_weight=loss_weight,
                         clip_gradient=clip_gradient, g_in=g_in, g_rec=g_rec, reg_type=reg_type, ortho_type=ortho_type,
                         fix_all=fix_all, trainset=trainset)
file_information = dict(savepath=savepath, date=date, inherit=inherit, inherit_path=inherit_path, model_type=model_type)
information = dict(
    time_information=SRM.time_information, train_information=train_information, file_information=file_information,
    Init=Init
)
torch.save(information, savepath + '//hyperparameters.pt')

# report&save frequency/epochs
n_epochs = 20000
freq_report = 50
freq_save = 100

if __name__ == '__main__':
    if inherit:
        RNN = torch.load(inherit_path)
        RNN.P['device'] = device
    else:
        # RNN = reparameterlizedRNN_sample(N_pop, N_I, N_R, N_O, 'Tanh', tau=SRM.tau)
        RNN = hierarchy_repar(N_pop, N_I, N_R, N_O, 'Tanh', tau=SRM.tau)
        RNN.set_no_z(no_z)
        RNN.set_R_to_z(R_to_z, Mask_R_to_z)
        RNN.set_pop_mod(pop_mod)
        RNN.mu_I=nn.Parameter(w_mu_I,requires_grad=True)
        RNN.C_I=nn.Parameter(w_C_I,requires_grad=True)
        RNN.set_gen_I(gen_I)
        RNN.set_R_to_I(R_to_I, Mask_R_to_I)
        # detailed setup of reinitialization
        RNN.reinit(**Init)

    # noz
    # RNN.set_no_z()

    # load model to device, initialization of optimizer and scheduler
    RNN = RNN.to(device)
    RNN.reset_noise_loading(device=device)
    optimizer = optim.Adam(RNN.parameters(), lr=learning_rate)
    scheduler = None

    # SRM.reporter_train_reparamter_seq(RNN, trainset, Embedding, optimizer, scheduler, Batch_Size=Batch_Size,
    #                                   max_t=max_t, g_in=g_in, g_rec=g_rec, loss_weight=loss_weight, reg_type=reg_type,
    #                                   ortho_type=ortho_type, fix_all=fix_all, clip_gradient=clip_gradient,
    #                                   device=device, savepath=savepath, n_epochs=n_epochs, freq_report=freq_report,
    #                                   freq_save=freq_save)

    SRM.reporter_train_reparamter_seq_separate(RNN, mixed_trainset, Embedding, optimizer, scheduler,
                                               mixed_Batch_Size=mixed_Batch_Size,
                                               mixed_max_t=mixed_max_t, g_in=g_in, g_rec=g_rec, loss_weight=loss_weight,
                                               reg_type=reg_type,
                                               ortho_type=ortho_type, fix_all=fix_all, clip_gradient=clip_gradient,
                                               device=device, savepath=savepath, n_epochs=n_epochs,
                                               freq_report=freq_report,
                                               freq_save=freq_save)
