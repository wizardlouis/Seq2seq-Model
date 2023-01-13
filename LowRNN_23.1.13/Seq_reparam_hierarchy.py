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
from Embedding_Dict import *
from network_modulator import *
import Seq_reparam_module as SRM

# max length of sequences
L = 2
# time periods /ms
max_t = SRM.get_max_t(L)
mixed_max_t = [SRM.get_max_t(l) for l in range(1, L + 1)]

n_items = 2
# item_angle = [2 * math.pi * i / n_items for i in range(n_items)]
# Embedding_2d = [[math.cos(angle), math.sin(angle)] for angle in item_angle]
Embedding_1d = [
    [1.],
    [-1.]
]

# training hyperparamaters
device = 'cuda:0'
Batch_Size = 24
mixed_Batch_Size = [8, 16]
#Learning rate is extremely important, for any learning rate higher than 1e-2 fell to converge! 3e-3 is recommended!!!
learning_rate = 3e-3
# loss weight list refers to weight of [positional loss, fixation loss, regularization loss, orthogonal loss]
loss_weight = [1., .3, 1e-6, 4e-3]
# loss_weight = [1., 1e-3, 1e-6, 1e-2]
clip_gradient = True
g_in = 0.02
g_rec = 0.02
reg_type = 'L1'
ortho_type = dict(I=False, U=True, V=True, W=True, IU=True, IW=True, corr_only=False, d_I=1., d_R=0.95, d_O=1.)
fix_all = True

# traininig dataset
trainset = SRM.Gen_Seq_Mix_Length(list(range(n_items)), list(range(1, L + 1)), repeat=True)
mixed_trainset = [SRM.Gen_Seq(list(range(n_items)), l, repeat=True) for l in range(1, L + 1)]

# initialization of populations and ranks:
Embedding = Embedding_1d
n_dim = len(Embedding[0])
N_pop = 2
N_I = 1
N_R = 3
N_O = n_dim * L
N_F = N_I + 2 * N_R + N_O

# Initialization of layers
g = 0.01
w_mu_I = torch.zeros(N_pop, N_I)
w_C_I = torch.zeros(N_pop, N_I, N_F)
w_C_I[:, :, :N_I] = torch.eye(N_I)
w_mu_O = g * torch.randn(N_pop, N_O)
w_C_O = g * torch.randn(N_pop, N_O, N_F)
w_C_O[:, :, -N_O:] = g * torch.eye(N_O)
out_Amplification = torch.tensor([4., 6.]).repeat_interleave(n_dim, dim=0)

Init = dict(
    G_train=True, mu_I_train=False, mu_R_train=True, mu_O_train=True, C_I_train=False, C_R_train=True,
    C_O_train=True,
    w_mu_I=w_mu_I, g_mu_R=g, g_mu_O=g, w_C_I=w_C_I, g_C_R=g, g_C_O=g,
    Out_Amplification=out_Amplification
)

# saving and date information
# savepath = f'Seq2seq_reparameter_sample//L{L}N{n_items}_hierarchy_trainI_ortho_IUV_softlatefixall_W12r2_delayout_noz_1'
savepath = f'Seq2seq_reparameter_sample_sep//L{L}N{n_items}_sep_hierarchy_trainO_ortho_UVW_lateout_softlatedixall_ref_regression_long'
if not os.path.exists(savepath):
    os.makedirs(savepath)
date = '20221015'
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
        RNN = hierarchy_repar_N2_normalI(N_pop, N_I, N_R, N_O, 'Tanh', tau=SRM.tau)
        RNN.set_pop_mod(N2_pop_mod)
        # detailed setup of reinitialization
        RNN.reinit(**Init)

    #noz
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
