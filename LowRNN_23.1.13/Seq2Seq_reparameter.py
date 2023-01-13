# -*- codeing = utf-8 -*-
# @time:2022/5/16 下午4:29
# Author:Xuewen Shen
# @File:Seq2Seq_reparameter.py
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
import argparse
import pandas as pd
from rw import *
from network import Batch_onehot
from Embedding_Dict import *

Cross_Entropy_Loss = nn.CrossEntropyLoss(reduction='sum')
Fixed_Loss = nn.MSELoss(reduction='sum')

Hill_threshold = 0.1
Readout_Amplification = [4., 6., 9.]


# torch.autograd.set_detect_anomaly(True)
def logcosh(x):
    return x.cosh().log()


# Generate mixed time period trials in one batch with random selected sequences
def Mixed_Batch_Input(RNN, Batch_seq, Size, g_in=None, device='cpu'):
    Batch_sample = torch.tensor([random.sample(Batch_seq, 1)[0] for _ in range(Size)], device=device)
    return Batch_sample, RNN.Batch2Input(Batch_sample, sync=False, g_in=g_in, device=device)


def Mixed_Epoch_train_(RNN, optimizer, scheduler, Seq_set, N_Batch, Batch_Size, g_in=None, g_rec=None, device='cpu'):
    P = RNN.P.copy()
    loss_t = np.zeros((4,))
    for batch_idx in range(N_Batch):
        optimizer.zero_grad()
        t_target = torch.tensor([], dtype=torch.long).to(device)
        t_prediction = torch.tensor([]).to(device)
        t_fixed_loss = torch.tensor(0.)
        for i in range(len(Seq_set)):
            # get Batch information
            t0 = time.time()
            Batch = Seq_set[i]
            length = len(Batch[0])
            size = Batch_Size[length - 1]

            # Go through pipeline
            t1 = time.time()
            Batch_seq, Input = Mixed_Batch_Input(RNN, Batch, size, g_in=g_in, device=device)
            t2 = time.time()
            out = RNN(Input, hidden=None, g_rec=g_rec, device=device)
            # amplification of readout
            amp = torch.tensor(Readout_Amplification, device=device).unsqueeze(dim=1).repeat(1, P['n_dim']).reshape(-1)[
                  :out.shape[-1]]
            out = out * amp

            # calculating positional loss, target[Batch_Size,T,length],prediction[Batch_Size,T,length,n_items]
            t4 = time.time()
            t_out = P['t_delay'] - 25
            # t_out_pos = random.randint(0, t_out)
            target = (Batch_seq.clone() - 1).unsqueeze(dim=1).repeat(1, t_out, 1).reshape(-1)
            readout = out[:, -t_out:, :length * P['n_dim']].clone().reshape(out.shape[0],t_out, length, P['n_dim'])

            prediction = torch.einsum('btld,md->btlm', readout, RNN.Embedding.clone().to(device))
            prediction = prediction.reshape(-1, prediction.shape[-1])
            t_target = torch.cat([t_target, target], dim=0)
            t_prediction = torch.cat([t_prediction, prediction], dim=0)

            # calculating fixation loss
            t5 = time.time()
            if length != len(Seq_set):
                # mode1 start from length*(P['t_on'] + P['t_off']
                # mode2 start from -P['t_delay']+25 to give more space to
                fixed_period = out[:, :, length * P['n_dim']:]
                fixed = fixed_period.clone().reshape(-1)
                # fixed_loss=(fixed**2).sum()
                fixed_loss = logcosh(fixed ** 3 / (fixed ** 2 + Hill_threshold ** 2)).sum() / fixed_period.shape[1]
                # get the largest loss in each Batch/readout channel
                fixed_max = fixed_period.clone().transpose(1, 2).reshape(-1, fixed_period.shape[1])
                max_idx = torch.abs(fixed_max).max(dim=1).indices
                fixed_max_from_idx = torch.tensor([fixed_max[i, max_idx[i]] for i in range(len(max_idx))],
                                                  device=device)
                # fixed_loss_max=(fixed_max_from_idx**2).sum()
                fixed_loss_max = logcosh(
                    fixed_max_from_idx ** 3 / (fixed_max_from_idx ** 2 + Hill_threshold ** 2)).sum()
                # add to total fixed_loss
                t_fixed_loss = t_fixed_loss + fixed_loss

            t6 = time.time()
            # print(
            #     f'Batch length={length};time interval=get infromation:{t1 - t0},Gen_Batch:{t2 - t1},forward:{t3 - t2},readout:{t4 - t3},loss_pos:{t5 - t4},loss_fixed:{t6 - t5}')

        loss_position = Cross_Entropy_Loss(t_prediction, t_target)/(P['t_delay']-25)

        t7 = time.time()
        loss_reg = sum(RNN.reg_loss(device=device))
        loss = loss_position + P['w_reg'] * loss_reg + P['w_fix'] * t_fixed_loss
        t8 = time.time()
        loss.backward()
        t9 = time.time()
        torch.nn.utils.clip_grad_norm_(RNN.parameters(), 1.)
        t10 = time.time()
        optimizer.step()
        t11 = time.time()
        if scheduler is not None:
            scheduler.step(loss)
        loss_t += np.array([LOSS.data.item() for LOSS in [loss, loss_position, t_fixed_loss, loss_reg]])
        # print(
        #     f'Backward time;time interval=loss_reg:{t8 - t7},backward:{t9 - t8},clip_grad:{t10 - t9},optim_step:{t11 - t10}')
    return loss_t


# report&save frequency/epochs
freq_report = 5
freq_save = 10

# hyperparameter file and extracted lines
hyperparameter = pd.read_excel('Seq2seq_reparameter_sample//new_hyperparameter_reparameter_sample.xlsx')
index = [24, 25]
for i in range(*index):
    # set hyperparameters:
    param = hyperparameter[i:i + 1].to_dict('list')
    for key in param.keys():
        param[key] = param[key][0]
    if type(param['Batch_Size'])==int:
        Batch_Size=[param['Batch_Size']]
    else:
        Batch_Size = list(param['Batch_Size'].split(';'))
        Batch_Size = [int(i) for i in Batch_Size]
    param['savepath'] = 'Seq2seq_reparameter_sample//' + param['savepath']
    param['Embedding'] = torch.tensor(eval('Emb_' + param['emb'])).float()
    param.update(eval('Emb_'+param['emb_time']))
    for key in ['mu_I_train', 'mu_O_train', 'C_I_train', 'C_O_train']:
        if key in param:
            param[key] = bool(param[key])
    # device
    device = param['device']

    # setting training schedule/epochs training
    n_epochs = param['n_epochs']
    N_Batch = param['N_Batch']

    if param['inherit'] == 0:
        RNN = reparameterlizedRNN_sample(param)
    else:
        RNN = torch.load('Seq2seq_reparameter_sample//' + param['inherit'])
        RNN.P['device'] = device
        RNN.P['savepath']=param['savepath']

    # load model to device
    RNN = RNN.to(device)
    RNN.set_sampling(device=device)
    optimizer = optim.Adam(RNN.parameters(), lr=param['lr'])
    scheduler = None

    # Additional setup for loading networks from pre-trained networks:

    savepath = param['savepath']
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # load Sequence set from load_set
    Seq_set = load_obj(param['seqpath'])
    totalset = Seq_set['totalset']
    trainset = Seq_set['trainset']
    testset = Seq_set['testset']

    # reconstruct trainset into a list
    trainset_ = [value for value in trainset.values()]

    # define starting point of training and report
    save_count = 0
    start0 = time.time()
    f = open(savepath + '//report.txt', 'a')
    f.write('Report of simulation:\n')
    f.close()

    # save initial model
    torch.save(RNN,savepath+'//model_0.pth')
    save_count+=1
    # Training
    for epoch in range(n_epochs):
        start = time.time()
        # Mixed Batch_training
        RNN.set_sampling(device=device)
        Epoch_loss = Mixed_Epoch_train_(RNN, optimizer, scheduler, trainset_, N_Batch, Batch_Size, device=device)

        # report
        if epoch % freq_report == freq_report - 1:
            end = time.time()
            f = open(savepath + '//report.txt', 'a')
            f.write(
                '\nEpoch {}:\nTotal Loss = {}\nPositional Loss={}\nFixed Loss = {}\nRegularization Loss = {}'
                    .format(str(epoch + 1), str(Epoch_loss[0]), str(Epoch_loss[1]), str(Epoch_loss[2]),
                            str(Epoch_loss[3])))
            f.write('\nThis Epoch takes:{} seconds.\nThe whole process takes:{} seconds'.format(str(end - start),
                                                                                                str(end - start0)))
            f.close()
        # save model
        if epoch % freq_save == freq_save - 1:
            torch.save(RNN, savepath + '//model_' + str(save_count) + '.pth')
            save_count += 1
        if not os.path.exists(savepath + '//loss.npz'):
            np.savez(savepath + '//loss.npz', name=np.array(['Loss', 'Loss_p', 'Loss_f', 'Loss_r']),
                     data=np.array([Epoch_loss]))
        else:
            Lossfile = np.load(savepath + '//loss.npz')
            np.savez(savepath + '//loss.npz', name=Lossfile['name'],
                     data=np.concatenate((Lossfile['data'], np.array([Epoch_loss])), axis=0))

    end0 = time.time()
    # print('Training finished in {} seconds!!!'.format(str(end0-start0)))
