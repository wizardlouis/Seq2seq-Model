# -*- codeing = utf-8 -*-
# @time:2021/11/3 下午11:38
# Author:Xuewen Shen
# @File:Seq2Seq.py
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


# Generate mixed time period trials in one batch
def Mixed_Batch_Input(RNN, Seq_set, Size, device='cpu', add_param=None, t_upb=100):
    '''
    :param RNN: RNN model
    :param Seq_set: candidate sequence set
    :param Size: sampling number
    :param device: device
    :param add_param: additional parameters
    :param t_upb: uppper bound of trial times
    :return: *[timing of each trial's end, Sampling Batch of Sequence, Batch Input signals]
    '''
    Batch_T = []
    Batch_seq = torch.tensor([], dtype=torch.long).to(device)
    Batch_Input = torch.zeros(Size, t_upb, RNN.P['in_Channel'], dtype=torch.float).to(device)
    # random chosen Sequence for trial generation
    for idx in range(Size):
        # Batch(1,seq_length)
        Batch = torch.tensor(random.sample(Seq_set, 1)).to(device)
        Batch_seq = torch.cat((Batch_seq, Batch), dim=0)
        Input = RNN.Batch2Input(Batch, add_param=add_param, device=device)
        T = Input.shape[1]
        Batch_T.append(T)
        Batch_Input[idx:idx + 1, :T] = Input
    return Batch_T, Batch_seq, Batch_Input


def Mixed_Epoch_train(RNN, optimizer, scheduler, Seq_set, N_Batch, Batch_Size, device='cpu', add_param=None, w_reg=0.,
                      t_upb=[80, 100]):
    re_seq = np.array([0., 0., 0.])
    acc = dict(l1r1=0, l2r1=0, l2r2=0, tl1r1=0, tl2r1=0, tl2r2=0)
    for bacth_idx in range(N_Batch):
        optimizer.zero_grad()
        EmbT = RNN.Embedding.T.float().to(device)
        t_pred = torch.tensor([]).to(device)
        t_target = torch.tensor([], dtype=torch.long).to(device)
        for i in range(len(Seq_set)):
            Batch = Seq_set[i]
            l = len(Batch[0])
            size = Batch_Size[l - 1]
            Batch_T, Batch_seq, Batch_Input = Mixed_Batch_Input(RNN, Batch, size, device=device, add_param=add_param,
                                                                t_upb=t_upb[l - 1])
            _, out = RNN(Batch_Input, Batch_T=Batch_T, decoder_steps=l, device=device)
            # hidden_N = RNN.encode(Batch_Input, device=device)
            # # take Batch_T[i]+1 th step of hidden as last hidden state
            # last_hidden_N = torch.cat([hidden_N[i:i + 1, Batch_T[i]] for i in range(size)], dim=0).to(device)
            # _, out = RNN.decode(last_hidden_N, device=device, decoder_steps=l)
            target = Batch_seq - 1
            pred = (out @ EmbT)[:, 1:l + 1, :]
            pred_onehot = pred.argmax(dim=2)
            acc_M = (pred_onehot == target)
            for k in range(l):
                acc['l' + str(l) + 'r' + str(k + 1)] += acc_M[:, k].sum()
                acc['tl' + str(l) + 'r' + str(k + 1)] += acc_M.shape[0]
            t_target = torch.cat((t_target, target.reshape(-1)), dim=0)
            t_pred = torch.cat((t_pred, pred.reshape(-1, pred.shape[-1])), dim=0)
        CEL = nn.CrossEntropyLoss(reduction='sum')
        main_loss = CEL(t_pred, t_target)
        reg_loss = 0.
        for param in RNN.parameters():
            reg_loss += torch.sum(param ** 2)
        tot_loss = main_loss + w_reg * reg_loss
        tot_loss.backward()
        # torch.nn.utils.clip_grad_norm_(RNN.parameters(), 1.)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(tot_loss)
        re_seq += [loss.data.item() for loss in [tot_loss, main_loss, reg_loss]]
    return re_seq, [acc[r] / acc['t' + r] for r in ['l1r1', 'l2r1', 'l2r2']]


# constraint of extra rank to 0 readout
def Mixed_Epoch_train_(RNN, optimizer, scheduler, Seq_set, N_Batch, Batch_Size, device='cpu', add_param=None, w_reg=0.,
                       w_fix=0., w_act=0., t_upb=[80, 100], decoder_steps=2):
    re_seq = np.array([0., 0., 0., 0., 0.])
    acc = dict()
    ranklist = []
    for i in range(1, decoder_steps + 1):
        for j in range(1, i + 1):
            acc['l{}r{}'.format(str(i), str(j))] = 0.
            acc['tl{}r{}'.format(str(i), str(j))] = 0.
            ranklist.append('l{}r{}'.format(str(i), str(j)))

    FXL = nn.MSELoss(reduction='sum')
    ACL = nn.MSELoss(reduction='sum')
    for bacth_idx in range(N_Batch):
        optimizer.zero_grad()
        EmbT = RNN.Embedding.T.float().to(device)
        t_fix = torch.tensor([]).to(device)
        t_act = torch.tensor([]).to(device)
        t_pred = torch.tensor([]).to(device)
        t_target = torch.tensor([], dtype=torch.long).to(device)
        for i in range(len(Seq_set)):
            Batch = Seq_set[i]
            l = len(Batch[0])
            size = Batch_Size[l - 1]
            Batch_T, Batch_seq, Batch_Input = Mixed_Batch_Input(RNN, Batch, size, device=device, add_param=add_param,
                                                                t_upb=t_upb[l - 1])
            # Go through pipeline
            Input = RNN.In(Batch_Input * RNN.in_strength).to(device)
            Encoder_hidden = RNN.Encoder(Input, device=device)
            last_hidden = torch.cat([Encoder_hidden[i:i + 1, Batch_T[i]] for i in range(Input.shape[0])], dim=0).to(
                device)
            if RNN.P['N_Encoder'] != RNN.P['N_Decoder']:
                last_hidden = RNN.W(RNN.act_func(last_hidden))
            Decoder_hidden = RNN.Decoder(n_steps=decoder_steps, hidden_0=last_hidden, device=device)
            out = RNN.out_strength * RNN.Out(Decoder_hidden).to(device)

            # _, out = RNN(Batch_Input, Batch_T, decoder_steps=decoder_steps, device=device)

            # hidden_N = RNN.encode(Batch_Input, device=device)
            # # take Batch_T[i]+1 th step of hidden as last hidden state
            # last_hidden_N = torch.cat([hidden_N[i:i + 1, Batch_T[i]] for i in range(size)], dim=0).to(device)
            # _, out = RNN.decode(last_hidden_N, device=device, decoder_steps=l)
            fixed = out[:, l + 1:].reshape(-1, out.shape[-1])
            act=Encoder_hidden.reshape(-1,Encoder_hidden.shape[-1])
            target = Batch_seq - 1
            pred = (out @ EmbT)[:, 1:l + 1]
            pred_onehot = pred.argmax(dim=2)
            acc_M = (pred_onehot == target)
            for k in range(l):
                acc['l' + str(l) + 'r' + str(k + 1)] += acc_M[:, k].sum().data.item()
                acc['tl' + str(l) + 'r' + str(k + 1)] += acc_M.shape[0]
            t_fix=torch.cat((t_fix,fixed),dim=0)
            t_act=torch.cat((t_act,act),dim=0)
            t_target = torch.cat((t_target, target.reshape(-1)), dim=0)
            t_pred = torch.cat((t_pred, pred.reshape(-1, pred.shape[-1])), dim=0)

        fixed_loss=FXL(t_fix,torch.zeros_like(t_fix))
        act_loss=ACL(t_act,torch.zeros_like(t_act))
        CEL = nn.CrossEntropyLoss(reduction='sum')
        main_loss = CEL(t_pred, t_target)
        reg_loss = 0.
        for param in RNN.parameters():
            if param.requires_grad:
                reg_loss += torch.sum(param ** 2)
        tot_loss = main_loss + w_fix * fixed_loss + w_reg * reg_loss + w_act * act_loss
        tot_loss.backward()
        # torch.nn.utils.clip_grad_norm_(RNN.parameters(), 1.)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(tot_loss)
        re_seq += [loss.data.item() for loss in [tot_loss, fixed_loss, main_loss, reg_loss, act_loss]]
    return re_seq, [(acc[r] / acc['t' + r]) for r in ranklist]


# report&save frequency/epochs
freq_report = 5
freq_save = 10

# device
device = 'cuda:1'

# hyperparameter file and extracted lines
hyperparameter = pd.read_excel('Seq2seq//new_hyperparameter.xlsx')
index = [62, 63]

# epochs training
n_epochs = 1200
N_Batch = 100

# training rate and regularization weight
# upper bound of trial length for length 1,2,3...
t_upb = [198, 220, 242]

# layer initialization for input/output require_grad
require_grad = [False, False]
weight = [1., 1.]

# after new_r5_W,set gain of U,V
gain = [1., 1.]

for i in range(*index):
    # set hyperparameters:
    param = hyperparameter[i:i + 1].to_dict('list')
    for key in param.keys():
        param[key] = param[key][0]
    Batch_Size = list(param['Batch_Size'].split(';'))
    Batch_Size = [int(i) for i in Batch_Size]
    param['savepath'] = 'Seq2seq//' + param['savepath']
    param['Embedding'] = torch.tensor(eval('Emb_' + param['emb'])).float()
    param.update(dict(require_grad=require_grad, weight=weight,
                      gain=gain,Wb=False))

    RNN = Seq2SeqModel(param)
    # 这里要修改
    RNN.reinit(g_En=param['g_En'], g_De=param['g_De'], g_W=param['g_W'])
    # W zeros
    RNN = RNN.to(device)
    optimizer = optim.Adam(RNN.parameters(), lr=param['lr'])
    scheduler = None

    ##Additional setup for loading networks from pre-trained networks:
    # pretrained = torch.load('Seq2seq//n2_r2_re0//model_99.pth', map_location=device)
    # RNN.W.weight.data = pretrained.W.weight.data
    # RNN.W.bias.data = pretrained.W.bias.data
    # RNN.W.weight.requires_grad = False
    # RNN.W.bias.requires_grad = False
    # RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0_long-delay25-125-1//model_99.pth', map_location=device)
    # RNN.W.weight.data = pretrained.W.weight.data
    # RNN.W.weight.requires_grad = False
    # RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad = False
    # RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    # RNN.Encoder.J.weight.requires_grad=True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    # n_clip=4
    # SVD=load_obj('l3n2d1_20220419//model_99_long-delay25-175-1//model_svd')
    # U,S,V=tt(SVD['J']['U'][:,:n_clip]),tt(SVD['J']['S'][:n_clip]),tt(SVD['J']['V'][:,:n_clip])
    # U_=U@torch.sqrt(torch.diag(S))
    # V_=V@torch.sqrt(torch.diag(S))
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0_long-delay25-175-1//model_99.pth', map_location=device)
    # RNN.W.weight.data = pretrained.W.weight.data
    # RNN.W.weight.requires_grad = False
    # RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad = False
    # RNN.Encoder.U.weight.data=U_.to(device)
    # RNN.Encoder.U.weight.requires_grad=True
    # RNN.Encoder.V.weight.data=V_.T.to(device)
    # RNN.Encoder.V.weight.requires_grad=True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    # [54,55]
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0//model_99.pth', map_location=device)
    # RNN.W.weight.data=pretrained.W.weight.data
    # RNN.W.weight.requires_grad=True
    # RNN.Decoder.Q.weight.data=pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad=True
    # RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    # RNN.Encoder.J.weight.requires_grad=True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    # [55，56]model_61 [59,60]model_99
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0-delay25-175-1//model_99.pth', map_location=device)
    # RNN.W.weight.data=pretrained.W.weight.data
    # RNN.W.weight.requires_grad=True
    # RNN.Decoder.Q.weight.data=pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad=True
    # RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    # RNN.Encoder.J.weight.requires_grad=True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    #[56，57]
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0-delay25-125-1//model_99.pth', map_location=device)
    # U_Q,S_Q,V_Q=torch.svd(pretrained.Decoder.Q.weight.data)
    # RNN.Decoder.Q.weight.data=U_Q[:,:3]@torch.diag(S_Q[:3])@(V_Q[:,:3].T)
    # RNN.Decoder.Q.weight.requires_grad=False
    # U_W,S_W,V_W=torch.svd(pretrained.W.weight.data)
    # RNN.W.weight.data=V_Q[:,[1,2,0]]@torch.diag(S_W[:3])@(V_W[:,:3].T)
    # RNN.W.weight.requires_grad = False
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    #[57,58]
    # SVD=load_obj('l3n2d1_20220419//model_99//model_svd')
    # RNN.Decoder.Q.weight.data=tt(SVD['Q']['U'][:,:3]@np.diag(SVD['Q']['S'][:3])@(SVD['Q']['V'][:,:3].T),device=device)
    # RNN.Decoder.Q.weight.requires_grad=False
    # RNN.W.weight.data=tt(SVD['Q']['V'][:,[1,2,0]]@np.diag(SVD['W']['S'][:3])@(SVD['W']['V'][:,:3].T),device=device)
    # RNN.W.weight.requires_grad = False
    # RNN.Out.weight.data=tt(SVD['Out']['U']@np.diag(SVD['Out']['S'])@(SVD['Out']['V'].T),device=device)
    # RNN.Out.weight.requires_grad=False

    #[60,61] from 25-175-2 and freeze decoder, get rid of W population structure
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0-delay25-175-2//model_99.pth', map_location=device)
    # U_W,S_W,_=torch.svd(pretrained.W.weight.data)
    # V_W_alter=torch.randn(4096,6).to(device)/64
    # RNN.W.weight.data=U_W[:,:6]@torch.diag(S_W[:6])@V_W_alter.T
    # RNN.W.weight.requires_grad = False
    # RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad = False
    # RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    # RNN.Encoder.J.weight.requires_grad = True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    #[61,62]
    # pretrained = torch.load('Seq2seq//l3n2_rf_init-0-delay25-175-freeze-1//model_99.pth', map_location=device)
    # RNN.W.weight.data=pretrained.W.weight.data
    # RNN.W.weight.requires_grad = False
    # RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    # RNN.Decoder.Q.weight.requires_grad = False
    # RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    # RNN.Encoder.J.weight.requires_grad = True
    # RNN.In.weight.data = pretrained.In.weight.data
    # RNN.In.weight.requires_grad = False
    # RNN.Out.weight.data = pretrained.Out.weight.data
    # RNN.Out.weight.requires_grad = False

    # [62,63] nw-no whitening 0-54，55-79，80-199
    pretrained = torch.load('Seq2seq//l3n2_rf_init-0-delay25-175-freeze-1-nw//model_79.pth', map_location=device)
    RNN.W.weight.data = pretrained.W.weight.data
    RNN.W.weight.requires_grad = False
    RNN.Decoder.Q.weight.data = pretrained.Decoder.Q.weight.data
    RNN.Decoder.Q.weight.requires_grad = False
    RNN.Encoder.J.weight.data = pretrained.Encoder.J.weight.data
    RNN.Encoder.J.weight.requires_grad = True
    RNN.In.weight.data = pretrained.In.weight.data
    RNN.In.weight.requires_grad = False
    RNN.Out.weight.data = pretrained.Out.weight.data
    RNN.Out.weight.requires_grad = False

    savepath = param['savepath']
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # load Sequence set from load_set
    Seq_set = load_obj(param['seqpath'])
    totalset = Seq_set['totalset']
    trainset = Seq_set['trainset']
    testset = Seq_set['testset']

    # batchify mixed sequence trainset and training
    # batch_size = (6, 30)
    # repeat_size = (1, 1)
    trainset_ = [value for value in trainset.values()]

    # Training
    # name of length/rank, save accuracy file
    Acclist = []
    for i in range(1, param['decoder_steps'] + 1):
        for j in range(1, i + 1):
            Acclist.append('l{}r{}'.format(str(i), str(j)))

    save_count = 80
    start0 = time.time()
    f = open(savepath + '//report.txt', 'a')
    f.write('Report of simulation:\n')
    f.close()


    for epoch in range(n_epochs):
        start = time.time()
        # Mixed Batch_training
        Epoch_loss, Epoch_Accuracy = Mixed_Epoch_train_(RNN, optimizer, scheduler, trainset_, N_Batch, Batch_Size,
                                                        device=device, w_reg=param['w_reg'], w_fix=param['w_fix'],
                                                        w_act=param['w_act'],
                                                        t_upb=t_upb, decoder_steps=param['decoder_steps'])
        Loss=Epoch_loss
        Acc=Epoch_Accuracy
        # Epoch_loss = np.zeros(3, dtype=float)
        # random.shuffle(trainset_)
        # for Batch in trainset_:
        #     Batch_loss = Mixed_Batch_train(RNN, optimizer, scheduler, Batch, Size_multiplier * len(Batch),
        #                                    device=device, w_reg=w_reg, t_upb=t_upb[len(Batch[0]) - 1])
        #     Epoch_loss += Batch_loss
        # Loss_t[:, epoch] = Epoch_loss

        # code for Batch training
        # random.shuffle(trainset_)
        # Epoch_loss = np.zeros(3, dtype=float)
        # for Batch_seq in trainset_:
        #     Batch_loss = train(RNN, optimizer, scheduler, Batch_seq, device=device, w_reg=w_reg)
        #     Epoch_loss += Batch_loss
        # Loss_t[:, epoch] = Epoch_loss

        # report
        if epoch % freq_report == freq_report - 1:
            end = time.time()
            f = open(savepath + '//report.txt', 'a')
            f.write(
                '\nEpoch {}:\nTotal Loss = {}\nFixed Loss={}\nMain Loss = {}\nRegularization Loss = {}\nActivity Regularization loss = {}'
                    .format(str(epoch + 1), str(Epoch_loss[0]), str(Epoch_loss[1]), str(Epoch_loss[2]),
                            str(Epoch_loss[3]), str(Epoch_loss[4])))
            f.write('\nThis Epoch takes:{} seconds.\nThe whole process takes:{} seconds'.format(str(end - start),
                                                                                                str(end - start0)))
        # save model
        if epoch % freq_save == freq_save - 1:
            torch.save(RNN, savepath + '//model_' + str(save_count) + '.pth')
            save_count += 1
        if not os.path.exists(savepath+'//loss.npz'):
            np.savez(savepath+'//loss.npz',name=np.array(['Loss','Loss_f','Loss_p','Loss_r','Loss_a']),data=np.array([Loss]))
        else:
            Lossfile=np.load(savepath+'//loss.npz')
            np.savez(savepath+'//loss.npz', name=Lossfile['name'],data=np.concatenate((Lossfile['data'],np.array([Loss])),axis=0))
        if not os.path.exists(savepath+'//acc.npz'):
            np.savez(savepath+'//acc.npz',name=Acclist,data=np.array([Acc]))
        else:
            Accfile = np.load(savepath + '//acc.npz')
            np.savez(savepath + '//acc.npz', name=Accfile['name'], data=np.concatenate((Accfile['data'],np.array([Acc])),axis=0))

    end0 = time.time()
    # print('Training finished in {} seconds!!!'.format(str(end0-start0)))
