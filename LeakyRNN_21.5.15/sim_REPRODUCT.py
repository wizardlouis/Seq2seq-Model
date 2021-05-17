#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from rw import *
from gene_seq import *
from seq_train import *
from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import argparse
import random
import math
import os
import time
from torch import _C

# *Input:default vector is used to represent sequence in three dimension:(x,y,cue)
# the first component must be Vec rep. of cue signal
default_Vector = ([[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                   [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                   [math.cos(0), math.sin(0)],
                   [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                   [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                   [math.cos(math.pi), math.sin(math.pi)]
                   ])

parser = argparse.ArgumentParser()
# Selective CL_mode:'w_reg','ortho_proj',corresponding CL_data is reference model(W,In,Out)--this can be derived from load_net/reference trajectories
parser.add_argument("--CL_mode", help="Which continual learning mode is used", type=str, default='')
parser.add_argument("--CL_data", help="continual learning reference data", type=str, default='')

parser.add_argument("--lowrank", help="if training a low rank network", type=int, default=0)
parser.add_argument("--N_rank", help="Number of rank in default low rank network", type=int, default=25)
parser.add_argument("--load_net", help="load network filepath", type=str, default='')
parser.add_argument("--load_set", help="load sequence training and test set filepath", type=str, default='')
parser.add_argument("--out_dir", help="data output filepath", type=str, default='')
parser.add_argument("--device", help="device used to train", type=str, default='cpu')
# We can rewrite parameters of network on a given pre-trained network
parser.add_argument("--rewrite_param", help="decide if rewrite parameters of networks on a given network", type=int,
                    default=0)

# Network parameters
parser.add_argument("--train_io", help="if training input/output channel parameters", type=int, default=0)
parser.add_argument("--train_hidden_0", help="if training initial hidden state", type=int, default=1)
parser.add_argument("--init_hidden", help="initial hidden in training phase", type=str, default='default')
parser.add_argument("--Input_strength", help="stimulus input strength", type=float, default=10.)
parser.add_argument("--act_func", help="activation function", type=str, default="Tanh")
parser.add_argument("--in_Channel", help="number of input channels", type=int, default=3)
parser.add_argument("--out_Channel", help="number of output channels", type=int, default=2)

parser.add_argument("--dt", help="updating step length/ms", type=float, default=10)
parser.add_argument("--tau", help="time constant of neuron/ms", type=float, default=100)
parser.add_argument("--g_in", help="scaling of noise in seq2Input signal at populational input", type=float,
                    default=0.01)
parser.add_argument("--g_rec", help="scaling of noise in recurrent weight matrix", type=float, default=0.15)

parser.add_argument("--N_Neuron", help="number of neurons in the network", type=int, default=128)
parser.add_argument("--t2fix", help="time period from fixation on to fiaxtion loss on", type=int, default=5)
parser.add_argument("--t_rest", help="time period before stimulus", type=int, default=30)
parser.add_argument("--Dt_rest", help="time period variation before stimulus", type=int, default=30)
parser.add_argument("--t_on", help="Input duration of one component", type=int, default=5)
parser.add_argument('--Dt_on', help='Variability of Input timing', type=int, default=1)
parser.add_argument('--t_item', help='Duration of one component', type=int, default=15)

parser.add_argument('--t_delay', help="delay duration", type=int, default=30)
parser.add_argument('--Dt_delay', help='Variability of delay duration', type=int, default=30)

parser.add_argument("--t_rrest", help="time period before retrieve after fixation off", type=int, default=5)
parser.add_argument("--t_retrieve", help="Retrieve duration of one component", type=int, default=10)
parser.add_argument("--t_ron", help="One time window length of retrieval", type=int, default=5)
parser.add_argument("--t_final", help="resting time period before the end of one trial", type=int, default=30)
parser.add_argument("--Dt_final", help="resting time period variation before the end of one trial", type=int,
                    default=30)

# Learning parameters
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.01)
parser.add_argument("--batch_size", help="batch_size of each length", type=int, nargs='+')  # default=6 6 6#分别有1、3、4组
parser.add_argument("--repeat_size", help="repeat times of each length", type=int, nargs='+')  # default=1 1 1
# input type:choose one from {'reprodcut','reproduct_rcue1','reproduct_rcue2'}
parser.add_argument("--input_type", help="input signal type", type=str, default='reproduct')
parser.add_argument("--w_reg", help="regularizaion weight", type=float, default=1.)
parser.add_argument("--loss_weight", help="loss weight", type=float, nargs='*')  # default=1 1 1 1 1
parser.add_argument("--n_epochs", help="number of epochs", type=int, default=10000)
parser.add_argument("--hregtype", help="hidden_0 training regularization loss type", type=str, default="L1")
parser.add_argument("--aregtype", help="activity regularization loss type", type=str, default="L1")
parser.add_argument("--regtype", help="regularization loss type", type=str, default="L2")

parser.add_argument("--freq_report", help="frequency/epochs of reporting training loss", type=int, default=500)
parser.add_argument("--freq_save", help="frequency/epochs of saving models", type=int, default=1000)
args = parser.parse_args()

# Network construction

P = dict(N_rank=args.N_rank, train_io=bool(args.train_io), train_hidden_0=bool(args.train_hidden_0),
         act_func=args.act_func, in_Channel=args.in_Channel, out_Channel=args.out_Channel,
         dt=args.dt, tau=args.tau, g_in=args.g_in, g_rec=args.g_rec,
         N_Neuron=args.N_Neuron, t2fix=args.t2fix, t_rest=args.t_rest, Dt_rest=args.Dt_rest,
         t_on=args.t_on, Dt_on=args.Dt_on, t_item=args.t_item,
         t_delay=args.t_delay, Dt_delay=args.Dt_delay, t_rrest=args.t_rrest, t_retrieve=args.t_retrieve,
         t_ron=args.t_ron, t_final=args.t_final, Dt_final=args.Dt_final
         )

# if not given pre-trained network,then start from initializing a RNN
if args.load_net == '':
    if bool(args.lowrank):
        rank_reserve = 0
        RNN = lowRNN(P)
        U, V = getUV(P['N_Neuron'], 1, P['N_rank'])
        RNN.U.weight = nn.Parameter(tt(U), requires_grad=True)
        RNN.V.weight = nn.Parameter(tt(V), requires_grad=True)
    else:
        RNN = fullRNN(P)
        W = tt(getJ(P['N_Neuron'], 1))
        RNN.setW(W)
else:
    if bool(args.lowrank):
        # Continual learning of low rank network requires reservation of U,V in RNN_0
        RNN_0 = torch.load(args.load_net, map_location=args.device)
        rank_reserve = RNN_0.P['N_rank']
        RNN = lowRNN(P)
        U, V = getUV(P['N_Neuron'], 1, P['N_rank'] - rank_reserve)
        RNN.U.weight = nn.Parameter(torch.cat((RNN_0.U.weight.data, tt(U).data), dim=1), requires_grad=True)
        RNN.V.weight = nn.Parameter(torch.cat((RNN_0.V.weight.data, tt(V).data), dim=0), requires_grad=True)
    else:
        RNN = torch.load(args.load_net, map_location=args.device)
        if bool(args.rewrite_param):
            RNN.P = P

device = args.device
RNN = RNN.to(device)
optimizer = optim.Adam(RNN.parameters(), lr=args.learning_rate)

scheduler = None
# optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,300,500,700],gamma=0.31623)
# optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=10)
# optim.lr_scheduler.MultiStepLR(optimizer,milestones=[],gamma=0.31623)

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# save args parameters
save(vars(args), out_dir, 'args.txt', 'dictxt')

# load Sequence set from load_set
Seq_set = load_obj(args.load_set)
totalset = Seq_set['totalset']
trainset = Seq_set['trainset']
testset = Seq_set['testset']

# batchify mixed sequence trainset and training
batch_size = tuple(args.batch_size)
repeat_size = tuple(args.repeat_size)

save_count = 0
start0 = time.time()
f = open(args.out_dir + '//report.txt', 'a')
f.write('Report of simluation:\n')
f.close()

trainset_ = [value for value in trainset.values()]
if not bool(args.CL_mode):
    Loss_t = np.zeros((6, int(args.n_epochs)), dtype=float)
    for epoch in range(args.n_epochs):
        start = time.time()
        mixed_trainset = mixed_batchify(trainset_, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Epoch_loss = np.zeros(6, dtype=float)
        for Batch_seq in mixed_trainset:
            # loss :total,h0,ar,Wr,fix,pos
            Batch_loss = train(RNN, optimizer, scheduler, Batch_seq, default_Vector, device=device,
                               input_type=args.input_type,
                               w_reg=args.w_reg, weight=args.loss_weight,
                               hregtype=args.hregtype,
                               aregtype=args.aregtype, regtype=args.regtype,
                               strength=args.Input_strength,
                               init_hidden=args.init_hidden)
            Epoch_loss += Batch_loss
        Loss_t[:, epoch] = Epoch_loss
        if epoch % args.freq_report == args.freq_report - 1:
            end = time.time()
            f = open(args.out_dir + '//report.txt', 'a')
            f.write(
                '\nEpoch {}:\nTotal Loss = {}\nhidden_0 Regularization Loss = {}\n Activity Regularization Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
                    .format(str(epoch + 1), str(Batch_loss[0]), str(Batch_loss[1]), str(Batch_loss[2]),
                            str(Batch_loss[3]), str(Batch_loss[4]), str(Batch_loss[5])))
            f.write('\nThis Epoch takes:{} seconds.\nThe whole process takes:{} seconds'.format(str(end - start),
                                                                                                str(end - start0)))
        if epoch % args.freq_save == args.freq_save - 1:
            torch.save(RNN, out_dir + '//model_' + str(save_count) + '.pth')
            save_count += 1

    # After training,save loss file
    np.savez(out_dir + '//loss.npz', Loss=Loss_t[0], Loss_h0=Loss_t[1], Loss_ar=Loss_t[2],
             Loss_r=Loss_t[3], Loss_f=Loss_t[4], Loss_p=Loss_t[5])
    end0 = time.time()
    print('Training finished in {} seconds!!!'.format(str(end0 - start0)))

elif args.CL_mode == 'w_reg':
    Loss_t = np.zeros((7, int(args.n_epochs)), dtype=float)
    ref_model = torch.load(args.load_net, map_location=args.device)
    if bool(args.lowrank):
        Uref = torch.cat((ref_model.U.weight.detach(), torch.zeros(P['N_Neuron'], P['N_rank'] - ref_model.P['N_rank'])),
                         dim=1)
        Vref = torch.cat((ref_model.V.weight.detach(), torch.zeros(P['N_rank'] - ref_model.P['N-rank'], P['N_neuron'])),
                         dim=0)
        Inref = ref_model.In.weight.detach()
        Outref = ref_model.Out.weight.detach()
        ref_param = {'U': Uref, 'V': Vref, 'In': Inref, 'Out': Outref}
    else:
        Wref = ref_model.W.weight.detach()
        Inref = ref_model.In.weight.detach()
        Outref = ref_model.Out.weight.detach()
        ref_param = {'W': Wref, 'In': Inref, 'Out': Outref}

    for epoch in range(args.n_epochs):
        start = time.time()
        mixed_trainset = mixed_batchify(trainset_, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Epoch_loss = np.zeros(7, dtype=float)
        for Batch_seq in mixed_trainset:
            # loss :total,h0,ar,Wr,fix,pos
            Batch_loss = train_w_reg(RNN, optimizer, scheduler, Batch_seq, default_Vector, device=device,
                                     input_type=args.input_type,
                                     w_reg=args.w_reg, weight=args.loss_weight,
                                     hregtype=args.hregtype,
                                     aregtype=args.aregtype, regtype=args.regtype,
                                     strength=args.Input_strength,
                                     init_hidden=args.init_hidden, **ref_param)
            Epoch_loss += Batch_loss
        Loss_t[:, epoch] = Epoch_loss
        if epoch % args.freq_report == args.freq_report - 1:
            end = time.time()
            f = open(args.out_dir + '//report.txt', 'a')
            f.write(
                '\nEpoch {}:\nTotal Loss = {}\nhidden_0 Regularization Loss = {}\n Activity Regularization Loss = {}\nRegularization Loss = {}\nReference model Regularization Loss={}\nFixed Loss = {}\nPositional Loss = {}'
                    .format(str(epoch + 1), str(Batch_loss[0]), str(Batch_loss[1]), str(Batch_loss[2]),
                            str(Batch_loss[3]), str(Batch_loss[4]), str(Batch_loss[5]), str(Batch_loss[6])))
            f.write('\nThis Epoch takes:{} seconds.\nThe whole process takes:{} seconds'.format(str(end - start),
                                                                                                str(end - start0)))
        if epoch % args.freq_save == args.freq_save - 1:
            torch.save(RNN, out_dir + '//model_' + str(save_count) + '.pth')
            save_count += 1

    # After training,save loss file
    np.savez(out_dir + '//loss.npz', Loss=Loss_t[0], Loss_h0=Loss_t[1], Loss_ar=Loss_t[2],
             Loss_r=Loss_t[3], Loss_rW=Loss_t[4], Loss_f=Loss_t[5], Loss_p=Loss_t[6])
    end0 = time.time()
    print('Training finished in {} seconds!!!'.format(str(end0 - start0)))

elif args.CL_mode == 'ortho_proj':
    #from CL_data load 'In/W/Out_Wz/z'
    data=load_obj(args.CL_mode)
    Proj={}
    for label in ['In','W','Out']:
        for subs in ['_Wz','z']:
            if label+subs in data.keys():
                Proj.update({label+subs:a2t(data[label+subs])})
    Loss_t = np.zeros((6, int(args.n_epochs)), dtype=float)
    for epoch in range(args.n_epochs):
        start = time.time()
        mixed_trainset = mixed_batchify(trainset_, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Epoch_loss = np.zeros(6, dtype=float)
        for Batch_seq in mixed_trainset:
            # loss :total,h0,ar,Wr,fix,pos
            Batch_loss = train(RNN, optimizer, scheduler, Batch_seq, default_Vector, device=device,
                               input_type=args.input_type,
                               w_reg=args.w_reg, weight=args.loss_weight,
                               hregtype=args.hregtype,
                               aregtype=args.aregtype, regtype=args.regtype,
                               strength=args.Input_strength,
                               init_hidden=args.init_hidden,**Proj)
            Epoch_loss += Batch_loss
        Loss_t[:, epoch] = Epoch_loss
        if epoch % args.freq_report == args.freq_report - 1:
            end = time.time()
            f = open(args.out_dir + '//report.txt', 'a')
            f.write(
                '\nEpoch {}:\nTotal Loss = {}\nhidden_0 Regularization Loss = {}\n Activity Regularization Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
                    .format(str(epoch + 1), str(Batch_loss[0]), str(Batch_loss[1]), str(Batch_loss[2]),
                            str(Batch_loss[3]), str(Batch_loss[4]), str(Batch_loss[5])))
            f.write('\nThis Epoch takes:{} seconds.\nThe whole process takes:{} seconds'.format(str(end - start),
                                                                                                str(end - start0)))
        if epoch % args.freq_save == args.freq_save - 1:
            torch.save(RNN, out_dir + '//model_' + str(save_count) + '.pth')
            save_count += 1

    # After training,save loss file
    np.savez(out_dir + '//loss.npz', Loss=Loss_t[0], Loss_h0=Loss_t[1], Loss_ar=Loss_t[2],
             Loss_r=Loss_t[3], Loss_f=Loss_t[4], Loss_p=Loss_t[5])
    end0 = time.time()
    print('Training finished in {} seconds!!!'.format(str(end0 - start0)))
