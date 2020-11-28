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

#*Input:default vector is used to represent sequence in three dimension:(x,y,cue)
#the first component must be Vec rep. of cue signal
default_Vector=[
    [0.,0.,1.],
    [math.cos(0), math.sin(0), 0.],
    [math.cos(math.pi / 3), math.sin(math.pi / 3), 0.],
    [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3), 0.],
    [math.cos(math.pi), math.sin(math.pi), 0],
    [math.cos(math.pi * 4 / 3), math.sin(math.pi * 4 / 3), 0.],
    [math.cos(math.pi * 5 / 3), math.sin(math.pi * 5 / 3), 0.]
]

#*readout:Direction is used to match output coordinates
Direction=[
    [math.cos(0),math.sin(0)],
    [math.cos(math.pi/3),math.sin(math.pi/3)],
    [math.cos(math.pi*2/3),math.sin(math.pi*2/3)],
    [math.cos(math.pi),math.sin(math.pi)],
    [math.cos(math.pi*4/3),math.sin(math.pi*4/3)],
    [math.cos(math.pi*5/3),math.sin(math.pi*5/3)]
]

parser=argparse.ArgumentParser()
parser.add_argument("--load_net",help="load network filepath",type=str,default='')
parser.add_argument("--load_set",help="load sequence training and test set filepath",type=str,default='')
parser.add_argument("--out_dir",help="data out put filepath",type=str,default='')
parser.add_argument("--device",help="device used to train",type=str,default='cpu')

#We can rewrite parameters of network on a given pre-trained network
parser.add_argument("--rewrite_param",help="decide if rewrite parameters of networks on a given network",type=bool,default=False)
parser.add_argument("--N_Neuron",help="number of neurons in the network",type=int,default=128)
parser.add_argument("--t_rest",help="resting time before target on",type=int,default=2)
parser.add_argument("--t_on",help="target on time",type=int,default=4)
parser.add_argument("--t_off",help="target off time",type=int,default=8)
parser.add_argument("--t_vary",help="flexible input timing",type=int,default=0)
parser.add_argument("--t_delay",help="delay time",type=int,default=30)
parser.add_argument("--ad_min", help="additional delay minimum", type=int, default=0)
parser.add_argument("--ad_max", help="additional delay maximum", type=int, default=10)
parser.add_argument("--t_retrieve",help="retrieve time",type=int,default=37)#对len=1/2/3来说分别可以取21/29/37
parser.add_argument("--t_ron",help="retrieve time window duration",type=int,default=3)
parser.add_argument("--t_rinterval",help="retrieve interval between two items",type=int,default=6)
parser.add_argument("--t_cue",help="cue on time",type=int,default=4)
parser.add_argument("--decay",help="decay parameter of leaky neuron",type=float,default=0.1)
parser.add_argument("--g",help="scaling of noise in seq2Input signal",type=float,default=0.1)
parser.add_argument("--learning_rate",help="learning rate of optimizer",type=float,default=0.005)

parser.add_argument("--t_rvary",help="retrieve time variance",type=int,default=5)
parser.add_argument("--t_rrest",help="resting time before retrieve on",type=int,default=8)
parser.add_argument("--batch_size",help="batch_size of each length",type=int,nargs='+')#default=6 6 6#分别有1、3、4组
parser.add_argument("--repeat_size",help="repeat times of each length",type=int,nargs='+')#default=1 1 1

parser.add_argument("--w_reg",help="regularizaion weight",type=float,default=1.)
parser.add_argument("--loss_weight_dir",help="a dictionary of loss weight through epochs",type=str,default='')
parser.add_argument("--loss_weight",help="number of epochs",type=float,nargs='*')#default=1 1 1
parser.add_argument("--n_epochs",help="number of epochs",type=int,default=200)
parser.add_argument("--n_loop",help="number of loops in training",type=int,default=1)
parser.add_argument("--n_windows",help="number of windows added on training process",type=int,default=6)
parser.add_argument("--act_func",help="activation function",type=str,default="Tanh")
args=parser.parse_args()

#load Sequence set from load_set
load_set=args.load_set
Seq_set=np.load(load_set,allow_pickle=True)
totalset=Seq_set['totalset'];trainset=Seq_set['trainset'];testset=Seq_set['testset']

#batchiy mixed sequence trainset
batch_size=tuple(args.batch_size);repeat_size=tuple(args.repeat_size)

#Network construction
load_net=args.load_net
N_Neuron=args.N_Neuron
P={'t_rest':args.t_rest,'t_on':args.t_on,'t_off':args.t_off,'t_vary':args.t_vary,
   't_delay':args.t_delay,'ad_min':args.ad_min,'ad_max':args.ad_max,'t_cue':args.t_cue,'t_rrest':args.t_rrest,
   't_retrieve':args.t_retrieve,'t_ron':args.t_ron,'t_rinterval':args.t_rinterval,'t_rvary':args.t_rvary,
   'decay':args.decay,'g':args.g,'n_windows':args.n_windows}

#if not given pre-trained network,then start from initializing a RNN
act_func=eval('torch.nn.'+args.act_func+'()')
if load_net=='':
    Vec_embedding=nn.Linear(in_features=3,out_features=N_Neuron,bias=False)
    #Vec_embedding.weight=Ve_embedding.weight/100
    W=tt(getJ(N_Neuron,1))
    W.requires_grad=True
    RNN=myRNN(N_Neuron,W,act_func,Vec_embedding,default_Vector,Direction,P,train=True)
else:
    RNN=torch.load(load_net)
    if args.rewrite_param:
        RNN.P=P
        RNN.act_func=act_func
device=args.device
RNN=RNN.to(device)
RNN.Vec_embedding=RNN.Vec_embedding.to(device)
RNN.Geometry=RNN.Geometry.to(device)
RNN.W=RNN.W.to(device)

optimizer=optim.Adam(RNN.parameters(),lr=args.learning_rate)
schedular=None
    #optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,300,500,700],gamma=0.31623)
    #optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=10)
    #optim.lr_scheduler.MultiStepLR(optimizer,milestones=[],gamma=0.31623)


out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
save(vars(args), out_dir, 'args.txt', 'dictxt')

if args.loss_weight_dir !='':
    loss_weight=np.load(args.loss_weight_dir).tolist()
else:
    loss_weight=[list(args.loss_weight)]
#number of periods of different loss weight
T=len(loss_weight)


start=time.time()
if T==1:
    for i in range(args.n_loop):
        print('loop:'+str(i)+'\n')
        Etl, El_r, El_f, El_p = iter_train(RNN, optimizer, schedular, trainset, batch_size, repeat_size, args.w_reg,
                                           weight=loss_weight[0], n_epochs=args.n_epochs,device=device)
        # Saving results and models
        np.savez(out_dir + '//loss_'+str(i)+'.npz', Loss=np.array(Etl), Loss_r=np.array(El_r), Loss_f=np.array(El_f),
                 Loss_p=np.array(El_p))
        torch.save(RNN, out_dir + '//model_'+str(i)+'.pth')
else:
    for i in range(T):
        print('loop:' + str(i) + '\n')
        print('Start training models with loss weight:',loss_weight[i])
        Etl, El_r, El_f, El_p = iter_train(RNN, optimizer, schedular, trainset, batch_size, repeat_size, args.w_reg,
                                           weight=loss_weight[i], n_epochs=args.n_epochs,device=device)
        # Saving results and models
        np.savez(out_dir + '//loss_'+str(i)+'.npz', Loss=np.array(Etl), Loss_r=np.array(El_r), Loss_f=np.array(El_f),
                 Loss_p=np.array(El_p))
        torch.save(RNN, out_dir + '//model_'+str(i)+'.pth')

end=time.time()
print('Training process takes:',str(end-start),' s!!!')







