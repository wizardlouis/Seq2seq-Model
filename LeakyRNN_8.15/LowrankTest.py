
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

parser=argparse.ArgumentParser()
#set sequence parameters

#These are parameters that should be given
parser.add_argument("--out_dir",help="data out put filepath",type=str,default='')
parser.add_argument("--n_epochs",help="number of epochs",type=int,default=100)


#These are parameters that may not change but can be modulated
parser.add_argument("--n_item",help="number of items",type=int,default=6)
parser.add_argument("--seq_length",help="length of sequence",type=int,default=2)
parser.add_argument("--N_Neuron",help="Number of neurons in the network",type=int,default=1000)
parser.add_argument("--N_rank",help="Number of ranks in the network",type=int,default=64)

parser.add_argument("--t_rest",help="resting time before target on",type=int,default=30)
parser.add_argument("--t_on",help="target on time",type=int,default=15)
parser.add_argument("--t_off",help="target off time",type=int,default=5)
parser.add_argument("--t_delay",help="delay time",type=int,default=60)
parser.add_argument("--t_retrieve",help="retrieve time",type=int,default=150)
parser.add_argument("--t_cue",help="cue on time",type=int,default=15)
parser.add_argument("--g_C",help="scaling of noise in connection matrix",type=float,default=0.)
parser.add_argument("--g_D",help="scaling of noise in seq2Input signal",type=float,default=0.1)
parser.add_argument("--decay",help="decay parameter of leaky neuron",type=float,default=0.1)
parser.add_argument("--learning_rate",help="learning rate of optimizer",type=float,default=0.01)



args=parser.parse_args()

out_dir=args.out_dir;n_epochs=args.n_epochs

#make and shuffle sequence datasets
n_item=args.n_item;seq_length=args.seq_length
itemlist=list(range(1,n_item+1))
Sequence=fixedLen_seq(itemlist,seq_length)
random.shuffle(Sequence)

t_rest=args.t_rest;t_on=args.t_on;t_off=args.t_off;t_delay=args.t_delay;t_retrieve=args.t_retrieve;t_cue=args.t_cue
N_Neuron=args.N_Neuron;N_rank=args.N_rank;g_C=args.g_C;g_D=args.g_D
unscaled_noise_M=torch.randn(N_Neuron,N_Neuron)
learning_rate=args.learning_rate

#training process include several parts: batchify(with task requirements) and embedding, training with different standards
left_Vector=torch.randn(N_rank,N_Neuron)
right_Vecotr=torch.randn(N_rank,N_Neuron)
Vec_embedding=nn.Embedding(3,N_Neuron)
l_RNN=lowrank_RNN(N_Neuron,N_rank,left_Vector,right_Vecotr,g_C,g_D,torch.tanh,Vec_embedding,default_Vector,
                  t_rest,t_on,t_off,t_delay,t_retrieve,t_cue,
                  Vector_train=True,unscaled_noise_M=unscaled_noise_M,noise_Train=False,decay=0.1)

optimizer=optim.Adam(l_RNN.parameters(),lr=learning_rate)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.2,10)


#devide sequence into train and test set;train set is not typical train set due to noise added to seq2Input
#trainset in test phase evaluate generalization on trials seen before
# #test set in test phase evaluate generalization on trials haven't seen before
n_train=int(len(Sequence)*2/3);n_test=int(len(Sequence)/3)
trainset=Sequence[:n_train];testset=Sequence[-n_test:]

Loss_t=[]
#Iter train return list
for epoch in range(n_epochs):
    print('Step=',str(epoch))
    TrainLoss,_,_=Iter_train(l_RNN,optimizer,scheduler,trainset,1,4)
    TrainLoss=TrainLoss[0]
    SeenLoss,_,_=Batch_testNet_Loss(l_RNN,trainset)
    UnseenLoss,_,_=Batch_testNet_Loss(l_RNN,testset)
    Loss_t.append([TrainLoss/n_train,SeenLoss/n_train,UnseenLoss/n_test])

Loss_t=np.array(Loss_t)

score_train=Batch_testNet_True(l_RNN,trainset)
score_test=Batch_testNet_True(l_RNN,testset)
print('final correct ratio:\ntrainset:{}\ntestset:{}'.format(str(score_train),str(score_test)))

#saving result and models
save(Loss_t,out_dir,'loss.npy','npy')
save(vars(args),out_dir,'args.txt','dictxt')
np.savez(out_dir+'//seq.npz',train=trainset,test=testset)
torch.save(l_RNN,out_dir+'//model.pth')






