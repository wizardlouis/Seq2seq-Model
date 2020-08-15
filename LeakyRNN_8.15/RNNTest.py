
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

Direction=[
    [math.cos(0),math.sin(0)],
    [math.cos(math.pi/3),math.sin(math.pi/3)],
    [math.cos(math.pi*2/3),math.sin(math.pi*2/3)],
    [math.cos(math.pi),math.sin(math.pi)],
    [math.cos(math.pi*4/3),math.sin(math.pi*4/3)],
    [math.cos(math.pi*5/3),math.sin(math.pi*5/3)]
]


parser=argparse.ArgumentParser()
#set sequence parameters

#These are parameters that should be given
parser.add_argument("--out_dir",help="data out put filepath",type=str,default='')
parser.add_argument("--n_epochs",help="number of epochs",type=int,default=600)
parser.add_argument("--burn_in",help="index of epoch that burn in is over",type=int,default=300)



#These are parameters that may not change but can be modulated
parser.add_argument("--n_item",help="number of items",type=int,default=6)
parser.add_argument("--seq_length",help="length of sequence",type=int,default=2)
parser.add_argument("--N_Neuron",help="Number of neurons in the network",type=int,default=100)
parser.add_argument("--batch_size",help="Number of sequence trained in a single batch",type=int,default=4)

parser.add_argument("--t_rest",help="resting time before target on",type=int,default=30)
parser.add_argument("--t_on",help="target on time",type=int,default=15)
parser.add_argument("--t_off",help="target off time",type=int,default=5)
parser.add_argument("--t_ron",help="retrieve on time",type=int,default=5)
parser.add_argument("--t_roff",help="retrieve off time",type=int,default=15)
parser.add_argument("--t_delay",help="delay time",type=int,default=60)
parser.add_argument("--t_retrieve",help="retrieve time",type=int,default=150)
parser.add_argument("--t_cue",help="cue on time",type=int,default=5)
parser.add_argument("--t_ad_min",help="additional delay minimum",type=int,default=10)
parser.add_argument("--t_ad_max",help="additional delay maximum",type=int,default=60)
parser.add_argument("--g_D",help="scaling of noise in seq2Input signal",type=float,default=0.1)
parser.add_argument("--decay",help="decay parameter of leaky neuron",type=float,default=0.1)
parser.add_argument("--learning_rate",help="learning rate of optimizer",type=float,default=0.01)
parser.add_argument("--trainset_ratio",help="ratio of training sequence number compared to total sequences number",type=float,default=0.3333333333334)



args=parser.parse_args()

out_dir=args.out_dir;n_epochs=args.n_epochs;burn_in=args.burn_in

#make and shuffle sequence datasets
n_item=args.n_item;seq_length=args.seq_length;batch_size=args.batch_size
itemlist=list(range(1,n_item+1))
Sequence=fixedLen_seq(itemlist,seq_length)

t_rest=args.t_rest;t_on=args.t_on;t_off=args.t_off;t_ron=args.t_ron;t_roff=args.t_roff
t_delay=args.t_delay;t_retrieve=args.t_retrieve;t_cue=args.t_cue;t_ad_min=args.t_ad_min;t_ad_max=args.t_ad_max
N_Neuron=args.N_Neuron;g_D=args.g_D
learning_rate=args.learning_rate
trainset_ratio=args.trainset_ratio

#training process include several parts: batchify(with task requirements) and embedding, training with different standards
Vec_embedding=nn.Embedding(3,N_Neuron)
W=tt(getJ(N_Neuron,1))
W.requires_grad=True
l_RNN=myRNN(N_Neuron,W,g_D,torch.tanh,Vec_embedding,default_Vector,
                  t_rest,t_on,t_off,t_ron,t_roff,t_delay,t_retrieve,t_cue,
                  decay=0.1,training=True)

l_RNN.set_additional_delay(t_ad_min,t_ad_max)

optimizer=optim.Adam(l_RNN.parameters(),lr=learning_rate)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=10)


#devide sequence into train and test set;train set is not typical train set due to noise added to seq2Input
#trainset in test phase evaluate generalization on trials seen before
# #test set in test phase evaluate generalization on trials haven't seen before
train_num=int(len(Sequence)*trainset_ratio)
print(train_num)
trainset,testset=balanced_select(Sequence,train_num,n_item)
print("trainset:",trainset,'\n')
print("testset:",testset,'\n')


#Iter train return list
Loss,reLoss,delayLoss=Iter_train_burn_pos(l_RNN,optimizer,None,trainset,n_epochs,batch_size,Direction,burn_in=burn_in)

# Loss_t=[]
# for epoch in range(n_epochs):
#     print('fiction=',str(epoch))
#     TrainLoss,_,_=Iter_train(l_RNN,optimizer,scheduler,trainset,1,4)
#     TrainLoss=TrainLoss[0]
#     SeenLoss,_,_=Batch_testNet_Loss(l_RNN,trainset)
#     UnseenLoss,_,_=Batch_testNet_Loss(l_RNN,testset)
#     Loss_t.append([TrainLoss/n_train,SeenLoss/n_train,UnseenLoss/n_test])

Loss=np.array(Loss);reLoss=np.array(reLoss);delayLoss=np.array(delayLoss)

score_train=testNet_distance(l_RNN,trainset,Direction,1)
score_test=testNet_distance(l_RNN,testset,Direction,1)
print('final correct ratio:\ntrainset:{}\ntestset:{}'.format(str(score_train),str(score_test)))

#saving result and models
save(vars(args),out_dir,'args.txt','dictxt')
np.savez(out_dir+'//loss.npz',Loss=Loss,reLoss=reLoss,delayLoss=delayLoss)
np.savez(out_dir+'//seq.npz',train=trainset,test=testset)
torch.save(l_RNN,out_dir+'//model.pth')






