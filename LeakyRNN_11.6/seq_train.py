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
#一般来说考虑三种loss:第一个是regularization(embedding,W,linear readout)
#input:Net,geo,Batch_seq,Vectorrep()

default_Vector=[
    [0.,0.,1.],
    [math.cos(0), math.sin(0), 0.],
    [math.cos(math.pi / 3), math.sin(math.pi / 3), 0.],
    [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3), 0.],
    [math.cos(math.pi), math.sin(math.pi), 0],
    [math.cos(math.pi * 4 / 3), math.sin(math.pi * 4 / 3), 0.],
    [math.cos(math.pi * 5 / 3), math.sin(math.pi * 5 / 3), 0.]
]

#10.26shift from mean to sum,but L_W is not changed
def regularization_loss(Net,w_reg):
    Loss=nn.MSELoss(reduction='mean')
    L_E=Loss(Net.Vec_embedding.weight,torch.zeros_like(Net.Vec_embedding.weight))
    L_W=Loss(Net.W,torch.zeros_like(Net.W))
    L_R=Loss(Net.Geometry.weight,torch.zeros_like(Net.Geometry.weight))+Loss(Net.Geometry.bias,torch.zeros_like(Net.Geometry.bias))
    return w_reg*(L_E+L_R)+L_W

#geo(Batch_Size,T,2)
def fixed_loss(Net,geo):
    P=Net.P
    geo_d=geo[:,:-P['t_retrieve']]
    geo_cue=geo[:,-P['t_retrieve']:-P['t_retrieve']+P['t_cue']]
    geo_f=geo[:,-5:]
    Loss=nn.MSELoss(reduction='sum')
    return Loss(geo_d,torch.zeros_like(geo_d))/(len(geo[0])-P['t_retrieve'])+Loss(geo_cue,torch.zeros_like(geo_cue))/P['t_cue']+Loss(geo_f,torch.zeros_like(geo_f))/5

#geo(Batch_Size,T,2),Batch_Seq(Batch_Size,L_seq)
#用这种形式得到的一个epoch的loss_maximum=sum(num_batch*L_seq)比如：1*1+3*2+8*3
def pos_loss(Net,geo,Batch_Seq,Direction,n_windows=7):
    P=Net.P;t0=-P['t_retrieve']+P['t_rrest'];dt=P['t_rinterval'];dt_=P['t_ron']+n_windows-1
    L_seq=len(Batch_Seq[0])
    Slides=torch.tensor([],requires_grad=True)
    total_pos_loss=0
    #将每个seq上item重复dt_次，然后转换成二维坐标(Batch_Size,L_seq*dt_,2)
    target=torch.tensor(Batch_Seq).reshape(-1,1).repeat(1,dt_).reshape(-1,L_seq*dt_)
    target=torch.tensor([[Direction[i-1] for i in Seq] for Seq in target])
    #Slides(Batch_Size,L_seq*dt_,2)将与目标比较的函数
    Slides=torch.cat([geo[:,t0+i*dt:t0+i*dt+dt_] for i in range(L_seq)],dim=1)
    #得到每一个block的差别，并且对二维坐标求和，随后按照rank分块(Batch_Size,L_seq,dt_)
    Diff=((Slides-target)**2).sum(dim=2).reshape(-1,L_seq,dt_)
    Window=torch.zeros(dt_,n_windows,requires_grad=False)
    for i in range(n_windows):
        Window[i:i+P['t_ron'],i]=1
    #按照窗口求平均的diff矩阵(Batch_Size,L_seq,n_windows)
    Diff_Window=Diff@Window/P['t_ron']
    L_DW=Diff_Window.clone().detach().numpy().tolist()
    Pmin=[[item.index(min(item)) for item in seq] for seq in L_DW]
    MPmin=torch.zeros_like(Diff_Window,requires_grad=False)
    for i in range(len(Batch_Seq)):
        for j in range(L_seq):
            MPmin[i,j,Pmin[i][j]]=1
    Min_Diff_Window=Diff_Window*MPmin
    # retrieve=geo[:,-P['t_retrieve']:]
    # cue=geo[:,-P['t_retrieve']:t0];sum_cue=torch.sqrt((cue**2).sum(dim=1)).unsqueeze(dim=1);cue_=cue/sum_cue
    # Direct=torch.tensor([Direction[Seq[0]-1] for Seq in Batch_Seq]).unsqueeze(dim=1).repeat((1,P['t_cue'],1))
    # D_loss=1-cue_*Direct
    Loss=nn.MSELoss(reduction='sum')
    return Loss(Min_Diff_Window,torch.zeros_like(Min_Diff_Window))
    #        0.01*L_seq*Loss(retrieve,torch.zeros_like(retrieve))/P['t_retrieve']+\
    #        0.1*Loss(D_loss,torch.zeros_like(D_loss))/P['t_cue']

#Seq(n_batch,Batch_Size,L_seq)
def train(Net,Net_optimizer,scheduler,Batch_seq,w_reg,weight=[1,1,1],n_windows=7):
    Net_optimizer.zero_grad()
    hidden_0=Net.reset_hidden()
    Input=Batch_Seq2Input(Batch_seq,Net.Vectorrep,Net.Vec_embedding,Net.P)
    hidden,geo=Net(hidden_0,Input)
    l_r=len(Batch_seq)*regularization_loss(Net,w_reg)
    l_f=fixed_loss(Net,geo)
    l_p=pos_loss(Net,geo,Batch_seq,Net.Direction,n_windows)
    tl=weight[0]*l_r+weight[1]*l_f+weight[2]*l_p
    tl.backward()
    Net_optimizer.step()
    if scheduler is not None:
        scheduler.step(tl)
    return tl.data.item(),l_r.data.item(),l_f.data.item(),l_p.data.item()

def iter_train(Net,Net_optimizer,scheduler,trainset,batch_size,repeat_size,w_reg,weight=[1,1,1],n_epochs=200,n_windows=7):
    Etl,El_r,El_f,El_p=[],[],[],[]
    for epoch in range(n_epochs):
        mixed_trainset = mixed_batchify(trainset, batch_size, repeat_size)
        random.shuffle(mixed_trainset)
        Btl,Bl_r,Bl_f,Bl_p=0,0,0,0
        for Batch_seq in mixed_trainset:
            tl,l_r,l_f,l_p=train(Net,Net_optimizer,scheduler,Batch_seq,w_reg,weight,n_windows=n_windows)
            Btl+=tl;Bl_r+=l_r;Bl_f+=l_f;Bl_p+=l_p
        Etl.append(Btl);El_r.append(Bl_r);El_f.append(Bl_f);El_p.append(Bl_p)
        print('Epoch {}:\nTotal Loss = {}\nRegularization Loss = {}\nFixed Loss = {}\nPositional Loss = {}'
              .format(str(epoch),str(Btl),str(Bl_r),str(Bl_f),str(Bl_p)))
    return Etl,El_r,El_f,El_p





if __name__=='__main__':
    Vec_embedding=nn.Embedding(3,128)
    W=tt(getJ(128,1))
    W.requires_grad=True
    P={'decay':0.1,'g':0.1,'t_rest':10,'t_on':5,'t_off':10,'t_vary':3,'t_cue':5,'t_ron':5,'t_rinterval':10,'t_retrieve':50,'t_delay':50,'ad_min':10,'ad_max':50}
    RNN=myRNN(128,W,torch.tanh,Vec_embedding,default_Vector,Direction,P,train=True)
    len_1=fixedLen_seq([1,2,3,4,5,6],1)
    Input=Batch_Seq2Input(len_1,default_Vector,Vec_embedding,P)
    print('Input shape=',Input.shape)
    hidden_0=RNN.reset_hidden()
    hidden,geo=RNN(hidden_0,Input)
    print('hidden shape=',hidden.shape)
    print('geometry shape=',geo.shape)
    Loss=pos_loss(RNN,geo,len_1,Direction)
    print(Loss)
