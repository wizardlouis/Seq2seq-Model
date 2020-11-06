import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rw import *

class myRNN(nn.Module):
    def __init__(self,N_Neuron,W,act_func,Vec_embedding,Vectorrep,Direction,P,train=True):
        super(myRNN,self).__init__()
        self.N=N_Neuron
        self.W=W
        self.P=P
        #P中包含的参数：decay默认为0.1，噪声g默认为0.1，t_rest,t_on,t_off,t_delay,ad_min,ad_max,t_ron,t_rinterval,t_retrieve,t_cue,decay,g
        self.act_func=act_func
        self.Vec_embedding=Vec_embedding;self.Vectorrep=Vectorrep;self.Direction=Direction
        self.Geometry=nn.Linear(in_features=self.N,out_features=2,bias=True)

    def reset_hidden(self):
        return torch.zeros(self.N,requires_grad=True)

    #given a batch of input of size (Batch_Size,T,N_Neuron)
    def forward(self,hidden_0,Batch_Input):
        Batch_Size,T=Batch_Input.shape[0:2]
        #hidden size=(Batch_Size,N)
        hidden=hidden_0.unsqueeze(dim=0).repeat(Batch_Size,1)
        hidden_t=torch.tensor([])
        for frame in range(T):
            hidden=(1-self.P['decay'])*hidden+self.P['decay']*(torch.einsum('ab,cb->ca',[self.W,self.act_func(hidden)])+Batch_Input[:,frame,:])
            hidden_t=torch.cat((hidden_t,self.act_func(hidden).unsqueeze(dim=0)),dim=0)
        geometry=self.Geometry(hidden_t)
        #返回的hidden_t(Batch_Size,T,N)和geometry(Batch_Size,T,2)
        return hidden_t.transpose(0,1),geometry.transpose(0,1)


