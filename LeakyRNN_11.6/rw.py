import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from gene_seq import *

Direction=[
    [math.cos(0),math.sin(0)],
    [math.cos(math.pi/3),math.sin(math.pi/3)],
    [math.cos(math.pi*2/3),math.sin(math.pi*2/3)],
    [math.cos(math.pi),math.sin(math.pi)],
    [math.cos(math.pi*4/3),math.sin(math.pi*4/3)],
    [math.cos(math.pi*5/3),math.sin(math.pi*5/3)]
]

def pf(a1,a2):plt.figure(figsize=(a1,a2))
def tn(x): return x.cpu().detach().numpy()
def tt(x,dtype=torch.float,device="cpu"):
    return torch.tensor(x,dtype=dtype,device=device)

def w_dict(filepath,dict):
    f=open(filepath,'w')
    f.write(str(dict))
    f.close()

def r_dict(filepath):
    pass

#saving different types of files
def save(object,filepath,filename,type,*args,**kwargs):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if type=='dict':
        np.save(filepath+'//'+filename,object)
    elif type=='dictxt':
        js=json.dumps(object)
        file=open(filepath+'//'+filename,'w')
        file.write(js)
        file.close()
    elif type=='npy':
        np.save(filepath+'//'+filename,object)
    pass

#read package from specific path
def loadpath(filepath,seq='seq.npz',loss='loss.npz',model='model.pth'):
    seq=np.load(filepath+'//'+seq)
    loss=np.load(filepath+'//'+loss)
    model=torch.load(filepath+'//'+model)
    return seq,loss,model

#Batch_Seq(Batch_Size,seq_length)
def plot_trajectory(Net,Batch_Seq,row,column=5):
    Direct=np.array(Direction)
    h_0=Net.reset_hidden()
    #geometry (T,Batch_Size,2)
    _,geometry,_=Net(h_0,Batch_Seq)
    geometry = geometry.detach().numpy()
    c=np.zeros(len(geometry))
    for i in range(len(Batch_Seq[0])):
        c[-Net.t_retrieve+Net.t_cue+i*(Net.t_ron+Net.t_roff):-Net.t_retrieve+Net.t_cue+i*(Net.t_ron+Net.t_roff)+Net.t_ron]=1
    cm=plt.cm.get_cmap('bwr')
    for k in range(len(Batch_Seq)):
        plt.subplot(row,column,k+1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:,0], Direct[:,1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:,k,0],geometry[:,k,1],vmin=0,vmax=1,c=c,cmap=cm)
        plt.text(-2,-2,str(Batch_Seq[k]))
    plt.show()


def plot_trajectory2(Net,Batch_Seq,row,column=5):
    Direct=np.array(Direction)
    h_0=Net.reset_hidden()
    #geometry (T,Batch_Size,2)
    Input=Batch_Seq2Input(Batch_Seq,Net.Vectorrep,Net.Vec_embedding,Net.P)
    _,geometry=Net(h_0,Input)
    seq_length=len(Batch_Seq[0])
    geometry = geometry.detach().numpy()
    c=np.zeros(len(geometry[0]))
    t0=-Net.P['t_retrieve']+Net.P['t_cue']
    for i in range(seq_length):
        c[t0+i*Net.P['t_rinterval']:t0+i*Net.P['t_rinterval']+Net.P['t_ron']]=1
    cm=plt.cm.get_cmap('bwr')
    for k in range(len(Batch_Seq)):
        plt.subplot(row,column,k+1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:,0], Direct[:,1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[k,:,0],geometry[k,:,1],vmin=0,vmax=1,c=c,cmap=cm)
        plt.text(-2,-2,str(Batch_Seq[k]),fontsize=20)
    plt.show()

default_colorset=np.array(['green','orange','blue','brown','purple','red'])

def plot_trajectory3(Net,Batch_Seq,row,column=5,colorset=default_colorset):
    P=Net.P
    Direction=np.array(Net.Direction)
    colorset=list(colorset)
    h_0=Net.reset_hidden()
    Input=Batch_Seq2Input(Batch_Seq,Net.Vectorrep,Net.Vec_embedding,Net.P)
    _,geometry=Net(h_0,Input)
    seq_length=len(Batch_Seq[0])
    geometry=geometry.detach().numpy()
    t0=-P['t_retrieve']+P['t_rrest']
    dt_t=P['t_on']+P['t_off']
    dt_r=P['t_rinterval']
    for k in range(len(Batch_Seq)):
        plt.subplot(row,column,k+1)
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.scatter(Direction[:,0],Direction[:,1],color='',edgecolors=colorset,s=500,marker='o')
        k_geo=geometry[k]
        plt.plot(k_geo[:,0],k_geo[:,1],c='grey')
        for i in range(seq_length):
            plt.scatter(k_geo[P['t_rest']+i*dt_t:P['t_rest']+i*dt_t+P['t_on'],0],k_geo[P['t_rest']+i*dt_t:P['t_rest']+i*dt_t+P['t_on'],1],s=200,marker='x',c=colorset[Batch_Seq[k][i]-1])
            plt.scatter(k_geo[t0+i*dt_r:t0+(i+1)*dt_r,0],k_geo[t0+i*dt_r:t0+(i+1)*dt_r,1],s=200,marker='.',c=colorset[Batch_Seq[k][i]-1])
            plt.scatter(-1.3+0.2*i,-1.2,s=200,marker='o',c=colorset[Batch_Seq[k][i]-1])
        plt.scatter(k_geo[-P['t_retrieve']:-P['t_retrieve']+P['t_cue'],0],k_geo[-P['t_retrieve']:-P['t_retrieve']+P['t_cue'],1],s=200,c='black',marker='x')
        plt.scatter(k_geo[-6:,0],k_geo[-6:,1],s=200,c='black',marker='.')
        plt.text(-1.4,-1.4,str(Batch_Seq[k]),fontsize=20)
    plt.show()

def plot_norm(Net,Batch_Seq):
    I=Batch_Seq2Input(Batch_Seq,Net.Vectorrep,Net.Vec_embedding,Net.P)
    hidden_0=Net.reset_hidden()
    hidden,geo=Net(hidden_0,I)
    n2_hidden=torch.mean(torch.mean(hidden**2,dim=0),dim=1)
    plt.plot(list(range(len(n2_hidden))),n2_hidden.clone().detach().numpy())
    plt.show()