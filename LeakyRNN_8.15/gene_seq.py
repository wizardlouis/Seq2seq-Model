

import math
import random
import torch
from rw import *

####################################################################################################
#                                     Define vector representation                                 #
#                                     Transform Sequence to Input                                  #
#                                                                                                  #
####################################################################################################
#default vector is used to represent sequence in three dimension:(x,y,cue)
#the first component must be Vec rep. of cue signal
default_Vector=[
    [0.,0.,1.],
    [math.cos(0),math.sin(0),0.],
    [math.cos(math.pi/3),math.sin(math.pi/3),0.],
    [math.cos(math.pi*2/3),math.sin(math.pi*2/3),0.],
    [math.cos(math.pi),math.sin(math.pi),0],
    [math.cos(math.pi*4/3),math.sin(math.pi*4/3),0.],
    [math.cos(math.pi*5/3),math.sin(math.pi*5/3),0.]
]


#transform single sequence(L_seq) to Input(T,N) with noise level g
#t_on,t_off,t_delay,t_retrieve define total time course of trial
def Seq2Input(Seq,Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue):
    L_seq = len(Seq)
    #Vec(L+1,3);axis(3,N);temporal(T,L+1);Input(T,N)=temporal@Vec@axis
    #(L+1)--0 for cue & 1: for seq;
    Vec=torch.tensor([Vectorrep[0]]+[Vectorrep[item] for item in Seq],dtype=torch.float,requires_grad=False)
    # elif type(Vectorrep)==torch.tensor:
    #     Vec=Vectorrep[0].unsqueeze(dim=0)
    #     for i in range(L_seq):
    #         Vec=torch.cat((Vec,Vectorrep[seq[i]].unsqueeze(dim=0)),dim=0)
    T=t_rest+L_seq*t_on+(L_seq-1)*t_off+t_delay+t_retrieve
    temporal=torch.zeros(T,L_seq+1,dtype=torch.float,requires_grad=False)
    temporal[-t_retrieve:-t_retrieve+t_cue,0]=1
    for i in range(L_seq):
        temporal[t_rest+i*(t_on+t_off):t_rest+i*(t_on+t_off)+t_on,i+1]=1
    #Vec_embedding for (x,y,cue)
    axis=Vec_embedding(torch.tensor([0,1,2]))
    pureInput=temporal@Vec@axis
    noise=g*torch.randn(pureInput.shape)
    Input=pureInput+noise
    return Input

#if delay period not fixed
def Seq2Input_vary_delay(Seq,Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue,min,max):
    var=random.randint(min,max)
    L_seq = len(Seq)
    #Vec(L+1,3);axis(3,N);temporal(T,L+1);Input(T,N)=temporal@Vec@axis
    #(L+1)--0 for cue & 1: for seq;
    Vec=torch.tensor([Vectorrep[0]]+[Vectorrep[item] for item in Seq],dtype=torch.float,requires_grad=False)
    # elif type(Vectorrep)==torch.tensor:
    #     Vec=Vectorrep[0].unsqueeze(dim=0)
    #     for i in range(L_seq):
    #         Vec=torch.cat((Vec,Vectorrep[seq[i]].unsqueeze(dim=0)),dim=0)
    T=t_rest+L_seq*t_on+(L_seq-1)*t_off+t_delay+t_retrieve+var
    temporal=torch.zeros(T,L_seq+1,dtype=torch.float,requires_grad=False)
    temporal[-t_retrieve:-t_retrieve+t_cue,0]=1
    for i in range(L_seq):
        temporal[t_rest+i*(t_on+t_off):t_rest+i*(t_on+t_off)+t_on,i+1]=1
    #Vec_embedding for (x,y,cue)
    axis=Vec_embedding(torch.tensor([0,1,2]))
    pureInput=temporal@Vec@axis
    noise=g*torch.randn(pureInput.shape)
    Input=pureInput+noise
    return Input

def Batch_Seq2Input(Batch_Seq,Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue):
    L_seq = len(Batch_Seq[0])
    #Vec(Batch_Size,L+1,3);axis(3,N);temporal(T,L+1);Input(T,N)=temporal@Vec@axis
    #(L+1)--0 for cue & 1: for seq;
    Vec=torch.tensor([[Vectorrep[0]]+[Vectorrep[item] for item in Seq] for Seq in Batch_Seq],dtype=torch.float,requires_grad=False)

    T=t_rest+L_seq*t_on+(L_seq-1)*t_off+t_delay+t_retrieve
    temporal=torch.zeros(T,L_seq+1,dtype=torch.float,requires_grad=False)
    temporal[-t_retrieve:-t_retrieve+t_cue,0]=1
    for i in range(L_seq):
        temporal[t_rest+i*(t_on+t_off):t_rest+i*(t_on+t_off)+t_on,i+1]=1
    #Vec_embedding for (x,y,cue)
    axis=Vec_embedding(torch.tensor([0,1,2]))
    pureInput=torch.einsum('ab,cbd->cad',[temporal,Vec])@axis
    noise=g*torch.randn(pureInput.shape)
    Input=pureInput+noise
    return Input

#Batch_Seq satisfy (number_seq,seq_length)
#return (num_seq,T,N)
def Batch_Seq2Input_vary_delay(Batch_Seq,Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue,min,max):
    var=random.randint(min,max)
    L_seq = len(Batch_Seq[0])
    #Vec(Batch_Size,L+1,3);axis(3,N);temporal(T,L+1);Input(T,N)=temporal@Vec@axis
    #(L+1)--0 for cue & 1: for seq;
    Vec=torch.tensor([[Vectorrep[0]]+[Vectorrep[item] for item in Seq] for Seq in Batch_Seq],dtype=torch.float,requires_grad=False)

    T=t_rest+L_seq*t_on+(L_seq-1)*t_off+t_delay+t_retrieve+var
    temporal=torch.zeros(T,L_seq+1,dtype=torch.float,requires_grad=False)
    temporal[-t_retrieve:-t_retrieve+t_cue,0]=1
    for i in range(L_seq):
        temporal[t_rest+i*(t_on+t_off):t_rest+i*(t_on+t_off)+t_on,i+1]=1
    #Vec_embedding for (x,y,cue)
    axis=Vec_embedding(torch.tensor([0,1,2]))
    pureInput=torch.einsum('ab,cbd->cad',[temporal,Vec])@axis
    noise=g*torch.randn(pureInput.shape)
    Input=pureInput+noise
    return Input


#transform single sequence (L_seq) to Readout(T,6)
def Seq2Readout(Seq,t_on,t_off):
    L_seq=len(Seq)
    Readout=torch.zeros(L_seq*t_on+(L_seq-1)*t_off,6,dtype=torch.float,requires_grad=False)
    #Readout y index 0~5 correspond to Seq 1~6
    for i in range(L_seq):
        Readout[i*(t_on+t_off):i*(t_on+t_off)+t_on,Seq[i]-1]=1.
    return Readout

####################################################################################################
#                            Generate sequence with multiple principles                            #
#                            select balanced sequence                                              #
#                                                                                                  #
####################################################################################################

#generate given length sequence with fixed trial numbers
#gene sequence with all possible combinations with matrix returned: size*n_item

#generate all possible sequence with length n in dataset items,output list (seq index,item ranks)
def fixedLen_seq(items,n):
    if n==1:
        return [[item] for item in items]
    else:
        fseq=[]
        for item in items:
            restitems=list(filter(lambda x:x!=item,items))
            fseq.extend([[item]+seq for seq in fixedLen_seq(restitems,n-1)])
        return fseq

def balanced_select(Sequence,n_seq,n_item):
    while not balanced(Sequence[:n_seq],n_item):
        random.shuffle(Sequence)
    selected = Sequence[:n_seq]
    unselected = Sequence[n_seq:]
    return selected, unselected


def balanced(Seqs,n_item):
    Seqs_np = np.array(Seqs)
    Batch_Size,seq_length=Seqs_np.shape
    avg=round(Batch_Size/n_item);std=math.ceil(avg*0.2)
    #bincount() count from 0, but seqs start from 1,minlength=n_item+1,and select from 1st component
    count=np.concatenate([np.bincount(Seqs_np[:,i],minlength=n_item+1)[1:] for i in range(seq_length)])
    print(count)
    if all(count>=avg-std) and all(count<=avg+std):
        print("selected trainset satisfy condition!!!\n")
        return True
    else:
        return False






def getJ(N,rho):
    J=np.random.normal(loc=0,scale=1/np.sqrt(N),size=(N,N))
    rs=max(np.real(np.linalg.eigvals(J)))
    return rho*J/rs


if __name__=="__main__":
    # seq=[1,2,3]
    # em=torch.nn.Embedding(3,5,_weight=torch.tensor([[1,0,0,0,0],[0,1,1,0,0],[0,0,0,1,1]],dtype=torch.float))
    # Input=Seq2Input(seq,default_Vector,em,0.001,1,1,1,2,2,1)
    # print('Input=',Input)
    # Readout=Seq2Readout(seq,1,1)
    # print('Readout=',Readout)

    f=fixedLen_seq([1,2,3,4,5,6],2)
    random.shuffle(f)
    print(f)
    print(len(f))