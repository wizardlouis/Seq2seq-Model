

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

def Batch_Seq2Input(Batch_Seq,Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue):
    L=len(Batch_Seq)
    Batch_Input=torch.tensor([])
    for seq_id in range(L):
        Input=Seq2Input(Batch_Seq[seq_id],Vectorrep,Vec_embedding,g,t_rest,t_on,t_off,t_delay,t_retrieve,t_cue)
        Batch_Input=torch.cat((Batch_Input,Input.unsqueeze(dim=0)),dim=0)
    return Batch_Input

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
#                                                                                                  #
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