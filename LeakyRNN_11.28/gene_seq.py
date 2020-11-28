import math
import random
import torch
import numpy as np
from functools import *

####################################################################################################
#                                     Define vector representation                                 #
#                                     Transform Sequence to Input                                  #
#                                                                                                  #
####################################################################################################

#从Batch_Seq(Batch_Size,seq_length)到Batch_Input(Batch_Size,T,N)
def Batch_Seq2Input(Batch_Seq,Vectorrep,Vector_embedding,P,add_delay=None,device="cpu"):
    Batch_Size=len(Batch_Seq);L_seq=len(Batch_Seq[0])
    #Batch_Vec为每一个item的三维表示(Batch_Size,L_seq+1,3)
    Batch_Vec=torch.tensor([[Vectorrep[0]]+[Vectorrep[item] for item in Seq] for Seq in Batch_Seq],dtype=torch.float,requires_grad=False)
    Noise=P['g']*torch.randn(Batch_Vec.shape)
    Batch_Vec=Batch_Vec+Noise
    t_retrieve=P['t_rrest']+(L_seq-1)*P['t_rinterval']+P['t_ron']+P['n_windows']+10
    if add_delay==None:
        T=P['t_rest']+L_seq*P['t_on']+(L_seq-1)*P['t_off']+P['t_delay']+t_retrieve+random.randint(P['ad_min'],P['ad_max'])
    else:
        T=P['t_rest']+L_seq*P['t_on']+(L_seq-1)*P['t_off']+P['t_delay']+t_retrieve+add_delay
    Vary=torch.zeros(Batch_Size,1,dtype=torch.long)
    for i in range(L_seq-1):
        dV=Vary[:,i]+torch.randint(-P['t_vary'],P['t_vary']+1,(Batch_Size,1))
        Vary=torch.cat((Vary,dV),dim=1)
    #用一个temporal的张量表示各个点上是否有输入(Batch_Size,T,L_seq+1)
    temporal=torch.zeros(Batch_Size,T,L_seq+1,dtype=torch.float,requires_grad=False)
    #cue信号的位置，如果是persistent_cue，那就在整个retrieve阶段加cue信号
    temporal[:,-t_retrieve:-t_retrieve+P['t_cue'],0]=1
    for i in range(Batch_Size):
        for j in range(L_seq):
            temporal[i,P['t_rest']+j*(P['t_on']+P['t_off'])+Vary[i,j]:P['t_rest']+j*(P['t_on']+P['t_off'])+P['t_on']+Vary[i,j],j+1]=1
    Batch_Vec=Batch_Vec.to(device)
    temporal=temporal.to(device)
    Input=Vector_embedding(torch.einsum('abc,adb->adc',(Batch_Vec,temporal)))
    return Input

#用一个函数生成loss_weight序列
def loss_weight(shape,start,end,start_point,type='x'):
    L=np.zeros(shape)
    for i in range(shape[1]):
        n=shape[0]-start_point[i]-1
        L[start_point[i], i] = start[i]
        if type=='x':
            step=math.pow(end[i]/start[i],1/n)
            for j in range(start_point[i] + 1, shape[0]):
                L[j, i] = L[j - 1, i] * step
    return L

####################################################################################################
#                            Generate sequence with multiple principles                            #
#                            select balanced sequence                                              #
#                            create mixed length of sequence set                                   #
####################################################################################################

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

# select subset from sequence set so that every rank has balanced distribution
def balanced_select(Sequence, n_seq, n_item):
    while not balanced(Sequence[:n_seq], n_item):
        random.shuffle(Sequence)
    selected = Sequence[:n_seq]
    unselected = Sequence[n_seq:]
    return selected, unselected

def balanced(Seqs, n_item):
    Seqs_np = np.array(Seqs)
    Batch_Size, seq_length = Seqs_np.shape
    avg = round(Batch_Size / n_item);std = math.ceil(avg * 0.)
    # bincount() count from 0, but seqs start from 1,minlength=n_item+1,and select from 1st component
    count = np.concatenate([np.bincount(Seqs_np[:, i], minlength=n_item + 1)[1:] for i in range(seq_length)])
    print(count)
    if all(count >= avg - std) and all(count <= avg + std):
        print("selected trainset satisfy condition!!!\n")
        return True
    else:
        return False

#batchify data (n_seq,n_item) into (n_batch,n_seq_per_batch,n_item)
def batchify(data,batch_size):
    random.shuffle(data)
    N=len(data)
    n_batch=math.ceil(N/batch_size)
    batch=[]
    for i in range(n_batch):
        batch.append(data[i*batch_size:(i+1)*batch_size])
    return batch

#generate mixed length sequence set through a tuple of different length of sequence
#size_tuple telling batch sizes of each length
#repeat_tuple telling times of each sequence set should repeat
#for instance: data_tuple=(len1_seq,len2_seq) & size_tuple=(6,10) & repeat=(4,1)
def mixed_batchify(data_tuple,batch_size_tuple,repeat_tuple):
    N=len(data_tuple)
    P=[]
    #iterate through all seqset
    for i in range(N):
        #repeat times
        for j in range(repeat_tuple[i]):
            #generate random distributed batch_seq
            batch_seq=batchify(data_tuple[i],batch_size_tuple[i])
            for batch in batch_seq:
                P.append(batch)
    random.shuffle(P)
    return P

#generate gaussian distributed N-N matrix and normalized through eigenvalue maximum fitting to 1
def getJ(N,rho):
    J=np.random.normal(loc=0,scale=1/np.sqrt(N),size=(N,N))
    rs=max(np.real(np.linalg.eigvals(J)))
    return rho*J/rs

if __name__=="__main__":
    filepath='loss_weight//'
    lw=loss_weight((100,3),(1e-4,1e-2,1),(1e-4,1e-2,1),(0,0,10),'x')
    print(lw)
    np.save(filepath+'1000_f_1e-4_1e-2_1.npy',lw)