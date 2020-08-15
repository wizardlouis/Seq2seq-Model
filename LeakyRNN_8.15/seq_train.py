from rw import *
from gene_seq import *
from network import *
import torch
import numpy as np
from torch import optim
import argparse
import random
import math

#batchify data (n_seq,n_item) into (n_batch,n_seq_per_batch,n_item)
def batchify(data,batch_size):
    N=len(data)
    n_batch=math.ceil(N/batch_size)
    batch=[]
    for i in range(n_batch):
        batch.append(data[i*batch_size:(i+1)*batch_size])
    return batch

def get_pos_Loss(Net,geo,Batch_Seq,temp,Vectorrep):
    Loss=nn.MSELoss()
    #get retrieve loss
    #(T,Batch_Size,2)
    tretrieve=Net.getTarget(geo,len(Batch_Seq[0]))
    #Batch_Seq (Batch_Size,seq_length) to trial(Batch_Size,seq_length*t_on)
    trial=[[val for val in row for i in range(Net.t_ron)] for row in Batch_Seq]
    #reverse Batch_Size axis and seq_length*t_on axis
    tretrieve_t=torch.tensor([[Vectorrep[item-1] for item in row] for row in trial]).transpose(0,1)
    retrieveLoss=Loss(tretrieve,tretrieve_t)
    #get delay loss under certain temperature
    tdelay=Net.getDelay(geo)
    tdelay_t=torch.zeros_like(tdelay)
    delayLoss=Loss(tdelay,tdelay_t)*temp
    totalLoss=retrieveLoss+delayLoss
    return totalLoss,retrieveLoss,delayLoss



#train a single batch of sequence using position MSELoss and return Loss/retrieveLoss/delayLoss
def train_pos(Net,Net_optimizer,scheduler,Seq,Vectorrep,temp):
    Net_optimizer.zero_grad()
    hidden_0=Net.reset_hidden()
    hidden, geometry, readout = Net(hidden_0, Seq)
    totalLoss, retrieveLoss, delayLoss = get_pos_Loss(Net, geometry, Seq, temp, Vectorrep)
    totalLoss.backward()
    Net_optimizer.step()
    if scheduler is not None:
        scheduler.step(totalLoss)
    return totalLoss.data.item(),retrieveLoss.data.item(),delayLoss.data.item()


#train a single batch of sequence and return Loss/CELoss/regLoss
def train(Net,Net_optimizer,scheduler,Seq,temp):
    Net_optimizer.zero_grad()
    n_batch=len(Seq)
    n_item=len(Seq[0])
    BatchCELoss=torch.tensor(0,dtype=torch.float);BatchscaleLoss=torch.tensor(0,dtype=torch.float)
    MainLoss = nn.CrossEntropyLoss()
    for seq_id in range(n_batch):
        hidden_0=Net.reset_hidden()
        hidden,geometry,readout=Net(hidden_0,Seq[seq_id])
        #Seq i for CrossEntropy index i-1
        Target=torch.cat([torch.tensor([item-1]*Net.t_ron) for item in Seq[seq_id]],dim=0)
        outTarget = Net.getTarget(readout, n_item)
        CELoss=MainLoss(outTarget,Target)
        #get scale loss with scale 1 in retrieve period so as to constrain the geometry on unit radius ring
        scaleLoss=Net.scale_loss(geometry,0,temp)
        BatchCELoss+=CELoss;BatchscaleLoss+=scaleLoss
    BatchLoss=BatchCELoss+BatchscaleLoss
    BatchLoss.backward()
    Net_optimizer.step()
    if scheduler is not None:
        scheduler.step(BatchLoss)
    return BatchLoss.data.item(),BatchCELoss.data.item(),BatchscaleLoss.data.item()

#Iter train with n_epochs and return Loss/CELoss/regLoss
def Iter_train(Net,Net_optimizer,scheduler,Seq,n_epochs,batch_size,temp):
    n_trial=len(Seq)
    n_batch=math.ceil(n_trial/batch_size)
    Batch_Seq=batchify(Seq,batch_size)
    order=list(range(n_batch))
    IterLoss,IterCELoss,IterscaleLoss=[],[],[]
    for epoch in range(n_epochs):
        Loss, CELoss, scaleLoss = 0., 0., 0.
        random.shuffle(order)
        for batch_id in order:
            BatchLoss,BatchCELoss,BatchscaleLoss=train(Net,Net_optimizer,scheduler,Batch_Seq[batch_id],temp)
            Loss+=BatchLoss;CELoss+=BatchCELoss;scaleLoss+=BatchscaleLoss
        IterLoss.append(Loss);IterCELoss.append(CELoss);IterscaleLoss.append(scaleLoss)
        print('Epoch {}:\nTotal Loss = {}\nCrossEntropy Loss = {}\nScale Loss = {}'
              .format(str(epoch),str(Loss),str(CELoss),str(scaleLoss)))
    return IterLoss,IterCELoss,IterscaleLoss

#train with position loss(delay fixed to zero point/retrieve to specific point)
def Iter_train_burn_pos(Net,Net_optimizer,scheduler,Seq,n_epochs,batch_size,Vectorrep,burn_in=300):
    n_trial=len(Seq)
    n_batch=math.ceil(n_trial/batch_size)
    Batch_Seq=batchify(Seq,batch_size)
    order=list(range(n_batch))
    IterLoss, IterreLoss, IterdelayLoss = [], [], []
    temp=0
    reach_max_temp=False
    for epoch in range(n_epochs):
        if burn_in==0 or temp>=1:
            reach_max_temp=True
            temp=1
        if epoch==burn_in:
            Net.use_scale_loss=True
            print("burn in is over, start using scale_loss!\n")
        if epoch>burn_in and not reach_max_temp:
            temp=(epoch-burn_in+1)/burn_in
        Loss, reLoss, delayLoss = 0., 0., 0.
        random.shuffle(order)
        for batch_id in order:
            BatchLoss,BatchreLoss,BatchdelayLoss=train_pos(Net,Net_optimizer,scheduler,Batch_Seq[batch_id],Vectorrep,temp)
            Loss+=BatchLoss;reLoss+=BatchreLoss;delayLoss+=BatchdelayLoss
        IterLoss.append(Loss);IterreLoss.append(reLoss);IterdelayLoss.append(delayLoss)
        print('Epoch {}:\nTotal Loss = {}\nretrieve Loss = {}\ndelay Loss = {}'
              .format(str(epoch),str(Loss),str(reLoss),str(delayLoss)))
    return IterLoss,IterreLoss,IterdelayLoss

def Iter_train_burn(Net,Net_optimizer,scheduler,Seq,n_epochs,batch_size,burn_in=300):
    n_trial=len(Seq)
    n_batch=math.ceil(n_trial/batch_size)
    Batch_Seq=batchify(Seq,batch_size)
    order=list(range(n_batch))
    IterLoss, IterCELoss, IterscaleLoss = [], [], []
    temp=0
    reach_max_temp=False
    for epoch in range(n_epochs):
        if burn_in==0 or temp>=1:
            reach_max_temp=True
            temp=1
        if epoch==burn_in:
            Net.use_scale_loss=True
            print("burn in is over, start using scale_loss!\n")
        if epoch>burn_in and not reach_max_temp:
            temp=(epoch-burn_in+1)/burn_in
        Loss, CELoss, scaleLoss = 0., 0., 0.
        random.shuffle(order)
        for batch_id in order:
            BatchLoss,BatchCELoss,BatchscaleLoss=train(Net,Net_optimizer,scheduler,Batch_Seq[batch_id],temp)
            Loss+=BatchLoss;CELoss+=BatchCELoss;scaleLoss+=BatchscaleLoss
        IterLoss.append(Loss);IterCELoss.append(CELoss);IterscaleLoss.append(scaleLoss)
        print('Epoch {}:\nTotal Loss = {}\nCrossEntropy Loss = {}\nScale Loss = {}'
              .format(str(epoch),str(Loss),str(CELoss),str(scaleLoss)))
    return IterLoss,IterCELoss,IterscaleLoss


#test single sequence and return Loss/CELoss/regLoss
def testNet_Loss(Net,Seq):
    n_item=len(Seq)
    MainLoss=nn.CrossEntropyLoss()
    hidden_0=Net.reset_hidden()
    hidden,geometry,readout=Net(hidden_0,Seq)
    Target = torch.cat([torch.tensor([item - 1] * Net.t_ron) for item in Seq], dim=0)
    outTarget = Net.getTarget(readout, len(Seq))
    CELoss=MainLoss(outTarget,Target)
    scaleLoss = Net.scale_loss(Net.getTarget(geometry,n_item),1)
    Loss=CELoss+scaleLoss
    return Loss.data.item(),CELoss.data.item(),scaleLoss.data.item()

#test a batch of sequence and return Loss/CELoss/regLoss
def Batch_testNet_Loss(Net,Batch_Seq):
    L=len(Batch_Seq)
    BatchLoss,BatchCELoss,BatchscaleLoss=0.,0.,0.
    for seq_id in range(L):
        Loss,CELoss,scaleLoss=testNet_Loss(Net,Batch_Seq[seq_id])
        BatchLoss+=Loss;BatchCELoss+=CELoss;BatchscaleLoss+=scaleLoss
    return BatchLoss,BatchCELoss,BatchscaleLoss

#Batch_test distance correct ratio
def testNet_distance(Net,Batch_Seq,Vectorrep,threshold):
    hidden_0 = Net.reset_hidden()
    hidden_t,geometry,readout=Net(hidden_0, Batch_Seq)
    #(T,Batch_Size,2)
    out=Net.getTarget(geometry,len(Batch_Seq[0]))
    trial = [[val for val in row for i in range(Net.t_ron)] for row in Batch_Seq]
    tretrieve_t = torch.tensor([[Vectorrep[item - 1] for item in row] for row in trial]).transpose(0, 1)
    #MSELoss=(T,Batch_Size)
    MSELoss=((out-tretrieve_t)**2).sum(dim=2)
    length=Net.t_ron
    MSELoss=torch.cat([MSELoss[i*length:(i+1)*length].sum(dim=0).unsqueeze(dim=0) for i in range(len(Batch_Seq[0]))],dim=0)
    #item_result (Batch_Size,seq_length) seq_result(Batch_Size)
    item_result=(MSELoss<threshold).transpose(0,1)
    seq_result=[all(result) for result in item_result]
    return sum(seq_result)/len(seq_result)


def Batch_testNet_distance(Net,Batch_Seq,Vectorrep,threshold):
    L=len(Batch_Seq)
    s=0
    for seq_id in range(L):
       if testNet_distance(Net,Batch_Seq[seq_id],Vectorrep,threshold):
           s+=1
    return s/L

#test single sequence and return True or False based on majority of decoding performance in target priod
def testNet_True(Net,Seq,totalitem):
    hidden_0 = Net.reset_hidden()
    hidden_t,geometry,readout=Net(hidden_0, Seq)
    n_item=len(Seq)
    outTarget=Net.getTarget(readout,n_item)
    length=int(len(outTarget)/n_item)
    result=True
    for item in range(n_item):
        if not result:
            return False
        else:
            out_item=outTarget[item*length:(item+1)*length]
            max_item=out_item.max(dim=1).indices.numpy()
            count=np.bincount(max_item,minlength=totalitem)
            if count[Seq[item]-1]<int(1/2*length):
                return False
    return result

#test a batch of sequence and return correct ratio
def Batch_testNet_True(Net,Batch_Seq,totalitem):
    L=len(Batch_Seq)
    s=0
    for seq_id in range(L):
       if testNet_True(Net,Batch_Seq[seq_id],totalitem):
           s+=1
    return s/L
