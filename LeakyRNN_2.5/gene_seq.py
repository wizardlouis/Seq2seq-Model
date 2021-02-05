import math
import random
import torch
import numpy as np
from functools import *

####################################################################################################
#                                                                                                  #
#                                     Transform Sequence to Input                                  #
#                                                                                                  #
####################################################################################################

default_Vector = ([[0., 0., 1.],
                   [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3), 0.],
                   [math.cos(math.pi / 3), math.sin(math.pi / 3), 0.],
                   [math.cos(0), math.sin(0), 0.],
                   [math.cos(-math.pi / 3), math.sin(-math.pi / 3), 0.],
                   [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3), 0.],
                   [math.cos(math.pi), math.sin(math.pi), 0.]
                   ])


# Batch_Seq(Batch_Size,seq_length) to Batch_Input(Batch_Size,T,In_Channel)
def Batch_Seq2Input(Batch_Seq, Vectorrep, P, type='reproduct', add_P=None):
    if type == 'reproduct':
        return Batch_Seq2Input_REPRODUCT(Batch_Seq, Vectorrep, P, add_P)
    elif type == 'reverse':
        return Batch_Seq2Input_REVERSE(Batch_Seq, Vectorrep, P, add_P)


def Batch_Seq2Input_REPRODUCT(Batch_Seq, Vectorrep, P, add_P):
    Batch_Size = len(Batch_Seq)
    L_seq = len(Batch_Seq[0])
    Vector = torch.tensor(Vectorrep)
    if add_P == None:
        t_total = L_seq * (P['t_item'] + P['t_retrieve']) + P['t_delay'] + random.randint(0, P['Dt_delay']) + P[
            't_cue'] + P['t_final']
    else:
        t_total = L_seq * (P['t_item'] + P['t_retrieve']) + P['t_delay'] + add_P['Dt_delay'] + P['t_cue'] + P['t_final']
    # Input=torch.zeros(Batch_Size,t_total,3)--0,1:x,y;2:Go signal
    Input = torch.zeros(Batch_Size, t_total, 3)
    for seq_id in range(Batch_Size):
        for item_id in range(L_seq):
            t_item_on = (item_id + 1) * P['t_item'] - P['t_on'] + random.randint(-P['Dt_on'], P['Dt_on'])
            # Seq从1开始，Vector 0 是cue信号，1开始是Seq
            Input[seq_id, t_item_on:t_item_on + P['t_on']] = Vector[Batch_Seq[seq_id][item_id]].unsqueeze(dim=0).repeat(
                P['t_on'], 1)
    t_cue_on = -P['t_final'] - L_seq * P['t_retrieve'] - P['t_cue']
    Input[:, t_cue_on:t_cue_on + P['t_cue']] = Vector[0]
    return Input


def Batch_Seq2Input_REVERSE(Batch_Seq, Vectorrep, P, add_P):
    pass


####################################################################################################
#                            Generate sequence with multiple principles                            #
#                            select balanced sequence                                              #
#                            create mixed length of sequence set                                   #
####################################################################################################

# generate all possible sequence with length n in dataset items,output list (seq index,item ranks)
def Gen_Seq(items, n, repeat=False):
    if repeat:
        return Gen_Seq_REPEAT(items, n)
    else:
        return Gen_Seq_NOREPEAT(items, n)


def Gen_Seq_NOREPEAT(items, n):
    if n == 1:
        return [[item] for item in items]
    else:
        fseq = []
        for item in items:
            restitems = list(filter(lambda x: x != item, items))
            fseq.extend([[item] + seq for seq in Gen_Seq_NOREPEAT(restitems, n - 1)])
        return fseq


def Gen_Seq_REPEAT(items, n):
    if n == 1:
        return [[item] for item in items]
    else:
        fseq = []
        for item in items:
            fseq.extend([[item] + seq for seq in Gen_Seq_REPEAT(items, n - 1)])
        return fseq


# select subset from sequence set so that every rank has balanced distribution
def balanced_select(Sequence, n_seq, n_item):
    while not balanced(Sequence[:n_seq], n_item):
        random.shuffle(Sequence)
    selected = Sequence[:n_seq]
    unselected = Sequence[n_seq:]
    return selected, unselected


def balanced(Seqs, n_item):
    if Seqs == []:
        return True
    else:
        Seqs_np = np.array(Seqs)
        Batch_Size, seq_length = Seqs_np.shape
        avg = round(Batch_Size / n_item)
        std = math.ceil(avg * 0.05)
        # bincount() count from 0, but seqs start from 1,minlength=n_item+1,and select from 1st component
        count = np.concatenate([np.bincount(Seqs_np[:, i], minlength=n_item + 1)[1:] for i in range(seq_length)])
        print(count)
        if all(count >= avg - std) and all(count <= avg + std):
            print("selected trainset satisfy condition!!!\n")
            return True
        else:
            return False


# batchify data (n_seq,n_item) into (n_batch,n_seq_per_batch,n_item)
def batchify(data, batch_size):
    if batch_size == 0:
        return []
    else:
        random.shuffle(data)
        N = len(data)
        n_batch = math.ceil(N / batch_size)
        batch = []
        for i in range(n_batch):
            batch.append(data[i * batch_size:(i + 1) * batch_size])
        return batch


# generate mixed length sequence set through a tuple of different length of sequence
# size_tuple telling batch sizes of each length
# repeat_tuple telling times of each sequence set should repeat
# for instance: data_tuple=(len1_seq,len2_seq) & size_tuple=(6,10) & repeat=(4,1)
def mixed_batchify(data_tuple, batch_size_tuple, repeat_tuple):
    N = len(data_tuple)
    P = []
    # iterate through all seqset
    for i in range(N):
        # repeat times
        for j in range(repeat_tuple[i]):
            # generate random distributed batch_seq
            batch_seq = batchify(data_tuple[i], batch_size_tuple[i])
            for batch in batch_seq:
                P.append(batch)
    random.shuffle(P)
    return P


# generate gaussian distributed N-N matrix and normalized through eigenvalue maximum fitting to rho
def getJ(N, rho):
    J = np.random.normal(loc=0, scale=1 / np.sqrt(N), size=(N, N))
    rs = max(np.real(np.linalg.eigvals(J)))
    return rho * J / rs
