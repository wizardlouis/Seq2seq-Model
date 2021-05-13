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

default_Vector = ([[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                   [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                   [math.cos(0), math.sin(0)],
                   [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                   [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                   [math.cos(math.pi), math.sin(math.pi)]
                   ])


# transform network parameters to sequence parameter
def p2p(net_param, L_seq, type='reproduct', add_param=None):
    if type == 'reproduct':
        return p2p_REPRODUCT(net_param, L_seq, add_param=add_param)


def p2p_REPRODUCT(net_param, L_seq, add_param=None):
    if add_param is None:
        add_param = dict()
    N = net_param.copy()
    param = dict()
    param.update({'in_Channel': N['in_Channel'], 't_on': N['t_on'], 't_ron': N['t_ron']})
    # fix_on:0
    pointer = 0
    # fixation loss t2fix(5 steps default) after fixation on
    param.update({'fix_b1':pointer,'fixl_b1': pointer + N['t2fix']})

    # determine if t_rest is randomly selected
    if 'Dt_rest' in add_param:
        pointer += N['t_rest'] + add_param['Dt_rest']
    else:
        pointer += N['t_rest'] + random.randint(0, N['Dt_rest'])

    if L_seq != 0:
        for i in range(L_seq):
            param.update({'t_b' + str(i): pointer, 't_e' + str(i): pointer + N['t_on']})
            if i != L_seq - 1:
                if 'Dt_on' + str(i) in add_param:
                    pointer += N['t_item'] + add_param['Dt_on' + str(i)]
                else:
                    pointer += N['t_item'] + random.randint(-N['Dt_on'], N['Dt_on'])
            else:
                pointer += N['t_on']
    if 'Dt_delay' in add_param:
        pointer += N['t_delay'] + add_param['Dt_delay']
    else:
        pointer += N['t_delay'] + random.randint(0, N['Dt_delay'])
    param.update({'fix_e1': pointer, 'fixl_e1': pointer})
    pointer += N['t_rrest']
    if L_seq != 0:
        for i in range(L_seq):
            param.update({'r_b' + str(i): pointer, 'r_e' + str(i): pointer + N['t_ron']})
            if i != L_seq - 1:
                pointer += N['t_retrieve']
            else:
                pointer += N['t_ron']
    param.update({'fix_b2':pointer,'fixl_b2':pointer+N['t2fix']})
    if 'Dt_final' in add_param:
        pointer+=N['t_final']+add_param['Dt_final']
    else:
        pointer+=N['t_final']+random.randint(0, N['Dt_final'])
    param.update({'total': pointer,'fix_e2':pointer,'fixl_e2':pointer})
    return param


# Batch_Seq(Batch_Size,seq_length) to Batch_Input(Batch_Size,T,In_Channel)
def Batch_Seq2Input(Batch_Seq, Vectorrep, P, strength=10, type='reproduct'):
    if type == 'reproduct':
        return Batch_Seq2Input_REPRODUCT(Batch_Seq, Vectorrep, P, strength=strength)
    elif type == 'reverse':
        return Batch_Seq2Input_REVERSE(Batch_Seq, Vectorrep, P)


# reproduct:fixation cue before retrieve
# t_delay flexible integrated before P given to this function
def Batch_Seq2Input_REPRODUCT(Batch_Seq, Vectorrep, P, strength=10):
    Batch_Size = len(Batch_Seq)
    L_seq = len(Batch_Seq[0])
    Vector = torch.tensor(Vectorrep)
    Input = torch.zeros(Batch_Size, P['total'], P['in_Channel'])
    for seq_id in range(Batch_Size):
        for item_id in range(L_seq):
            Input[seq_id, P['t_b' + str(item_id)]:P['t_e' + str(item_id)], :2] = strength * Vector[
                Batch_Seq[seq_id][item_id] - 1].unsqueeze(dim=0).repeat(
                P['t_on'], 1)
    Input[:, P['fix_b1']:P['fix_e1'], 2] = 1. * strength
    Input[:, P['fix_b2']:P['fix_e2'], 2] = 1. * strength
    return Input


def Batch_Seq2Input_REVERSE(Batch_Seq, Vectorrep, P):
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
        # repeat times,if 0 then rule out corresponding batch
        if batch_size_tuple[i] != 0:
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


def getUV(N, sigma, r):
    J = np.random.normal(loc=0, scale=1 / np.sqrt(N), size=(N, N))
    u, s, v = np.linalg.svd(J)
    return u[:, :r], sigma * v[:r]
