import random
import torch
from utils import *
from configs import *


####################################################################################################
# Define vector representation                                 #
# Transform Sequence to Input                                  #
#                                                                                                  #
####################################################################################################

# select a balanced train set and test set in which element is single int.
def gene_rank_sequence_set(items, rank_size, train_ratio):
    all_sequences = fixed_rank_size_set(items, rank_size)
    train_size = int(train_ratio * len(all_sequences))
    while not balanced(all_sequences[:train_size], N_ITEM):
        random.shuffle(all_sequences)
    selected = all_sequences[:train_size]
    unselected = all_sequences[train_size:]
    return selected, unselected


# generate all possible sequence with length n in dataset items,output list (seq index,item ranks)
def fixed_rank_size_set(items, rank_size):
    if rank_size == 1:
        return [[item] for item in items]
    else:
        fixed_seq = []
        for item in items:
            rest_items = list(filter(lambda x: x != item, items))
            fixed_seq.extend([[item] + seq for seq in fixed_rank_size_set(rest_items, rank_size - 1)])
        return fixed_seq


# deciding whether the sequence set is balanced
def balanced(sequences, n_item):
    seqs_np = np.array(sequences)
    batch_size, seq_length = seqs_np.shape
    avg = round(batch_size / n_item)
    if batch_size % n_item == 0:
        std = 0
    else:
        std = math.ceil(avg * 0.1)
    # np.bincount() count from 0, but seqs start from 1, minlength=n_item+1, and select from 1st component
    count = np.concatenate([np.bincount(seqs_np[:, i], minlength=n_item + 1)[1:] for i in range(seq_length)])
    if all(count >= avg - std) and all(count <= avg + std):
        print(count)
        print("selected trainset satisfy condition!!!\n")
        return True
    else:
        return False


def gene_input_batch_seq(rank_seq_set, batch_size, hps):
    """
    randomly select some sequences from rank_seq_set,
     return a continue sequence batch which will as an input to the network,
     and corresponding rank sequence batch, target batch, t_delay
    :param rank_seq_set: a given sequence set composed by rank int
    :param batch_size:
    :param hps:
    :return:
    """
    random.shuffle(rank_seq_set)
    rank_batch = rank_seq_set[:batch_size]
    input_batch, target_batch, t_delay = rank_batch2input_batch(rank_batch, hps)

    return input_batch, rank_batch, target_batch, t_delay


def rank_batch2input_batch(rank_batch, hps):
    """
    convert a rank batch sequence set to a continue sequence batch
    :param rank_batch: a batch of rank seqs, (batch_size, rank_size)
    :param hps: some hyper parameters of sequences
    :return: a batch sequence with (seq_len, batch_size, feature_dim),
            corresponding target batch(delay and r_on) with (t_delay+n*t_ron, batch_size, 2),
            t_delay
    """
    vec = torch.tensor([[DEFAULT_VECTOR[0]] + [DEFAULT_VECTOR[item] for item in seq] for seq in rank_batch],
                       dtype=torch.float,
                       requires_grad=False)
    t_rest = hps['t_rest']
    t_on = hps['t_on']
    t_off = hps['t_off']
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_delay = hps['t_delay']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']

    rank_size = hps['rank_size']

    if not hps['delay_fixed']:
        t_add_delay_max = hps['t_add_delay_max']
        t_add = random.randint(0, t_add_delay_max)
        t_delay += t_add
    seq_len = t_rest + rank_size * t_on + (rank_size - 1) * t_off + t_delay + t_retrieve

    # generating input batch (seq_len, batch_size, feature_dim=3)
    temporal = torch.zeros(seq_len, rank_size + 1, dtype=torch.float, requires_grad=False)
    temporal[-t_retrieve: -t_retrieve + t_cue, 0] = 1
    for i in range(rank_size):
        temporal[t_rest + i * (t_on + t_off):t_rest + i * (t_on + t_off) + t_on, i + 1] = 1
    input_batch = torch.einsum('ab,cbd->acd', [temporal, vec])

    # generating target batch (t_delay + rank_size * t_ron, batch_size, 2)
    vec = vec[:, :, :2]
    temporal = torch.zeros(t_delay + rank_size * t_ron, rank_size + 1, dtype=torch.float, requires_grad=False)
    temporal[:t_delay, 0] = 1
    for i in range(rank_size):
        temporal[t_delay + i * t_ron: t_delay + (i + 1) * t_ron, i + 1] = 1
    target_batch = torch.einsum('ab,cbd->acd', [temporal, vec])

    if hps['add_noise']:
        noise = hps['g'] * torch.randn(input_batch.shape)
        input_batch += noise
    return input_batch, target_batch, t_delay


if __name__ == "__main__":
    # seq=[1,2,3]
    # em=torch.nn.Embedding(3,5,_weight=torch.tensor([[1,0,0,0,0],[0,1,1,0,0],[0,0,0,1,1]],dtype=torch.float))
    # Input=Seq2Input(seq,default_Vector,em,0.001,1,1,1,2,2,1)
    # print('Input=',Input)
    # Readout=Seq2Readout(seq,1,1)
    # print('Readout=',Readout)

    f = fixed_rank_size_set([1, 2, 3, 4, 5, 6], 2)
    random.shuffle(f)
    print(f)
    print(len(f))
