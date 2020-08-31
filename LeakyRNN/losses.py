from utils import *
from network import *
from gene_seq import gene_input_batch_seq

import numpy as np
import time


class Trainer(object):
    def __init__(self, model, burn_in, batch_size, hps, criterion=None, optimizer=None, train_set=None):
        """

        :param model:
        :param burn_in:
        :param hps:
        :param criterion:
        :param optimizer:
        :param train_set: rank seq train set
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_set = train_set
        self.batch_size = batch_size
        self.iterations = 0
        self.burn_in = burn_in
        self.hps = hps

    def run(self, epochs=1):
        total_losses, retrieve_losses, delay_losses = [], [], []
        for i in range(epochs):
            print("epoch: %i" % (i + 1))
            start = time.time()
            if i > self.burn_in:
                total_loss, retrieve_loss, delay_loss = self.train_step(self.hps, burn_in=True)
            else:
                total_loss, retrieve_loss, delay_loss = self.train_step(self.hps)
            total_losses.append(total_loss)
            retrieve_losses.append(retrieve_loss)
            delay_losses.append(delay_loss)
            end = time.time()
            cost = end - start
            print("using time: %f5 s \n" % cost)
        return total_losses, retrieve_losses, delay_losses

    def train_step(self, hps, burn_in=False):
        input_batch, rank_batch, target_batch, t_delay = gene_input_batch_seq(self.train_set, self.batch_size, hps)
        self.optimizer.zero_grad()

        hidden_0 = self.model.reset_hidden()
        hidden_t, geometry_t, _, _ = self.model(hidden_0, input_batch)
        total_loss, retrieve_loss, delay_loss = get_pos_loss(geometry_t, target_batch, t_delay, hps)
        if burn_in:
            total_loss.backward()
            total_loss, retrieve_loss, delay_loss = total_loss.data.item(), retrieve_loss.data.item(), delay_loss.data.item()
            print('total Loss = {}\nretrieve Loss = {}\ndelay Loss = {}'
                  .format(str(total_loss), str(retrieve_loss), str(delay_loss)))
        else:
            retrieve_loss.backward()
            total_loss, retrieve_loss, delay_loss = total_loss.data.item(), retrieve_loss.data.item(), delay_loss.data.item()
            print('retrieve Loss = {}'.format(str(retrieve_loss)))
        self.optimizer.step()
        return total_loss, retrieve_loss, delay_loss


def get_pos_loss(geometry_batch, target_batch, t_delay, hps):
    """
    loss function defined for output geometry batch sequence from network,
    calculate MSE loss of delay stage and ron stage.
    :param geometry_batch: (seq_len, batch_size, 2)
    :param target_batch: (t_delay + n * t_ron, batch_size, 2)
    :param t_delay:
    :param hps:
    :return:
    """
    criterion = nn.MSELoss()
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']
    rank_size = hps['rank_size']

    delay_target = target_batch[:t_delay, :, :]
    retrieve_target = target_batch[t_delay:, :, :]
    delay_geometry = geometry_batch[-t_retrieve - t_delay: -t_retrieve, :, :]
    retrieve_geometry = torch.tensor([])
    # TODO verify this split and cat
    for i in range(rank_size):
        retrieve_geometry = torch.cat((retrieve_geometry, geometry_batch[-t_retrieve + t_cue + i * (
                t_ron + t_roff): -t_retrieve + t_cue + i * (t_ron + t_roff) + t_ron, :, :]))

    delay_loss = criterion(delay_geometry, delay_target)
    retrieve_loss = criterion(retrieve_geometry, retrieve_target)
    total_loss = retrieve_loss + delay_loss
    return total_loss, retrieve_loss, delay_loss


# test single sequence and return Loss/CELoss/regLoss
def testNet_Loss(Net, Seq):
    n_item = len(Seq)
    MainLoss = nn.CrossEntropyLoss()
    hidden_0 = Net.reset_hidden()
    hidden, geometry, readout = Net(hidden_0, Seq)
    Target = torch.cat([torch.tensor([item - 1] * Net.t_ron) for item in Seq], dim=0)
    outTarget = Net.getTarget(readout, len(Seq))
    CELoss = MainLoss(outTarget, Target)
    scaleLoss = Net.scale_loss(Net.getTarget(geometry, n_item), 1)
    Loss = CELoss + scaleLoss
    return Loss.data.item(), CELoss.data.item(), scaleLoss.data.item()


# test a batch of sequence and return Loss/CELoss/regLoss
def Batch_testNet_Loss(Net, Batch_Seq):
    L = len(Batch_Seq)
    BatchLoss, BatchCELoss, BatchscaleLoss = 0., 0., 0.
    for seq_id in range(L):
        Loss, CELoss, scaleLoss = testNet_Loss(Net, Batch_Seq[seq_id])
        BatchLoss += Loss
        BatchCELoss += CELoss
        BatchscaleLoss += scaleLoss
    return BatchLoss, BatchCELoss, BatchscaleLoss


# Batch_test distance correct ratio
def testNet_distance(Net, Batch_Seq, Vectorrep, threshold):
    if Batch_Seq == []:
        return 'Test set is empty!'
    hidden_0 = Net.reset_hidden()
    hidden_t, geometry, readout = Net(hidden_0, Batch_Seq)
    # (T,Batch_Size,2)
    out = Net.getTarget(geometry, len(Batch_Seq[0]))
    trial = [[val for val in row for i in range(Net.t_ron)] for row in Batch_Seq]
    tretrieve_t = torch.tensor([[Vectorrep[item - 1] for item in row] for row in trial]).transpose(0, 1)
    # MSELoss=(T,Batch_Size)
    MSELoss = ((out - tretrieve_t) ** 2).sum(dim=2)
    length = Net.t_ron
    MSELoss = torch.cat(
        [MSELoss[i * length:(i + 1) * length].sum(dim=0).unsqueeze(dim=0) for i in range(len(Batch_Seq[0]))], dim=0)
    # item_result (Batch_Size,seq_length) seq_result(Batch_Size)
    item_result = (MSELoss < threshold).transpose(0, 1)
    seq_result = [all(result) for result in item_result]
    return sum(seq_result) / len(seq_result)


def Batch_testNet_distance(Net, Batch_Seq, Vectorrep, threshold):
    L = len(Batch_Seq)
    s = 0
    for seq_id in range(L):
        if testNet_distance(Net, Batch_Seq[seq_id], Vectorrep, threshold):
            s += 1
    return s / L


# test single sequence and return True or False based on majority of decoding performance in target priod
def testNet_True(Net, Seq, totalitem):
    hidden_0 = Net.reset_hidden()
    hidden_t, geometry, readout = Net(hidden_0, Seq)
    n_item = len(Seq)
    outTarget = Net.getTarget(readout, n_item)
    length = int(len(outTarget) / n_item)
    result = True
    for item in range(n_item):
        if not result:
            return False
        else:
            out_item = outTarget[item * length:(item + 1) * length]
            max_item = out_item.max(dim=1).indices.numpy()
            count = np.bincount(max_item, minlength=totalitem)
            if count[Seq[item] - 1] < int(1 / 2 * length):
                return False
    return result


# test a batch of sequence and return correct ratio
def Batch_testNet_True(Net, Batch_Seq, totalitem):
    L = len(Batch_Seq)
    s = 0
    for seq_id in range(L):
        if testNet_True(Net, Batch_Seq[seq_id], totalitem):
            s += 1
    return s / L
