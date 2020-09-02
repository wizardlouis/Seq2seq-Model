from utils import *
from configs import *
from gene_seq import *
from losses import *
from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import argparse

import os

parser = argparse.ArgumentParser()
# set sequence parameters

# These are parameters that should be given
parser.add_argument("--out_dir", help="data out put filepath", type=str, default='./test')
parser.add_argument("--n_epochs", help="number of epochs", type=int, default=N_EPOCHS)
parser.add_argument("--learning_rate", help="learning rate of optimizer", type=float, default=LEARNING_RATE)
parser.add_argument("--burn_in", help="index of epoch that burn in is over", type=int, default=BURN_IN)
parser.add_argument("--batch_size", help="Number of sequence trained in a single batch", type=int, default=BATCH_SIZE)

# These are parameters that may not change but can be modulated
parser.add_argument("--n_item", help="number of items", type=int, default=N_ITEM)
parser.add_argument("--rank_size", help="number of points in sequence", type=int, default=RANK_SIZE)
parser.add_argument("--n_neuron", help="Number of neurons in the network", type=int, default=N_NEURONS)

parser.add_argument("--t_rest", help="resting time before target on", type=int, default=T_REST)
parser.add_argument("--t_on", help="target on time", type=int, default=T_ON)
parser.add_argument("--t_off", help="target off time", type=int, default=T_OFF)
parser.add_argument("--t_ron", help="retrieve on time", type=int, default=T_RON)
parser.add_argument("--t_roff", help="retrieve off time", type=int, default=T_ROFF)
parser.add_argument("--t_delay", help="delay time", type=int, default=T_DELAY)
parser.add_argument("--t_retrieve", help="retrieve time", type=int, default=T_RETRIEVE)
parser.add_argument("--t_cue", help="cue on time", type=int, default=T_CUE)
parser.add_argument("--t_add_delay_max", help="additional delay maximum", type=int, default=T_ADD)
parser.add_argument("--g_d", help="scaling of noise in seq2Input signal", type=float, default=G_D)
parser.add_argument("--decay", help="decay parameter of leaky neuron", type=float, default=DECAY)
parser.add_argument("--train_ratio", help="ratio of training sequence number compared to total sequences number",
                    type=float, default=TRAIN_RATIO)
parser.add_argument("--resume", help="resume training the model from a checkpoint", type=str, default="False")
parser.add_argument("--checkpoint_index", help="choose a checkpoint for resuming", type=int, default=0)
parser.add_argument("--save_checkpoint", help="save as a checkpoint after training", type=str, default="False")

args = parser.parse_args()

out_dir = args.out_dir
n_epochs = args.n_epochs
burn_in = args.burn_in
learning_rate = args.learning_rate

n_item = args.n_item
rank_size = args.rank_size
batch_size = args.batch_size
train_ratio = args.train_ratio
resume = args.resume
checkpoint_index = args.checkpoint_index
save_checkpoint = args.save_checkpoint

t_rest = args.t_rest
t_on = args.t_on
t_off = args.t_off
t_ron = args.t_ron
t_roff = args.t_roff
t_delay = args.t_delay
t_retrieve = args.t_retrieve
t_cue = args.t_cue
t_add_delay_max = args.t_add_delay_max
n_neuron = args.n_neuron
g_d = args.g_d

# -------------------preparing for training-----------------------------
activation = nn.Tanh()
model = MyRNN(n_neuron, activation)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = None
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
print(model.parameters())

data_hps = {
    't_rest': t_rest,
    't_on': t_ron,
    't_off': t_off,
    't_ron': t_ron,
    't_roff': t_roff,
    't_cue': t_cue,
    't_delay': t_delay,
    't_add_delay_max': t_add_delay_max,
    't_retrieve': t_retrieve,
    'rank_size': rank_size,
    'delay_fixed': False,
    'add_noise': False,
    'g': g_d
}

if resume == 'True':
    checkpoint = torch.load(out_dir + '/checkpoint/' + str(checkpoint_index) + '/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    seq = np.load(out_dir + '/checkpoint/' + str(checkpoint_index) + '/seq.npz')
    train_set = seq['train'].tolist()
    test_set = seq['test'].tolist()
else:
    train_set, test_set = gene_rank_sequence_set(ITEM_LIST, rank_size, train_ratio)
print("train set:", train_set, '\n')
print("test set:", test_set, '\n')

# ---------------------training------------------------------

trainer = Trainer(model, burn_in, batch_size, data_hps, optimizer=optimizer, lr_scheduler=scheduler,
                  train_set=train_set)
total_losses, retrieve_losses, delay_losses = trainer.run(n_epochs)

total_losses = np.array(total_losses)
retrieve_losses = np.array(retrieve_losses)
delay_losses = np.array(delay_losses)

# score_train = testNet_distance(model, trainset, Direction, 1)
# score_test = testNet_distance(model, testset, Direction, 1)
# print('final correct ratio:\ntrainset:{}\ntestset:{}'.format(str(score_train), str(score_test)))

# saving result and models
print(vars(args))

if save_checkpoint == 'True':
    if not os.path.exists(out_dir + '/checkpoint/' + str(checkpoint_index + 1)):
        os.makedirs(out_dir + '/checkpoint/' + str(checkpoint_index + 1))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               out_dir + '/checkpoint/' + str(checkpoint_index + 1) + '/model.pth')
    save(vars(args), out_dir + '/checkpoint/' + str(checkpoint_index + 1), 'args.txt', 'dictxt')
    np.save(out_dir + '/checkpoint/' + str(checkpoint_index + 1) + '/data_hps.npy', data_hps)
    np.savez(out_dir + '/checkpoint/' + str(checkpoint_index + 1) + '/seq.npz', train=train_set, test=test_set)
else:
    save(vars(args), out_dir, 'args.txt', 'dictxt')
    np.savez(out_dir + '//loss.npz', total_losses=total_losses, retrieve_losses=retrieve_losses,
             delay_losses=delay_losses)
    np.savez(out_dir + '//seq.npz', train=train_set, test=test_set)
    np.save(out_dir + '//data_hps.npy', data_hps)
    torch.save({'model': model, 'optimizer_state_dict': optimizer.state_dict()}, out_dir + '//model.pth')
