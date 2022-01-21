import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
import os
import math

default_Vector = [[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                  [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                  [math.cos(0), math.sin(0)],
                  [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                  [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                  [math.cos(math.pi), math.sin(math.pi)]
                  ]
default_colorset = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown'])


# save and load param dictionaries(if not needed in fast reading)
def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def pf(a1, a2): plt.figure(figsize=(a1, a2))


def tn(x): return x.cpu().detach().numpy()


def tt(x, dtype=torch.float, device="cpu"):
    return torch.tensor(x, dtype=dtype, device=device)


def w_dict(filepath, dict):
    f = open(filepath, 'w')
    f.write(str(dict))
    f.close()


# saving different types of files
def save(object, filepath, filename, type):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if type == 'dict':
        np.save(filepath + '//' + filename, object)
    elif type == 'dictxt':
        js = json.dumps(object)
        file = open(filepath + '//' + filename, 'w')
        file.write(js)
        file.close()
    elif type == 'npy':
        np.save(filepath + '//' + filename, object)
    pass


# read package from specific path
def loadpath(filepath, seq='seq.npz', loss='loss.npz', model='model.pth'):
    seq = np.load(filepath + '//' + seq)
    loss = np.load(filepath + '//' + loss)
    model = torch.load(filepath + '//' + model)
    return seq, loss, model


def plot_loss(filepath, n, label):
    data = np.array([])
    for i in range(n):
        loss = np.load(filepath + '//loss_' + str(i) + '.npz', allow_pickle=True)[label]
        data = np.concatenate((data, loss))
    plt.plot(list(range(len(data))), data)
    plt.show()

if __name__ == '__main__':
    # test = dict(a=1, b=[1, 2])
    # save_obj(test, 'test')
    test = load_obj('test')
    print(test)
