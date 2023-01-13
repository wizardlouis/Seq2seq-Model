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
from sklearn import mixture
import pandas as pd
# from Embedding_Dict import *

def search_kwargs(kwargs, name, value):
    if name in kwargs:
        return kwargs[name]
    else:
        return value

default_Vector = [[math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
                  [math.cos(math.pi / 3), math.sin(math.pi / 3)],
                  [math.cos(0), math.sin(0)],
                  [math.cos(-math.pi / 3), math.sin(-math.pi / 3)],
                  [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)],
                  [math.cos(math.pi), math.sin(math.pi)]
                  ]
default_colorset = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown'])

def GaussMixResample(n_samples, weights, means, covariances):
    """Generate random samples from the fitted Gaussian distribution.

            Parameters
            ----------
            n_samples : int, default=1
                Number of samples to generate.
            weights : array,
                weights of components.
            means : array,
                mean value of mixture Gaussians


            Returns
            -------
            X : array, shape (n_samples, n_features)
                Randomly generated sample.

            y : array, shape (nsamples,)
                Component labels.
            """

    _, n_features = means.shape
    rng = np.random.mtrand._rand
    n_samples_comp = rng.multinomial(n_samples, weights)

    X = np.vstack(
        [
            rng.multivariate_normal(mean, covariance, int(sample))
            for (mean, covariance, sample) in zip(
            means, covariances, n_samples_comp
        )
        ]
    )

    y = np.concatenate(
        [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
    )
    return (X, y)

class LoadingVector:
    def __init__(self, I: np.array, V: np.array, U: np.array, W=None, labels=None):
        self.I = I
        self.V = V
        self.U = U
        self.N = self.I.shape[0]
        assert self.N == self.V.shape[0]
        assert self.N == self.U.shape[0]
        self.N_I = self.I.shape[1]
        self.N_R = self.V.shape[1]
        assert self.N_R == self.U.shape[1]
        if W is not None:
            self.W = W
            assert self.N == self.W.shape[0]
            self.N_O = self.W.shape[1]
            self.Loading = np.concatenate([self.I, self.V, self.U, self.W], axis=1)
        else:
            self.Loading = np.concatenate([self.I, self.V, self.U], axis=1)
        if labels is not None:
            self.labels = labels.astype(int)
            assert self.N == len(self.labels)

    # return None for attribute not in object
    def __getattr__(self, item):
        return None

    def copy(self):
        return LoadingVector(self.I, self.V, self.U, W=self.W, labels=self.labels)

    def set_Statistics(self, n_components, weights, means, covariances):
        self.n_componnets = n_components
        self.weights = weights
        self.means = means
        self.covariances = covariances
        return self

    @staticmethod
    def from_mean_cov(n_components, weights, means, covariances, Iid, Vid, Uid, Wid=None):
        loading_vector, labels = GaussMixResample(n_components, weights, means, covariances)
        if Wid == None:
            return LoadingVector(loading_vector[:, Iid], loading_vector[:, Vid], loading_vector[:, Uid],
                                 labels=labels).set_Statistics(n_components, weights, means, covariances)
        else:
            return LoadingVector(loading_vector[:, Iid], loading_vector[:, Vid], loading_vector[:, Uid],
                                 W=loading_vector[:, Wid], labels=labels).set_Statistics(n_components, weights, means,
                                                                                         covariances)

    def save(self, filepath: str):
        data = dict(I=self.I, V=self.V, U=self.U)
        if self.W is not None:
            data.update(dict(W=self.W))
        if self.labels is not None:
            data.update(dict(labels=self.labels))
        if self.n_components is not None:
            data.update(dict(n_components=self.n_components, weights=self.weights, means=self.means,
                             covariances=self.covariances))
        np.savez(filepath, **data)

    @staticmethod
    def load(filepath: str):
        data = np.load(filepath, allow_pickle=True)
        file = dict()
        for key in list(data.keys()):
            file.update({key: data[key]})
        if 'W' not in file.keys():
            file['W'] = None
        if 'labels' not in file.keys():
            file['labels'] = None
        return LoadingVector(file['I'], file['V'], file['U'], W=file['W'], labels=file['labels'])

    def GaussFit(self, n_components: int, param: dict = None):
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, **param)
        dpgmm.fit(self.Loading)
        self.labels = dpgmm.predict(self.Loading).astype(int)
        self.n_components = n_components
        self.weights = dpgmm.weights_
        self.means = dpgmm.means_
        self.covariances = dpgmm.covariances_

    def get_Statistic(self):
        N_pop = max(self.labels) + 1
        N_R = self.Loading.shape[1]
        loading_aligned = [self.Loading[np.where(self.labels == p)[0], :] for p in range(N_pop)]
        self.weights = np.array([loading_aligned[p].shape[1] for p in range(N_pop)]) / self.N
        self.means = np.concatenate([loading_aligned[p].mean(axis=0).reshape((1, N_R)) for p in range(N_pop)], axis=0)
        self.covariances = np.concatenate([np.cov(loading_aligned[p].T).reshape((1, N_R, N_R)) for p in range(N_pop)],
                                          axis=0)

    def resample(self, n_samples):
        Sample, label = GaussMixResample(n_samples, self.weights, self.means, self.covariances)
        if self.W is not None:
            return LoadingVector(Sample[:, :self.N_I], Sample[:, self.N_I:self.N_I + self.N_R],
                                 Sample[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R],
                                 W=Sample[:, self.N_I + 2 * self.N_R:])
        else:
            return LoadingVector(Sample[:, :self.N_I], Sample[:, self.N_I:self.N_I + self.N_R],
                                 Sample[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R])

class table:
    def __init__(self, **kwargs):
        self.keys = list(kwargs.keys())
        for key in kwargs.keys():
            self[key] = kwargs[key]

    def __setitem__(self, key, value):
        if key!='keys':
            super().__setattr__(key, value)
            if key not in self.keys:
                self.keys.append(key)

    def __getitem__(self, key):
        assert key in self.keys
        super().__getattribute__(key)

    def to(self,device='cpu'):
        for key in self.keys:
            if type(self[key])==torch.Tensor:
                self[key]=self[key].to(device)

    def toDict(self):
        Data = dict()
        for key in self.keys:
            Data[key] = self.__getattribute__(key)
        return Data

    def show(self):
        for key in self.keys:
            print(key,'=',self.__getattribute__(key))

    def save(self, savepath):
        np.savez(savepath, **self.toDict())

    @staticmethod
    def load(filepath):
        Data = np.load(filepath)
        return table(**Data)

    def __add__(self, other):
        Dict1 = self.toDict()
        Dict2 = other.toDict()
        return table(**Dict1, **Dict2)

#hyperparameters reader

def hyper_reader(filepath,line):
    hyperparameter = pd.read_excel(filepath)
    param = hyperparameter[line:line + 1].to_dict('list')
    for key in param.keys():
        param[key] = param[key][0]
    param['savepath'] = 'Seq2seq_reparameter_sample//' + param['savepath']
    param['Embedding'] = torch.tensor(eval('Emb_' + param['emb'])).float()
    param.update(eval('Emb_' + param['emb_time']))
    return param


# save and load param dictionaries(if not needed in fast reading)
def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def create(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pf(a1, a2): plt.figure(figsize=(a1, a2))

def pfsp(m,n,a1,a2):
    fig,ax=plt.subplots(m,n,figsize=(a1,a2))
    return fig,ax


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
