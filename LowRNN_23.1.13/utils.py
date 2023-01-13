# -*- codeing = utf-8 -*-
# @time:2021/8/8 下午4:31
# Author:Xuewen Shen
# @File:utils.py
# @Software:PyCharm

import os
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
import torch
import torch.nn as nn
import pandas as pd
import copy
import math
from rw import tn, tt, search_kwargs


# mean_precision_prior=1e5,mean_prior=0;
# dpgmm = mixture.BayesianGaussianMixture(n_components=5,
#                                         covariance_type='full').fit(X)


def reporter(path, string):
    f = open(path, 'a')
    f.write(string)
    f.close()


###################################
# model structure analysis
###################################
# save model svd result
def save_model_svd(model, savepath):
    '''
    :param model: seq2seq model
    :param savepath: saving file path
    :return:
    '''
    # svd of weight matrix if linear projection is given between encoder and decoder
    I = model.In.weight.data
    U_I, S_I, V_I = torch.svd(I)
    if model.P['R_Encoder'] == -1:
        J = model.Encoder.J.weight.data
    else:
        J = model.Encoder.U.weight.data @ model.Encoder.V.weight.data
    U_J, S_J, V_J = torch.svd(J)
    W = model.W.weight.data
    U_W, S_W, V_W = torch.svd(W)
    Q = model.Decoder.Q.weight.data
    U_Q, S_Q, V_Q = torch.svd(Q)
    O = model.Out.weight.data
    U_O, S_O, V_O = torch.svd(O)
    np.savez(savepath + '//svd.npz', U_I=U_I, S_I=S_I, V_I=V_I,
             U_J=U_J, S_J=S_J, V_J=V_J, U_W=U_W, S_W=S_W, V_W=V_W,
             U_Q=U_Q, S_Q=S_Q, V_Q=V_Q, U_O=U_O, S_O=S_O, V_O=V_O)


def get_svd_eigen_thru_training(filepath, N_m=100, N_neuron=4096, device='cpu'):
    Nlist = list(range(N_m))
    N = len(Nlist)
    NE = N_neuron
    SingL = [np.empty((N, NE, NE)), np.empty((N, NE)), np.empty((N, NE, NE))]
    EgL = [np.empty((N, NE), dtype=complex), np.empty((N, NE, NE), dtype=complex)]
    for i in Nlist:
        model = torch.load(filepath + '//model_{}.pth'.format(str(i)), map_location=device)
        J = (model.Encoder.U.weight.data @ model.Encoder.V.weight.data)
        Sing = torch.svd(J)
        for j in range(3):
            SingL[j][i] = Sing[j]
            if j == 2:
                print('{}th S finished!\n'.format(str(i)))
        Jn = J.numpy()
        Eg = np.linalg.eig(Jn)
        for j in range(2):
            EgL[j][i] = Eg[j]
            if j == 1:
                print('{}th E finished!\n'.format(str(i)))
    np.savez(filepath + '//Sing+Eigen.npz', U=SingL[0], S=SingL[1], V=SingL[2], Eval=EgL[0], Evec=EgL[1])


# save loading space from model vector loading space, Gaussian mixture model and savepath
def save_loading(vector, gmm, savepath):
    '''
    :param vector: (N_neuron,N_feature)
    :param gmm: mixture.BayesianGaussian model
    :param savepath: saving file path
    :return:
    '''
    labels = gmm.predict(vector)
    np.savez(savepath + '//loading.npz', labels=labels, n_components=gmm.n_components, weights_=gmm.weights_,
             means_=gmm.means_, covariances_=gmm.covariances_)


# save Gaussian mixture information to cluster_n=comp.xlsx
def save_GaussMix(gmm, columns, savepath):
    writer = pd.ExcelWriter(savepath + '//cluster_n=' + str(gmm.n_components) + '.xlsx')
    dfn = pd.DataFrame([[gmm.n_components]])
    dfn.to_excel(writer, 'n_components')
    dfw = pd.DataFrame(gmm.weights_)
    dfw.to_excel(writer, 'weights')
    dfm = pd.DataFrame(gmm.means_, columns=columns)
    dfm.to_excel(writer, 'means')
    for i in range(gmm.n_components):
        dfc = pd.DataFrame(gmm.covariances_[i], index=columns, columns=columns)
        dfc.to_excel(writer, 'covariances' + str(i))
    writer.save()
    writer.close()


# plot statistic of loading space with heatmap
def plot_statistics(loadpath, savepath, scale=[3., 1.]):
    '''
    :param loadpath: loading.xlsx file path
    :param scale: float, the scale of color bar
    :return:
    '''
    n_components = pd.read_excel(loadpath, sheet_name='n_components')[0][0]
    weights = list(pd.read_excel(loadpath, sheet_name='weights')[0])
    # plot cluster_mean
    dfm = pd.read_excel(loadpath, sheet_name='means')
    plt.figure(figsize=(len(dfm.columns) + 1, len(dfm.index) + 1), dpi=200)
    tot = weights @ np.array(dfm)
    dft = pd.DataFrame(tot, columns=dfm.columns, index=['tot'])
    dfm_ = pd.concat((dft, dfm), axis=0)
    sns.heatmap(data=dfm_, cmap=plt.get_cmap('bwr'), vmin=-scale[0], vmax=scale[0], annot=True)
    plt.title('cluster_mean')
    plt.savefig(savepath + '//cluster_mean_n=' + str(n_components))
    plt.close()
    # plot cluster_covariances respectively
    for i in range(n_components):
        dfc = pd.read_excel(loadpath, sheet_name='covariances' + str(i))
        plt.figure(figsize=(len(dfc.columns) + 1, len(dfc.index)), dpi=200)
        sns.heatmap(dfc, cmap=plt.get_cmap('bwr'), vmin=-scale[1], vmax=scale[1], annot=True)
        plt.title('cluster_cov' + str(i))
        plt.savefig(savepath + '//cluster_cov_n=' + str(n_components) + '_subpop=' + str(i))
        plt.close()


default_colorset=[f'C{i}' for i in range(10)]
def plot_loading(ax, x, y, labels, colorset=None, **kwargs):
    '''
    :param ax: subplots' axe to plot on
    :param x: x_data (N,)
    :param y: y_data (N,)
    :param labels: labels of clusters
    :param colorset: list of color str. color of each cluster
    :param kwargs:
    :return:
    '''
    if colorset is None:
        colorset=default_colorset
    xlabel_fontsize = search_kwargs(kwargs, 'xlabel_fontsize', 20)
    ylabel_fontsize = search_kwargs(kwargs, 'ylabel_fontsize', 20)

    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'],fontsize=xlabel_fontsize)
        ax.xaxis.set_ticks_position('')
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'],fontsize=ylabel_fontsize)
        ax.yaxis.set_ticks_position('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    N_pop = max(labels) + 1
    for i in range(N_pop):
        #get all points' index in i-th cluster
        i_index=np.where(np.array(labels)==i)[0]
        ax.scatter(x[i_index],y[i_index],color=colorset[i],s=10)

def plot_Gaussian_loading(axe, labels, coordinate, vector, cluster, colorset, xlabel=False, ylabel=True):
    """
    :param axe: plt.axes(subplots)
    :param labels: label of loading space
    :param coordinate: list (2,) 1-st/2-nd component rep. x/y axis component
    :param vector: array(n_labels,N_neuron) loding space vectors
    :param cluster: list of int. Vary from 0 to n-1; label of neuron clusters
    :param colorset: list of color str. color of each cluster
    :return: (n,n) upper triangle plots
    """
    # get x/y label index
    index_x = labels.index(coordinate[0])
    index_y = labels.index(coordinate[1])
    if xlabel:
        axe.set_xlabel(coordinate[0])
        axe.xaxis.set_ticks_position('')
    if ylabel:
        axe.set_ylabel(coordinate[1])
        axe.yaxis.set_ticks_position('')
    axe.set_xticks([])
    axe.set_yticks([])
    axe.spines['right'].set_color('none')
    axe.spines['top'].set_color('none')
    axe.spines['bottom'].set_position(('data', 0))
    axe.spines['left'].set_position(('data', 0))

    n_cluster = max(cluster) + 1
    for i in range(n_cluster):
        # get all points' index in i-th cluster
        n_array = [j for j, x in enumerate(cluster) if x == i]
        axe.scatter(vector[index_x][n_array], vector[index_y][n_array], c=colorset[i], s=10)


def matrix_reduction(M, eye):
    '''
    apply reduction to target matrix, only elements whose index in 'eye' will be maintained.
    :param M: target matrix to be 'reduced'
    :param eye: tuple of index([x11,x12..],[x21,x22...]),each element of which in the matrix=1.
    :return: a sparse matrix with 1. or 0. element
    '''
    Z = np.zeros_like(M)
    Z[eye] = 1.
    return Z * M


default_param = dict(tol=1e-5, covariance_type='full', init_params='kmeans', mean_precision_prior=0.0001,
                     max_iter=10000)


def GaussmixFit(loading, n_cluster, savepath=None, **kwargs):
    '''
    :param loading: Data of loading space to fit
    :param kwargs: A dict for hyperparameters setting in BayesianGaussianMixture model fitting
    :return fitted Mixture model
    '''
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_cluster,
                                            # mean_prior=(0,)*n_columns,
                                            **kwargs).fit(loading)
    if savepath is not None:
        np.savez(savepath + 'GaussMixFit.npz', labels=dpgmm.predict(loading), weights_=dpgmm.weights_,
                 means_=dpgmm.means_, covariances_=dpgmm.covariances_)
    return dpgmm


#####################################
# Reconstruct networks
#####################################


def Reduction_Rank(RefRNN, U, S, V):
    '''
    :param RefRNN:
    :param U: Right vector
    :param S:
    :param V:
    :return:
    '''
    RNN = copy.deepcopy(RefRNN)
    if RNN.Encoder.P['R'] == -1:
        RNN.Encoder.J.weight.data = U @ torch.diag(S) @ V.T
    else:
        RNN.P['R_Encoder'] = len(S)
        RNN.Encoder_P['R'] = len(S)
        RNN.Encoder.P['R'] = len(S)
        RNN.Encoder.U.weight.data = U @ torch.diag(torch.sqrt(S))
        RNN.Encoder.V.weight.data = torch.diag(torch.sqrt(S)) @ V.T
    return RNN


def Reduction_Population(RefRNN, labels):
    '''
    :param RefRNN: Reference RNN model
    :param labels: reverse neuron labels
    :return: population-reduction RNN model
    '''
    RNN = copy.deepcopy(RefRNN)
    RNN.P['N_Encoder'] = len(labels)
    RNN.Encoder_P['N'] = len(labels)
    RNN.Encoder.P['N'] = len(labels)

    RNN.In = nn.Linear(in_features=RNN.P['in_Channel'], out_features=RNN.P['N_Encoder'], bias=False)
    RNN.In.weight.data = RefRNN.In.weight.data[labels, :]
    if RNN.Encoder.P['R'] == -1:
        RNN.Encoder.J.weight.data = RefRNN.Encoder.J.weight.data[labels, labels]
    else:
        RNN.Encoder.U.weight.data = RefRNN.Encoder.U.weight.data[labels, :]
        RNN.Encoder.V.weight.data = RefRNN.Encoder.U.weight.data[:, labels]
    RNN.W.weight.data = RefRNN.W.weight.data[:, labels]
    return RNN


def resampling(loading_svd, RNN, N_rank, n_cluster, N_neuron, device='cpu', N_I=1):
    # calculate loading space
    ms = loading_svd
    NE = RNN.P['N_Encoder']
    U_I_n = (ms['In']['U'][:, :N_I]) * math.sqrt(NE)
    V_J_n = (ms['J']['V'] * np.sqrt(ms['J']['S']))[:, :N_rank] * math.sqrt(NE)
    U_J_n = (ms['J']['U'] * np.sqrt(ms['J']['S']))[:, :N_rank] * math.sqrt(NE)
    V_W_n = ms['W']['V'][:, :N_rank] * math.sqrt(NE)
    loading = np.concatenate((U_I_n, V_J_n, U_J_n, V_W_n), axis=1)

    # resampling from model
    gmm = GaussmixFit(loading, n_cluster, **default_param)
    sample = gmm.sample(n_samples=N_neuron)[0]
    U_I, V_J, U_J, V_W = sample[:, :N_I], sample[:, N_I:N_I + N_rank], sample[:, N_I + N_rank:N_I + 2 * N_rank], sample[
                                                                                                                 :,
                                                                                                                 N_I + 2 * N_rank:]
    model = copy.deepcopy(RNN)
    model.P['N_Encoder'] = N_neuron
    model.In.weight.data = tt(U_I) @ torch.diag(ms['In']['S'][:N_I].clone().detach()) @ ms['In']['V'][:,
                                                                                        N_I].clone().detach().T / math.sqrt(
        N_neuron)
    if model.P['R_Encoder'] != -1:
        model.Encoder.V.weight.data = tt(V_J).T / math.sqrt(N_neuron)
        model.Encoder.U.weight.data = tt(U_J) / math.sqrt(N_neuron)
    else:
        model.Encoder.J.weight.data = tt(U_J @ V_J.T) / N_neuron
    model.W.weight.data = ms['W']['U'][:, :N_rank].clone().detach().T @ torch.diag(
        ms['W']['s'][:, :N_rank].clone().detach()) @ tt(V_W.T)
    return model
