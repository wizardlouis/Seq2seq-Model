# -*- codeing = utf-8 -*-
# @time:2022/7/19 下午2:29
# Author:Xuewen Shen
# @File:PAIRS.py
# @Software:PyCharm

# This is a reproduction of Projection Angle Index of Response Similarity Analysis

import torch
import torch.nn as nn
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from rw import tn, tt, GaussMixResample, search_kwargs
import tqdm
import seaborn as sns


def data_norm(data: torch.Tensor):
    return data / (data.norm(dim=1).unsqueeze(dim=1))


def generate_distribution(n_sample, weight, means, covs):
    dist, _ = GaussMixResample(n_sample, weight, means, covs)
    return tt(dist)


def PAIRS(data: torch.Tensor, k=1, reduction=False, elliptical=False, metric='angle', **kwargs):
    '''
    :param data: a tensor of response matrix (N_Neurons,N_features)
    :param k: search through k-nearest neighbours, if None, using default method for searching, which requires hyperparameter thres_theta in kwargs
    :param reduction: if reduction, return clusterness only, if not Reduction, return spectrum of projection angles of sampling neurons
    :param elliptical: if elliptical, using elliptical single population gaussian distribution extracted from neural data as baseline,
                       else, using isotropic single population gaussian distribution as baseline.
    :param kwargs: [thres_theta,n_sample,n_bootstrap,max_k]
    :return:
    '''
    # Hyperparameters setup
    N, n_dim = data.shape

    # centering data
    data = data - data.mean(dim=0)
    # The median of angle distribution threshold
    thres_theta = search_kwargs(kwargs, 'thres_theta', math.pi / 4)
    # sample neurons
    n_sample = search_kwargs(kwargs, 'n_sample', 200)
    # bootstraps
    n_bootstrap = search_kwargs(kwargs, 'n_bootstrap', 10000)

    # If elliptical mode is used, first extract elliptical structure from data.
    data_null_weight = np.array([1.])
    if not elliptical:
        data_null_mean = np.zeros((1, n_dim))
        data_null_cov = np.expand_dims(np.eye(n_dim), axis=0)
    else:
        data_null_mean = tn(data.mean(dim=0).unsqueeze(dim=0))
        data_dm = data - data.mean(dim=0)
        data_null_cov = tn((data_dm.T @ data_dm / N).unsqueeze(dim=0))
    # determining minimal k value
    # if k is not given, calculating theta_threshold of median angles to get minimal sufficient k, else use given k values
    max_k = search_kwargs(kwargs, 'max_k', 8)
    if k is None:
        _K = 1
        while _K <= max_k:
            data_random = generate_distribution(n_sample, data_null_weight, data_null_mean, data_null_cov)
            random_mean_angle = PAIRS_base(data_random, _K)
            theta_random = torch.median(random_mean_angle)
            if theta_random > thres_theta:
                break
            _K += 1
        print(_K)
    else:
        _K = k

    # response matrix normalization
    data_norm_ = data_norm(data)

    # knn bootstrap for data distribution
    Dist_data = torch.zeros(n_bootstrap, n_sample)
    theta_data = torch.zeros(n_bootstrap)
    for b in tqdm.tqdm(range(n_bootstrap), desc='data distribution bootstrap:'):
        select = (torch.rand(n_sample) * N).long()
        Dist_data[b] = PAIRS_base(data_norm_[select], _K, metric=metric, normal=False)
        theta_data[b] = torch.median(Dist_data[b])

    # knn bootstrap for null distribution
    Dist_random = torch.zeros(n_bootstrap, n_sample)
    theta_random = torch.zeros(n_bootstrap)
    for b in tqdm.tqdm(range(n_bootstrap), desc='null distribution bootstrap:'):
        data_random = generate_distribution(n_sample, data_null_weight, data_null_mean, data_null_cov)
        Dist_random[b] = PAIRS_base(data_random, _K, metric=metric)
        theta_random[b] = torch.median(Dist_random[b])

    avg_theta_data = theta_data.mean().item()
    avg_theta_random = theta_random.mean().item()
    std_theta_random=theta_random.std().item()
    if reduction:
        return (avg_theta_random - avg_theta_data) / std_theta_random
    else:
        return (avg_theta_random - avg_theta_data) / std_theta_random, Dist_data, Dist_random


def PAIRS_base(data, k: int, metric='angle', normal=True):
    # Base method for PAIRS analysis
    # In this method, knn of data is calculated and the mean angle of knn is returned
    '''
    :param data: given data, Tensor of shape[n_sample, n_features]
    :param k: k-nearest neightbours, int
    :param norm: if norm, do normalization first, else, just leaving out (not recommended, but can be optimized if data is pre-processed)
    :return: Average angle of knn distribution of data
    '''
    if normal:
        data_norm_ = data_norm(data)
    else:
        data_norm_ = data.clone()
    if metric=='angle':
        Projection_matrix = data_norm_ @ data_norm_.T
        Angle_matrix = torch.acos(Projection_matrix)
        Nearest_Angle_matrix = Angle_matrix.sort(dim=1).values
        Mean_knn_Angle = Nearest_Angle_matrix[:, :k].mean(dim=1)
    elif metric=='cos':
        Projection_matrix = data_norm_ @ data_norm_.T
        cosine_matrix=Projection_matrix.sort(dim=1,descending=True).values
        Mean_knn_Angle=torch.acos(cosine_matrix[:,:k].mean(dim=1))
    return Mean_knn_Angle


def Dist_sort(Dist: torch.Tensor, limit: list):
    # A distribution of data and a list of K+1 boundaries are given, this function calculates how many data lies in each of the K intervals formed by K+1 boundaries
    '''
    :param Dist: A distribution of nearest nerghbour angle
    :param limit: a list of angle boundaries for plotting histogram
    :return: A distribution for how many angles lie in specific boundaries
    '''
    n_cluster = len(limit) - 1
    return torch.tensor([((Dist >= limit[i]) * (Dist < limit[i + 1])).sum().item() for i in range(n_cluster)],
                        dtype=torch.float)


angle_limit = [0., math.pi]


def ShowHist2(ax, Dist_data: torch.Tensor, Dist_null: torch.Tensor, **kwargs):
    '''
    :param ax:
    :param Dist_data:
    :param Dist_null:
    :param kwargs:
    :return:
    '''
    title = search_kwargs(kwargs, 'title', 'knn hist')
    xlabel_on = search_kwargs(kwargs, 'xlabel_on', False)
    ylabel_on = search_kwargs(kwargs, 'ylabel_on', False)
    data_hist_color = search_kwargs(kwargs, 'data_hist_color', 'orange')
    ax.set_title(title, fontsize=20)
    if xlabel_on:
        ax.set_xlabel('Angle', fontsize=20)
    sns.histplot(Dist_data.ravel(), color=data_hist_color, stat='density', ec=None, ax=ax)
    sns.kdeplot(Dist_null.ravel(), fill=False, color='black', ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def ShowHist(ax, Dist_data: torch.Tensor, Dist_null: torch.Tensor, resolution=12, **kwargs):
    '''
    :param ax: axes to plot on
    :param Dist_data: A distribution of knn for bootstraps shape of [n_bootstrap1,n_sample] or [n_sample,] in case not many data is provided
    :param Dist_null: A distribution of knn for bootstrap of null distribution [n_bootstrap2,n_sample]
    :param resolution: number of histograms
    :param kwargs: [] a dict of plotting hyperparameters
    :return:
    '''

    # plotting hyperparameter setup
    title = search_kwargs(kwargs, 'title', 'knn hist')
    xlabel_on = search_kwargs(kwargs, 'xlabel_on', False)
    ylabel_on = search_kwargs(kwargs, 'ylabel_on', False)
    null_line_color = search_kwargs(kwargs, 'null_line_color', 'blue')
    null_line_width = search_kwargs(kwargs, 'null_line_width', 4.)
    null_face_color = search_kwargs(kwargs, 'null_face_color', 'blue')
    null_face_alpha = search_kwargs(kwargs, 'null_face_alpha', 0.5)
    null_errorbar = search_kwargs(kwargs, 'null_errorbar', False)
    data_hist_color = search_kwargs(kwargs, 'data_hist_color', 'orange')
    data_hist_errorbar = search_kwargs(kwargs, 'data_hist_errorbar', False)
    data_hist_errorparam = search_kwargs(kwargs, 'data_hist_errorparam', dict(elinewidth=4., ecolor='black', capsize=5))
    n_null_bootstrap = Dist_null.shape[0]

    angle_width = (angle_limit[1] - angle_limit[0]) / resolution
    angle_list = [angle_limit[0] + i * angle_width for i in range(resolution + 1)]
    xtick_list = [(angle_list[i + 1] + angle_list[i]) / 2 for i in range(resolution)]

    ax.set_title(title, fontsize=20)
    if xlabel_on:
        ax.set_xlabel('Nearest Neighbour Angle', fontsize=20)
    if ylabel_on:
        ax.set_ylabel('Number of pairs', fontsize=20)

    # plotting baseline of null distribution
    Dist_null_sort = torch.cat([Dist_sort(Dist_null[b], angle_list).unsqueeze(dim=0) for b in range(n_null_bootstrap)],
                               dim=0)
    ax.plot(xtick_list, tn(Dist_null_sort.mean(dim=0)), color=null_line_color, linewidth=null_line_width)
    if null_errorbar:
        ax.fill_between(xtick_list, tn(Dist_null_sort.mean(dim=0) - Dist_null_sort.std(dim=0)),
                        tn(Dist_null_sort.mean(dim=0) + Dist_null_sort.std(dim=0)), facecolor=null_face_color,
                        alpha=null_face_alpha)
    # plotting histogram of data distribution
    if Dist_data.dim() == 1:
        Dist_data_sort = Dist_sort(Dist_data, angle_list)
        ax.bar(xtick_list, tn(Dist_data_sort), color=data_hist_color, width=angle_width)
    elif Dist_data.dim() == 2:
        n_data_bootstrap = Dist_data.shape[0]
        Dist_data_sort = torch.cat(
            [Dist_sort(Dist_data[b], angle_list).unsqueeze(dim=0) for b in range(n_data_bootstrap)], dim=0)
        if data_hist_errorbar:
            ax.bar(xtick_list, tn(Dist_data_sort.mean(dim=0)), yerr=tn(Dist_data_sort.std(dim=0)),
                   color=data_hist_color, width=angle_width,
                   error_kw=data_hist_errorparam)
        else:
            ax.bar(xtick_list, tn(Dist_data_sort.mean(dim=0)), color=data_hist_color, width=angle_width)
