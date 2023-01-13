# -*- codeing = utf-8 -*-
# @time:2022/5/8 下午5:18
# Author:Xuewen Shen
# @File:SubpopulationAnalysis.py
# @Software:PyCharm

import numpy as np
import torch
import torch.nn as nn
import math
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rw import *
from sklearn import cluster
import multiprocessing
from multiprocessing import Pool, TimeoutError
from tqdm import tqdm, trange
from collections import Counter
import random
from itertools import combinations

default_param = dict(tol=1e-5, covariance_type='full', init_params='kmeans', mean_precision_prior=0.0001,
                     max_iter=10000)


# generating resampling loading vectors (sample,label) through statistics
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

def CreatePath(savepath):
    if savepath != None and not os.path.exists(savepath):
        os.makedirs(savepath)


def MultiPopTrialFitting(loading, Poplist, param, n_trials=128, savepath=None):
    CreatePath(savepath)
    result = {}
    N = loading.shape[0]
    for p in Poplist:
        result_p = np.empty((0, N))
        for t in tqdm(range(n_trials), desc=f"computation for N_pop={p}"):
            dpgmm = mixture.BayesianGaussianMixture(n_components=p, **param)
            dpgmm.fit(loading)
            labels = dpgmm.predict(loading).reshape((1, N))
            result_p = np.concatenate((result_p, labels), axis=0)
        result.update({'p' + str(p): result_p})
    if savepath != None:
        np.savez(savepath + 'MultiPopTrialFitting.npz', **result)
        print(f'Saving result to {savepath}//MultiPopTrialFitting.npz')
    return result


# initiate Ideal Affinity matrix
def IdealAffinity(labels, init='random'):
    n_cluster = max(labels) + 1
    N = len(labels)
    if init == 'random':
        M = np.ones((N, N)) / n_cluster
    if init == 'zero':
        M = np.zeros((N, N))
    for i in range(N):
        idx = np.where(labels == i)[0]
        for k in idx:
            M[k, idx] = 1
    return M


def perm(List: list, i: int, j: int) -> list:
    '''
    get permutation i<->j of List
    :param List:
    :param i:
    :param j:
    :return:
    '''
    l = List.copy()
    l[j], l[i] = List[i], List[j]
    return l


def all_perm(List: list) -> list:
    '''
    get all permutation from list
    :param List: A list of items
    :return:
    '''
    if len(List) == 1:
        return [[i] for i in List]
    else:
        return [[i] + k for i in List for k in all_perm(list(filter(lambda x: x != i, List)))]


def get_prob_matrix(n_components: int, label1: np.array, label2: np.array):
    assert len(label1) == len(label2)
    assert min(label1) == 0
    assert min(label2) == 0
    assert max(label1) == n_components - 1
    assert max(label2) == n_components - 1
    N = len(label1)
    M = np.zeros((n_components, n_components))
    for i in range(N):
        M[label1[i], label2[i]] += 1.
    return M / N


class SimilarityOptimizer:
    def __init__(self, prob_matrix):
        self.prob_matrix = prob_matrix.copy()
        assert self.prob_matrix.shape[0] == self.prob_matrix.shape[1]
        self.N = self.prob_matrix.shape[0]
        # [N,2]
        self.permutation = list(range(self.N))

    def reinit(self):
        self.combinations = list(combinations((list(range(self.N))), 2))
        self.overlap = self.get_overlap()

    def get_permutation(self):
        return self.permutation

    def get_overlap(self, permutation=None):
        if permutation is None:
            permutation = self.permutation.copy()
        return sum([self.prob_matrix[i, pos] for i, pos in enumerate(permutation)])

    def search_greedy(self):
        '''
        searching for permutations that contribute to overlap in optimized direction
        :param n_steps:
        :param batch:
        :return:
        '''
        InitP = self.permutation.copy()
        neighboor_permutation = np.array([perm(InitP, comb[0], comb[1]) for comb in self.combinations])
        overlap = np.array([self.get_overlap(permutation=nbp) for nbp in neighboor_permutation])
        return neighboor_permutation[np.where(overlap == overlap.max())], overlap.max()

    def update_greedy_once(self):
        next_permutation, next_overlap = self.search_greedy()
        if next_overlap > self.overlap:
            self.permutation = next_permutation
            self.overlap = next_overlap
            return True
        else:
            return False

    def update_greedy(self, n_steps=-1, get_traj=False):
        '''
        optimize permutation and overlap through greedy algorithm
        :param n_steps: limit steps to optimize if not==-1
        :param get_traj: whether we should get the optimization history
        :return:
        '''
        traj_t = [self.get_permutation()]
        overlap_t = [self.overlap()]

        if n_steps == -1:
            next = True

            def generator():
                while next:
                    yield

            for _ in tqdm(generator(), desc='searching for max_probability with unlimited greedy method!'):
                next = self.update_greedy_once()
                if get_traj:
                    traj_t.append(self.get_permutation())
                    overlap_t.append(self.overlap)
                else:
                    traj_t = [self.get_permutation()]
                    overlap_t = [self.overlap]
        else:
            for _ in tqdm(range(n_steps),
                          desc=f'searching for max_probability with limited(n_steps={n_steps}) greedy method'):
                self.update_greedy_once()
                if get_traj:
                    traj_t.append(self.get_permutation())
                    overlap_t.append(self.overlap)
                else:
                    traj_t = [self.get_permutation()]
                    overlap_t = [self.overlap]
        return traj_t, overlap_t

    def update_full(self):
        '''
        Using this function with caution, the complexity of computation is O(self.N!)
        :return: None
        '''
        Perm = all_perm(list(range(self.N)))
        self.overlap = 0.
        for i in tqdm(range(len(Perm)), desc='searching for max_probability with full method!'):
            new_overlap = self.get_overlap(permutation=Perm[i])
            if new_overlap > self.overlap:
                self.overlap = new_overlap
                self.permutation = Perm[i]

        return self.get_permutation(), self.overlap


class ProbGenerator:
    # generator of probability matrix
    def __init__(self, N, coherence):
        self.N = N
        self.coherence = coherence

    def sample(self, K):
        seq_prob = []
        for _ in tqdm(range(K), desc=f'Generating Similarity Matrix.'):
            Noncoherencepart = np.random.rand(self.N, self.N)
            coherencepart = np.random.rand(self.N)
            sample = (1 - self.coherence) * Noncoherencepart / Noncoherencepart.sum() + np.diag(
                self.coherence * coherencepart / coherencepart.sum())
            np.random.shuffle(sample)
            seq_prob.append(sample)
        return seq_prob

    def set_N(self, N):
        self.N = N

    def set_coherence(self, coherence):
        self.coherence = coherence
