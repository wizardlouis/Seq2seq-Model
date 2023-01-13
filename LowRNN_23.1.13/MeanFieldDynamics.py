# -*- codeing = utf-8 -*-
# @time:2022/4/17 下午3:03
# Author:Xuewen Shen
# @File:MeanFieldDynamics.py
# @Software:PyCharm

import torch
import torch.nn as nn
import numpy as np
import random
import math
from rw import *
import torch.optim as optim
from network import reparameterlizedRNN_sample, Seq2SeqModel_simp, Batch_onehot
from torch.autograd import Function
import copy

default_MF_param_n2d1r3p6 = dict(
    N_pop=6, N_r=3, N_I=1
)

def _merge(a, ac, c):
    p = a.shape[0]
    R = a.shape[1]
    I = c.shape[1]
    r = torch.zeros(p, R + I, R + I).to(a.device)
    r[:, :R, :R] = a
    r[:, :R, R:] = ac
    r[:, R:, :R] = torch.einsum("pij->pji", ac)
    r[:, R:, R:] = c
    return r


class tanh_div_x_function(Function):
    @staticmethod
    def forward(ctx, x):
        tanh_x = x.tanh()
        ctx.save_for_backward(x, tanh_x)
        return torch.where(x.abs() < 1e-4, 1 - x.pow(2) / 3, tanh_x / x)

    @staticmethod
    def backward(ctx, grad_output):
        x, tanh_x = ctx.saved_tensors
        prime = torch.where(x.abs() < 1e-4, -2 / 3 * x + 8 / 15 * x.pow(3), (x - tanh_x) / x.pow(2) - tanh_x.pow(2) / x)
        return prime * grad_output


def rho(x):
    return tanh_div_x_function.apply(x)


def P_full(P: torch.Tensor, N_I: int) -> torch.Tensor:
    '''
    :param P:Projection matrix of coordination with rank N_R
    :param N_I: input dimensions
    :return: full projection matrix with N_R+N_I dimensions with last N_I ranks of diagonal elements of 1.
    '''
    N_R = P.shape[0]
    P_F = torch.zeros(N_R + N_I, N_R + N_I)
    P_F[:N_R, :N_R] = P
    for n in range(N_I):
        P_F[N_R + n, N_R + n] = 1.
    return P_F


def get_Overlap(weights, means, covariances, reduction_p=True):
    Overlap_p = torch.einsum('pu,pv->puv', means, means) + covariances
    if reduction_p:
        Overlap = torch.einsum('p,puv->uv', weights, Overlap_p)
        return Overlap
    else:
        return Overlap_p

def GramSchemidt(P:torch.tensor):
    # P is an matrix of shape [K,N]
    assert P.dim()==2
    Q=torch.zeros_like(P)
    for r in range(P.shape[0]):
        if r==0:
            Q[r]=P[r]/P[r].norm()
        else:
            P_r_ortho=P[r]-sum([Q[j]*(P[r]*Q[j]).sum() for j in range(0,r)])
            Q[r]=P_r_ortho/P_r_ortho.norm()
    return Q

#A set of gaussian populations with Statistical variables
class Statistics:
    def __init__(self, hyper, weights, means, covariances):
        self.hyper = hyper
        N_pop = self.hyper.N_pop
        N_F = self.hyper.N_I + 2 * self.hyper.N_R + self.hyper.N_O
        assert weights.shape == (N_pop,)
        assert means.shape == (N_pop, N_F)
        assert covariances.shape == (N_pop, N_F, N_F)
        self.N_pop = N_pop
        self.N_I = self.hyper.N_I
        self.N_R = self.hyper.N_R
        self.N_O = self.hyper.N_O
        self.N_F = N_F
        self.weights = weights.cpu()
        self.means = means.cpu()
        self.covariances = covariances.cpu()

    @staticmethod
    def load_from_reparameterlizedRNN(model):
        return Statistics(table(N_pop=model.N_pop, N_I=model.N_I, N_R=model.N_R, N_O=model.N_O),
                          model.get_alpha_p().detach(), model.get_mu().detach(), model.get_cov().detach())

    def normalize_U(self, weights, means, covariances):
        U_index = list(range(self.N_I, self.N_I + self.N_R))
        V_index = list(range(self.N_I + self.N_R, self.N_I + 2 * self.N_R))
        Overlap_diag = get_Overlap(weights, means, covariances, reduction_p=True).diag()
        Norm_U = Overlap_diag[U_index].sqrt()
        mu = self.means.clone()
        C = torch.linalg.cholesky(self.covariances)
        mu[:, U_index] /= Norm_U
        mu[:, V_index] *= Norm_U
        C[:, U_index] /= Norm_U.unsqueeze(dim=1)
        C[:, V_index] *= Norm_U.unsqueeze(dim=1)
        return Statistics(self.hyper, self.weights, mu, torch.einsum('pxz,pyz->pxy', C, C))

    def transform_P(self, P):
        # transformation under Projection matrix P, applied after normalization_U, if U are not normalized, the result will not be orthogonal
        U_index = list(range(self.N_I, self.N_I + self.N_R))
        V_index = list(range(self.N_I + self.N_R, self.N_I + 2 * self.N_R))
        assert P.shape == (self.N_R, self.N_R)
        mu = self.means.clone()
        C = torch.linalg.cholesky(self.covariances)
        mu[:, U_index] = torch.einsum('pv,uv->pu', mu[:, U_index], P)
        mu[:, V_index] = torch.einsum('pv,uv->pu', mu[:, V_index], P)
        C[:, U_index] = torch.einsum('pvj,uv->puj', C[:, U_index], P)
        C[:, V_index] = torch.einsum('pvj,uv->puj', C[:, V_index], P)
        return Statistics(self.hyper, self.weights, mu, torch.einsum('pxz,pyz->pxy', C, C))


class MFDStatistics:
    default_hyper = table(N_I=1, N_R=3, N_O=2)

    def __init__(self, hyper: table, weights, means, covariances, no_z=False, device='cpu'):
        '''
        :param hyper: a table of model statistics containing N_I,N_R,N_O and weights, means, covariances
        :param device:
        '''
        self.device = device
        self.hyper = hyper
        self.hyper.I_index = list(range(self.hyper.N_I))
        self.hyper.U_index = list(range(self.hyper.N_I, self.hyper.N_I + self.hyper.N_R))
        self.hyper.V_index = list(range(self.hyper.N_I + self.hyper.N_R, self.hyper.N_I + 2 * self.hyper.N_R))
        self.Statistics = table(alpha_p=weights, mu=means, cov=covariances)
        self.no_z = no_z
        if not self.no_z:
            self.hyper.N_F = self.hyper.N_I + 2 * self.hyper.N_R + self.hyper.N_O
            self.hyper.O_index = list(range(self.hyper.N_F - self.hyper.N_O, self.hyper.N_F))
        else:
            self.hyper.N_F = self.hyper.N_I + 2 * self.hyper.N_R
        self.reduction = False
        self.exclude_DI = False

    # duprecateed
    def set_Statistics(self, Table: table):
        '''
        setup model Statistics: weights, means and covariances
        :param Table: a table of model statistics containing weights, means ,covariances information
        :return:
        '''
        self.Statistics = Table

    def toLowRNN(self, n_samples, add_param=dict(), device='cpu'):
        '''
        generate simplified low rank RNN from Statistics
        :param n_samples: sample neurons
        :param add_param: additional hyperparameters
        :param device: device to put on
        :return: Seq2SeqModel_simp
        '''
        loading, labels = GaussMixResample(n_samples, tn(self.Statistics.alpha_p), tn(self.Statistics.mu),
                                           tn(self.Statistics.cov))
        loadingvector = LoadingVector(loading[:, :self.hyper.N_I],
                                      loading[:, self.hyper.N_I + self.hyper.N_R:self.hyper.N_I + 2 * self.hyper.N_R],
                                      loading[:, self.hyper.N_I:self.hyper.N_I + self.hyper.N_R],
                                      W=loading[:, -self.hyper.N_O:])
        return Seq2SeqModel_simp.from_loading(loadingvector, add_param=add_param, device=device)

    def compute_Statistics_reduction(self):
        '''
        generate reduced model structure from statistics which is designed to fit collective variable dynamics including:
        alpha_p,mu_I,mu_m,mu_n,mu_O,sigma_nI,sigma_nm,sigma_mI,sigma_I,sigma_m,sigma_n,sigma_OI,sigma_Om,sigma
        :param no_z: no z implies readout dimension ignored
        :return:
        '''
        h = self.hyper
        s = self.Statistics
        alpha_p = s.alpha_p
        mu_I = s.mu[:, h.I_index]
        mu_m = s.mu[:, h.U_index]
        mu_n = s.mu[:, h.V_index]
        sigma_I = s.cov[:, h.I_index][:, :, h.I_index]
        sigma_m = s.cov[:, h.U_index][:, :, h.U_index]
        sigma_n = s.cov[:, h.V_index][:, :, h.V_index]
        sigma_nI = s.cov[:, h.V_index][:, :, h.I_index]
        sigma_nm = s.cov[:, h.V_index][:, :, h.U_index]
        sigma_mI = s.cov[:, h.U_index][:, :, h.I_index]
        if self.no_z:
            self.Statistics_reduction = table(alpha_p=alpha_p, mu_I=mu_I, mu_m=mu_m, mu_n=mu_n, sigma_I=sigma_I,
                                              sigma_m=sigma_m, sigma_n=sigma_n, sigma_nI=sigma_nI, sigma_nm=sigma_nm,
                                              sigma_mI=sigma_mI)
        else:
            mu_O = s.mu[:, h.O_index]
            sigma_O = s.cov[:, h.O_index][:, :, h.O_index]
            sigma_OI = s.cov[:, h.O_index][:, :, h.I_index]
            sigma_Om = s.cov[:, h.O_index][:, :, h.U_index]
            self.Statistics_reduction = table(alpha_p=alpha_p, mu_I=mu_I, mu_m=mu_m, mu_n=mu_n, sigma_I=sigma_I,
                                              sigma_m=sigma_m, sigma_n=sigma_n, sigma_nI=sigma_nI, sigma_nm=sigma_nm,
                                              sigma_mI=sigma_mI, mu_O=mu_O, sigma_O=sigma_O, sigma_OI=sigma_OI,
                                              sigma_Om=sigma_Om)
        self.reduction = True

    def compute_Direct_Input(self):
        assert self.reduction
        h = self.hyper
        s = self.Statistics_reduction
        mu_mI = torch.einsum('pu,pi->pui', s.mu_m, s.mu_I)
        O_mI = torch.einsum('p,pui->ui', s.alpha_p, mu_mI + s.sigma_mI)
        O_m = torch.einsum('p,pu->u', s.alpha_p, s.mu_m ** 2 + s.sigma_m[:, list(range(h.N_R)), list(range(h.N_R))])
        D_mI = (O_mI.T / O_m).T
        return D_mI

    def exclude_Direct_Input(self):
        assert self.reduction
        h = self.hyper
        s = self.Statistics_reduction
        D_mI = self.compute_Direct_Input()
        new_mu_I = s.mu_I - torch.einsum('ui,pu->pi', D_mI, s.mu_m)
        new_sigma_I = s.sigma_I - torch.einsum('ui,puj->pij', D_mI, s.sigma_mI) \
                      - torch.einsum('uj,pui->pij', D_mI, s.sigma_mI) \
                      + torch.einsum('ui,vj,puv->pij', D_mI, D_mI, s.sigma_m)
        new_sigma_mI = s.sigma_mI - torch.einsum('ui,puv->pvi', D_mI, s.sigma_m)
        new_sigma_nI = s.sigma_nI - torch.einsum('ui,pvu->pvi', D_mI, s.sigma_nm)
        # Statistics under z/no_z conditions
        if self.no_z:
            self.Statistics_reduction_exclude = table(alpha_p=s.alpha_p, mu_I=new_mu_I, mu_m=s.mu_m, mu_n=s.mu_n,
                                                      sigma_I=new_sigma_I, sigma_m=s.sigma_m, sigma_n=s.sigma_n,
                                                      sigma_nI=new_sigma_nI, sigma_nm=s.sigma_nm, sigma_mI=new_sigma_mI,
                                                      D_mI=D_mI)
        else:
            new_sigma_OI = s.sigma_OI - torch.einsum('ui,pou->poi', D_mI, s.sigma_Om)
            self.Statistics_reduction_exclude = table(alpha_p=s.alpha_p, mu_I=new_mu_I, mu_m=s.mu_m, mu_n=s.mu_n,
                                                      sigma_I=new_sigma_I, sigma_m=s.sigma_m, sigma_n=s.sigma_n,
                                                      sigma_nI=new_sigma_nI, sigma_nm=s.sigma_nm, sigma_mI=new_sigma_mI,
                                                      mu_O=s.mu_O, sigma_O=s.sigma_O, sigma_OI=new_sigma_OI,
                                                      sigma_Om=s.sigma_Om, D_mI=D_mI)
        self.exclude_DI = True

    def normalization(self):
        assert self.reduction and self.exclude_DI
        h = self.hyper
        s = self.Statistics_reduction_exclude
        norm_I = torch.einsum('p,pi->i', s.alpha_p,
                              s.mu_I ** 2 + s.sigma_I[:, list(range(h.N_I)), list(range(h.N_I))]).sqrt()
        inv_norm_I = 1. / norm_I
        norm_m = torch.einsum('p,pu->u', s.alpha_p,
                              s.mu_m ** 2 + s.sigma_m[:, list(range(h.N_R)), list(range(h.N_R))]).sqrt()
        inv_norm_m = 1. / norm_m
        new_mu_I = s.mu_I * inv_norm_I
        new_mu_m = s.mu_m * inv_norm_m
        new_mu_n = s.mu_n * norm_m
        new_sigma_I = torch.einsum('i,pij,j->pij', inv_norm_I, s.sigma_I, inv_norm_I)
        new_sigma_m = torch.einsum('u,puv,v->puv', inv_norm_m, s.sigma_m, inv_norm_m)
        new_sigma_n = torch.einsum('u,puv,v->puv', norm_m, s.sigma_n, norm_m)
        new_sigma_mI = torch.einsum('u,pui,i->pui', inv_norm_m, s.sigma_mI, inv_norm_I)
        new_sigma_nI = torch.einsum('v,pvi,i->pvi', norm_m, s.sigma_nI, inv_norm_I)
        new_sigma_nm = torch.einsum('v,pvu,u->pvu', norm_m, s.sigma_nm, inv_norm_m)
        new_D_mI = torch.einsum('u,ui', norm_m, s.D_mI)
        new_D_I = norm_I
        # Statistics under z/no_z conditions
        if self.no_z:
            self.Statistics_reduction_exclude_normalized = table(alpha_p=s.alpha_p, mu_I=new_mu_I, mu_m=new_mu_m,
                                                                 mu_n=new_mu_n, sigma_I=new_sigma_I,
                                                                 sigma_m=new_sigma_m, sigma_n=new_sigma_n,
                                                                 sigma_nI=new_sigma_nI, sigma_nm=new_sigma_nm,
                                                                 sigma_mI=new_sigma_mI, D_mI=new_D_mI, D_I=new_D_I)
        else:
            new_sigma_OI = torch.einsum('poi,i->poi', s.sigma_OI, inv_norm_I)
            new_sigma_Om = torch.einsum('pou,u->pou', s.sigma_Om, inv_norm_m)
            self.Statistics_reduction_exclude_normalized = table(alpha_p=s.alpha_p, mu_I=new_mu_I, mu_m=new_mu_m,
                                                                 mu_n=new_mu_n, sigma_I=new_sigma_I,
                                                                 sigma_m=new_sigma_m, sigma_n=new_sigma_n,
                                                                 sigma_nI=new_sigma_nI, sigma_nm=new_sigma_nm,
                                                                 sigma_mI=new_sigma_mI, mu_O=s.mu_O, sigma_O=s.sigma_O,
                                                                 sigma_OI=new_sigma_OI, sigma_Om=new_sigma_Om,
                                                                 D_mI=new_D_mI, D_I=new_D_I)

    def standard_preprocessing(self):
        self.compute_Statistics_reduction()
        self.exclude_Direct_Input()
        self.normalization()
        return self.Statistics_reduction_exclude_normalized

    def transform_P(self, P):
        assert P.shape == torch.Size(self.hyper.N_R, self.hyper.N_R)

    @staticmethod
    def from_reparameterlizedRNN(RNN, device='cpu'):
        # Extract Statistics from reparameterlizedRNN_sample
        assert type(RNN) == reparameterlizedRNN_sample
        hyper = table(N_I=RNN.N_I, N_R=RNN.N_R, N_O=RNN.N_O)
        # hyper = table(N_I=RNN.N_I, N_R=RNN.N_R, N_O=RNN.N_O, N_F=RNN.N_F, act_func=RNN.P['act_func'],
        #               Embedding=RNN.P['Embedding'].to(device), g_in=RNN.P['g_in'], g_rec=RNN.P['g_rec'], dt=RNN.P['dt'],
        #               tau=RNN.P['tau'])
        Statistics = MFDStatistics(hyper, RNN.get_alpha_p(), RNN.get_mu(), RNN.get_cov(), no_z=False, device=device)
        return Statistics


class MFD_simulator(nn.Module):
    randomsample = 1000
    default_param = dict(
        act_func='Tanh',
        N_I=1,
        N_R=3,
        N_O=2,
        N_pop=3,
        dt=20,
        tau=100,
        g_rec=0.03
    )

    def __init__(self, Sta: Statistics, act_func='Tanh', tau=100):
        super(MFD_simulator, self).__init__()
        self.hyper = Sta.hyper
        self.Statistics = Sta
        self.act_func = eval('torch.nn.' + act_func + '()')
        self.tau = tau
        self.noise_sample = torch.randn(MFD_simulator.randomsample)
        self.init_Mask()
        self.I_index = list(range(self.hyper.N_I))
        self.U_index = list(range(self.hyper.N_I, self.hyper.N_I + self.hyper.N_R))
        self.V_index = list(range(self.hyper.N_I + self.hyper.N_R, self.hyper.N_I + 2 * self.hyper.N_R))
        self.O_index = list(
            range(self.hyper.N_I + 2 * self.hyper.N_R, self.hyper.N_I + 2 * self.hyper.N_R + self.hyper.N_O))

    def init_Mask(self, Mask_G=None, Mask_mu=None, Mask_cov=None):
        if Mask_G is None:
            self.Mask_G = torch.ones(self.Statistics.weights.shape)
        else:
            assert Mask_G.shape == self.Statistics.weights.shape
            self.Mask_G = Mask_G.detach().cpu()
        if Mask_mu is None:
            self.Mask_mu = torch.ones(self.Statistics.means.shape)
        else:
            assert Mask_mu.shape == self.Statistics.means.shape
            self.Mask_mu = Mask_mu.cpu()
        if Mask_cov is None:
            self.Mask_cov = torch.ones(self.Statistics.covariances.shape)
        else:
            assert Mask_cov.shape == self.Statistics.covariances.shape
            self.Mask_cov = Mask_cov.cpu()

    def get_alpha_p(self, device='cpu'):
        return (self.Statistics.weights * self.Mask_G).to(device)

    def get_mu(self, device='cpu'):
        return (self.Statistics.means * self.Mask_mu).to(device)

    def get_cov(self, device='cpu'):
        return (self.Statistics.covariances * self.Mask_cov).to(device)

    def mu_sigma_from_z(self, z, kappa_I, device='cpu'):
        # z:[Batch_Size,T,n_dim] or [Batch_Size,n_dim]
        z_ = torch.cat([kappa_I, z], dim=-1).to(device)
        mu = torch.einsum('...d,pd->...p', z_, self.get_mu(device=device)[:, self.I_index + self.U_index])
        sigma = torch.einsum('...u,puv,...v->...p', z_,
                             self.get_cov(device=device)[:, self.I_index + self.U_index][:, :,
                             self.I_index + self.U_index], z_)
        return mu, sigma

    @staticmethod
    def MT_mean_f(f, mu, sigma, N=4096, device='cpu'):
        # mu: [batch, T, p] or [batch, p]
        # sigma: [batch, T, p] or [batch.p]
        assert mu.shape == sigma.shape
        shape = mu.shape
        mu = mu.to(device)
        sigma = sigma.to(device)
        x = torch.randn(*shape, N).to(device)
        x = x * sigma.unsqueeze(dim=-1) + mu.unsqueeze(dim=-1)
        return f(x).mean(-1)

    @torch.no_grad()
    def analyse_z_rec_source(self, z, kappa_I, u, N=4096, device='cpu'):
        # z [Batch_Size,T,n_dim]; I [Batch_Size,T,I_dim]; u [Batch_size,T,I_dim]
        Batch_Size = z.shape[0]
        assert z.dim() == 3
        assert kappa_I.dim() == 3
        assert u.dim() == 3

        z = z.to(device)
        kappa_I = kappa_I.to(device)
        weights = self.get_alpha_p(device=device)
        means = self.get_mu(device=device)
        covariances = self.get_cov(device=device)

        mu_z, sigma_z_2 = self.mu_sigma_from_z(z, kappa_I, device=device)
        sigma_z = (1e-30 + sigma_z_2).pow(0.5)
        rho_x_p = self.MT_mean_f(lambda x: rho(x), mu_z, sigma_z, N=N, device=device)  # [batch, T, p]
        gain_x_p = self.MT_mean_f(lambda x: 1 - x.tanh().pow(2), mu_z, sigma_z, N=N, device=device)  # [batch, T, p]
        phi_x_p = self.MT_mean_f(lambda x: x.tanh(), mu_z, sigma_z, N=N, device=device)  # [bacth, T, p]

        mean_func = table(rho=rho_x_p, gain=gain_x_p, phi=phi_x_p)

        mu_n = means[:, self.V_index]
        mu_nm = torch.einsum('pn,pm->pnm', means[:, self.V_index], means[:, self.U_index])
        mu_nI = torch.einsum('pn,pi->pni', means[:, self.V_index], means[:, self.I_index])
        sigma_nm = covariances[:, self.V_index][:, :, self.U_index]
        sigma_nI = covariances[:, self.V_index][:, :, self.I_index]

        # From each time point, how we could extract contribution
        # from axis #4 to axis #3 in population #2 at time step #1 in trial #0
        from_z_mean = torch.einsum("p,btp,pnm,btm->btpnm", weights, rho_x_p, mu_nm, z)
        from_z_std = torch.einsum("p,btp,pnm,btm->btpnm", weights, gain_x_p, sigma_nm, z)
        from_I_mean = torch.einsum("p,btp,pni,bti->btpni", weights, rho_x_p, mu_nI, kappa_I)
        from_I_std = torch.einsum("p,btp,pni,bti->btpni", weights, gain_x_p, sigma_nI, kappa_I)
        from_mean = torch.einsum("p,btp,pn->btpn", weights, phi_x_p, mu_n)

        # get approximated contribution
        z_rec_p = (from_z_mean + from_z_std).sum(-1) + (from_I_mean + from_I_std).sum(-1)
        z_rec = torch.einsum("btpn->btn", z_rec_p)
        velocity = z_rec - z
        approx = table(z_rec=z_rec, z_rec_p=z_rec_p, velocity=velocity)

        # get correct contribution
        z_rec_p = from_mean + from_z_std.sum(-1) + from_I_std.sum(-1)
        z_rec = torch.einsum("btpn->btn", z_rec_p)
        velocity = z_rec - z
        correct = table(z=z, kappa_I=kappa_I, z_rec=z_rec, z_rec_p=z_rec_p, velocity=velocity)

        # get effective connection from axes
        psi_from_z_mean = torch.einsum("p,btp,pnm->btpnm", weights, rho_x_p, mu_nm)
        psi_from_z_std = torch.einsum("p,btp,pnm->btpnm", weights, gain_x_p, sigma_nm)
        psi_from_I_mean = torch.einsum("p,btp,pni->btpni", weights, rho_x_p, mu_nI)
        psi_from_I_std = torch.einsum("p,btp,pni->btpni", weights, gain_x_p, sigma_nI)

        value_table = table(from_z_mean=from_z_mean, from_z_std=from_z_std, from_I_mean=from_I_mean,
                            from_I_std=from_I_std, from_mean=from_mean)
        coefficient_table = table(correct=correct, approx=approx, from_z_mean=psi_from_z_mean,
                                  from_z_std=psi_from_z_std, from_I_mean=psi_from_I_mean, from_I_std=psi_from_I_std)

        return table(correct=correct, approx=approx, value=value_table, coefficient=coefficient_table,
                     mean_func=mean_func)

    def gradient(self, z, kappa_I, u, N=4096, device='cpu'):
        assert z.dim() == 2
        assert kappa_I.dim() == 2
        assert u.dim() == 2

        z = z.to(device)
        kappa_I = kappa_I.to(device)

        mu_z, sigma_z_2 = self.mu_sigma_from_z(z, kappa_I, device=device)
        sigma_z = (1e-30 + sigma_z_2).pow(0.5)
        phi_x_p = self.MT_mean_f(lambda x: x.tanh(), mu_z, sigma_z, N=N, device=device)  # [Batch_Size,p]
        gain_x_p = self.MT_mean_f(lambda x: 1 - x.tanh().pow(2), mu_z, sigma_z, N=N,
                                  device=device)  # [Batch_Size, p]

        z_ = torch.cat([kappa_I, z], dim=-1)
        z_rec_from_mean = torch.einsum('p,bp,pn->bn', self.get_alpha_p(device=device), phi_x_p,
                                       self.get_mu(device=device)[:, self.V_index])
        z_rec_from_std = torch.einsum('p,bp,pnm,bm->bn', self.get_alpha_p(device=device), gain_x_p,
                                      self.get_cov(device=device)[:, self.V_index][:, :, self.I_index + self.U_index],
                                      z_)

        z_rec = z_rec_from_mean + z_rec_from_std
        I_rec = u

        return z_rec - z, I_rec - kappa_I

    @torch.no_grad()
    def forward(self, u, init_z=None, init_kappa_I=None, N=4096, with_kappa_I=False, dt=20, device='cpu'):
        assert u.dim() == 3
        Batch_Size, T = u.shape[:2]
        z_t = torch.zeros(Batch_Size, T + 1, self.hyper.N_R).to(device)
        if init_z is not None:
            assert z_t.shape == (Batch_Size, self.hyper.N_R)
            z_t[:, 0] = init_z.to(device)
        kappa_I_t = torch.zeros(Batch_Size, T + 1, self.hyper.N_I).to(device)
        if init_kappa_I is not None:
            assert init_kappa_I.shape == (Batch_Size, self.hyper.N_I)
            kappa_I_t[:, 0] = init_kappa_I.to(device)
        for t in range(T):
            g_z, g_I = self.gradient(z_t[:, t], kappa_I_t[:, t], u[:, t], N=N, device=device)
            z_t[:, t + 1] = z_t[:, t] + g_z * dt / self.tau
            kappa_I_t[:, t + 1] = kappa_I_t[:, t] + g_I * dt / self.tau
        if not with_kappa_I:
            return z_t
        else:
            return z_t, kappa_I_t


# After comfirming MF dynamics, try dealing with mask problems.
class MaskDict:
    truncate_upper_bound = 1

    def __init__(self, Statistic_Data: table):
        self.keys = Statistic_Data.keys
        self.Mask = table()
        for key in self.keys:
            self.Mask[key] = torch.ones_like(Statistic_Data.__getattribute__(key), dtype=torch.long)

    def keys(self):
        return self.keys

    def block(self, key, *loc):
        assert self.legal(key, *loc)
        self.Mask[key][loc] = 1

    def activate(self, key, *loc):
        assert self.legal(key, *loc)
        self.Mask[key][loc] = 1

    def legal(self, key, *loc):
        pass

    def truncate(self, threshold):
        assert threshold < MaskDict.truncate_upper_bound


class MFDmodel(nn.Module):
    def __init__(self, Statistics: MFDStatistics):
        super(MFDmodel, self).__init__()
        self.device = Statistics.device
        self.hyper = Statistics.hyper
        self.compute_connection(Statistics)

    def set_connection(self, Statistics):
        self.connection = table(alpha_p=Statistics.alpha_p.to(self.device), mu=Statistics.mu.to(self.device),
                                cov=Statistics.cov.to(self.device))

    def sample(self, n_samples, add_param=dict(), device='cpu'):
        loading, labels = GaussMixResample(n_samples, tn(self.connection.alpha_p), tn(self.connection.mu),
                                           tn(self.connection.cov))
        loadingvector = LoadingVector(loading[:, :self.hyper.N_I],
                                      loading[:, self.hyper.N_I + self.hyper.N_R:self.hyper.N_I + 2 * self.hyper.N_R],
                                      loading[:, self.hyper.N_I:self.hyper.N_I + self.hyper.N_R],
                                      W=loading[:, -self.hyper.N_O:])
        return Seq2SeqModel_simp.from_loading(loadingvector, add_param=add_param, device=device)

    def compute_connection_reduced(self):
        alpha_p = self.connection.alpha_p.clone()

        self.connection_reduced = table(alpha_p=self.connection.alpha_p.clone(), )
