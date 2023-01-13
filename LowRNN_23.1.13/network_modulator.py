# -*- codeing = utf-8 -*-
# @time:2022/8/15 下午2:53
# Author:Xuewen Shen
# @File:network_modulator.py
# @Software:PyCharm
from curses.ascii import FF
from unicodedata import decimal
import torch
import torch.nn as nn
import math
from network import reparameterlizedRNN_sample


def uns(x, dim=0):
    return x.unsqueeze(dim=dim)


def rotate_2d(angle):
    return torch.tensor([
        [math.cos(angle), math.sin(angle)],
        [-math.sin(angle), math.cos(angle)]
    ])


def tensor_rotates_2d(angle_list):
    cos = angle_list.cos()
    sin = angle_list.sin()
    return torch.cat([uns(cos), uns(sin), uns(-sin), uns(cos)], dim=0).reshape(2, 2, -1).permute([2, 0, 1])


def rotation_sym_group(n):
    return [rotate_2d(2 * math.pi / n * i) for i in range(n)]


C_6_group = rotation_sym_group(6)
C_3_group = rotation_sym_group(3)


def mesh_pos(axes_lim):
    n_axes = len(axes_lim)
    M = torch.zeros(n_axes, *axes_lim, dtype=torch.int)
    for axis_idx in range(n_axes):
        M_ = M.transpose(axis_idx + 1, -1)
        M_[axis_idx] = torch.arange(axes_lim[axis_idx], dtype=torch.int)
        M = M_.transpose(axis_idx + 1, -1)
    return M.permute(*list(range(1, n_axes + 1)), 0)


def gen_rotation_symmetry_pops(groups, rank_labels):
    '''
    :param groups: a list of groups to generate pops from
    :param rank_labels: labeling which group to attached with ranks in network, such as [0,1,0,0,1,0] attached dim [0,2,3,5] to gourp_0 and dim [1,4] to group_1
    :return: a tensor of rotations :[n_1,n_2...n_k,N_rank,n_dim,n_dim]
    '''
    N_rank = len(rank_labels)

    max_idx_groups = [len(group) for group in groups]
    M = mesh_pos(max_idx_groups)
    M_flatten = M.reshape(-1, M.shape[-1])
    rotation_sym_pop = torch.cat([torch.cat([groups[rank_labels[r]][M_flatten[p, rank_labels[r]]].unsqueeze(
        dim=0) for r in range(N_rank)], dim=0).unsqueeze(dim=0) for p in range(M_flatten.shape[0])], dim=0)
    return rotation_sym_pop


# Apply a sequence of rotations of [n_dim,n_dim] on a data tensor and return a list of rotated matrices
# rotate matrix M by rotation projection in rotations on rotate_dim, and return a list of rotated matrices
def sym_rotate(M, rotations, rotate_dim=1, device='cpu'):
    '''
    :param M: Data tensor
    :param rotations: a list of rotations applied on Data, in which element rotation is of shape [n_dim,n_dim]
    :param rotate_dim: rotated dimension of M
    :param device: device to project
    :return:
    '''
    n_dim = rotations[0].shape[0]
    assert rotations[0].shape == (n_dim, n_dim)
    n_channels = int(M.shape[rotate_dim] / n_dim)
    M_0 = M.to(device).transpose(0, rotate_dim)
    M_n_dim = M_0.reshape(n_channels, n_dim, *M_0.shape[1:])
    M_n_dim_rotate = [torch.einsum('xy,cy...->cx...', rotation.to(device), M_n_dim.to(device)) for rotation in
                      rotations]
    M_rotate = [Matrix.reshape(n_dim * n_channels, *Matrix.shape[2:]).transpose(0, rotate_dim) for Matrix in
                M_n_dim_rotate]
    return M_rotate


# Apply a sequence of K rotations of shape [K,n_dim,n_dim] on rotate_dim of a data tensor with dimensionality K*n_dim
def sym_rotate_tensor(M, rotations, rotate_dim=1, device='cpu'):
    '''
    :param M: Data tensor
    :param rotations: a sequence of rotations with shape [K,n_dim,n_dim]
    :param rotate_dim: rotated dimension of M with length K*n_dim
    :param device: device to project
    :return:
    '''
    K, n_dim = rotations.shape[:2]
    assert rotations.shape == (K, n_dim, n_dim)
    assert M.shape[rotate_dim] == K * n_dim
    M_standard = M.transpose(rotate_dim, -1).to(device)
    M_rotate = torch.einsum('...kj,kij->...ki', M_standard.view(M_standard.shape[:-1], K, n_dim),
                            rotations.to(device)).view(M_standard.shape[:-1], K * n_dim).transpose(rotate_dim, -1)
    return M_rotate


class hierarchy_repar_N2_noz(reparameterlizedRNN_sample):
    pop_expand = [1, 2]
    dim_expand = 1
    # subspaces' indices [U1,U2,U3,V1,V2,V3,W1,W2]
    pop_mod1 = torch.ones([1, 6, 1, 1])
    pop_mod2 = torch.ones([2, 6, 1, 1])
    # Using G1xE gauge:{(E,E),(R,E)} rotations for (rank1,rank2) correlated subspaces
    pop_mod2[1, :, 0, 0] = torch.tensor([-1., 1., -1., -1., 1., -1.])
    pop_mod = [pop_mod1, pop_mod2]

    def __init__(self, N_pop, N_I, N_R, N_O, act_func, tau=100):
        super().__init__(N_pop, N_I, N_R, N_O, act_func, tau=tau)
        self.gen_O = torch.zeros(self.N_O, 2 * self.N_R)
        self.gen_O[0, 3] = 1.
        self.gen_O[1, 4] = 1.

    def mod_population(self, M, device='cpu'):
        if M.dim() == 2:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        elif M.dim() == 3:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand, -1),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        else:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(M.shape[0])],
                              dim=0).to(device)
        return M_mod

    def mod_Mask(self, Mask, device='cpu'):
        if Mask.dim() > 1:
            Mask_mod = [Mask[i:i + 1].to(device).repeat_interleave(self.pop_expand[i], dim=0) for i in
                        range(len(self.pop_expand))]
        else:
            Mask_mod = torch.cat(
                [Mask[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(Mask.shape[0])], dim=0).to(
                device)
        return Mask_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, device=device)

    def mu_mod(self, device='cpu'):
        mu_R = torch.cat(self.mod_population(self.mu_R, device=device), dim=0)
        mu_O = torch.einsum('or,pr->po', self.gen_O.to(device), mu_R)
        return [self.mu_I.to(device), mu_R, mu_O]

    def C_mod(self, device='cpu'):
        C_R = torch.cat(self.mod_population(self.C_R, device=device), dim=0)
        C_O = torch.einsum('or,pry->poy', self.gen_O.to(device), C_R)
        return [self.C_I.to(device), C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_mu, device=device))

    def Mask_C_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_C, device=device))

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            sum(self.pop_expand), 1, self.N_F)
        return loadingvectors

    def ortho_loss(self, I=False, U=False, V=False, W=False, IU=False, IW=False, corr_only=False, d_I=1., d_R=0.9,
                   d_O=0., device='cpu'):
        def MSE0(x):
            L = nn.MSELoss(reduction='sum')
            return L(x, torch.zeros_like(x))

        Overlap = self.get_Overlap(reduction_p=True, device=device)

        # Overlap under hierarchical structure refer to overlap between subspaces rather than between axes
        dim_expand = self.dim_expand
        Overlap_hierarchy = Overlap.reshape(int(self.N_F / dim_expand), dim_expand, int(self.N_F / dim_expand),
                                            dim_expand).transpose(1, 2).det()

        # the number of subspaces by hierarchy
        N_I = int(self.N_I / dim_expand)
        N_R = int(self.N_R / dim_expand)
        N_O = int(self.N_O / dim_expand)

        # the hierarchical indices of subspaces
        I_index = list(range(N_I))
        U_index = list(range(N_I, N_I + N_R))
        V_index = list(range(N_I + N_R, N_I + 2 * N_R))
        O_index = list(range(N_I + 2 * N_R, N_I + 2 * N_R + N_O))

        # The I,U,V,W Overlap
        O_I = Overlap_hierarchy[I_index][:, I_index]
        O_I_nd = O_I - d_I * torch.diag(O_I.diag())
        O_U = Overlap_hierarchy[U_index][:, U_index]
        O_U_nd = O_U - d_R * torch.diag(O_U.diag())
        O_V = Overlap_hierarchy[V_index][:, V_index]
        O_V_nd = O_V - d_R * torch.diag(O_V.diag())
        O_W = Overlap_hierarchy[O_index][:, O_index]
        O_W_nd = O_W - d_O * torch.diag(O_W.diag())
        O_IU = Overlap_hierarchy[I_index][:, U_index]
        O_IW = Overlap_hierarchy[I_index][:, O_index]

        # compute Orthogonal loss with Masks
        keys = [I, U, V, W, IU, IW]
        if not corr_only:
            values = [O_I_nd, O_U_nd, O_V_nd, O_W_nd, O_IU, O_IW]
        else:
            # correlation only orthogonal loss, avoid diagonal amplitude decaying
            O_I_d = O_I.diag()
            O_U_d = O_U.diag()
            O_V_d = O_V.diag()
            O_W_d = O_W.diag()
            corr_I = O_I / torch.einsum('u,v->uv', O_I_d, O_I_d).sqrt()
            corr_I_nd = corr_I - torch.diag(corr_I.diag())
            corr_U = O_U / torch.einsum('u,v->uv', O_U_d, O_U_d).sqrt()
            corr_U_nd = corr_U - torch.diag(corr_U.diag())
            corr_V = O_V / torch.einsum('u,v->uv', O_V_d, O_V_d).sqrt()
            corr_V_nd = corr_V - torch.diag(corr_V.diag())
            corr_W = O_W / torch.einsum('u,v->uv', O_W_d, O_W_d).sqrt()
            corr_W_nd = corr_W - torch.diag(corr_W.diag())
            corr_IU = O_IU / torch.einsum('u,v->uv', O_I_d, O_U_d).sqrt()
            corr_IW = O_IW / torch.einsum('u,v->uv', O_I_d, O_W_d).sqrt()
            values = [corr_I_nd, corr_U_nd, corr_V_nd, corr_W_nd, corr_IU, corr_IW]
        Ortho_Loss = sum([float(keys[idx]) * MSE0(values[idx]) for idx in range(len(keys))])
        return Ortho_Loss


class hierarchy_repar_N2(reparameterlizedRNN_sample):
    pop_expand = [1, 2]
    dim_expand = 1
    # subspaces' indices [U1,U2,U3,V1,V2,V3,W1,W2]
    pop_mod1 = torch.ones([1, 8, 1, 1])
    pop_mod2 = torch.ones([2, 8, 1, 1])
    # Using G1xE gauge:{(E,E),(R,E)} rotations for (rank1,rank2) correlated subspaces
    pop_mod2[1, :, 0, 0] = torch.tensor([-1., 1., -1., -1., 1., -1., -1., 1.])
    pop_mod = [pop_mod1, pop_mod2]
    gen_O_from_R = False

    def __init__(self, N_pop, N_I, N_R, N_O, act_func, tau=100):
        super().__init__(N_pop, N_I, N_R, N_O, act_func, tau=tau)
        self.gen_I_from_R = False
        self.gen_O_from_R = False

    def gen_mu_from_R(self, mu_R, gen_L, gen_L_mu_bias, device='cpu'):
        mu = torch.einsum('iu,pu->pi', gen_L.to(device), mu_R.to(device)) + gen_L_mu_bias.to(device)
        return mu

    def gen_C_from_R(self, C_R, gen_L, gen_L_C_bias, device='cpu'):
        C = torch.einsum('iu,puv->piv', gen_L.to(device), C_R.to(device)) + gen_L_C_bias.to(device)
        return C

    def set_gen_I(self, gen_I=None, gen_I_mu_bias=None, gen_I_C_bias=None):
        self.gen_I_from_R = True
        # number of functional clusters (different with population numbers, a
        # cluster is a total of populations with the same role of function
        n_cluster = len(self.pop_expand)
        gen_I_shape = (n_cluster, self.N_I, 2 * self.N_R)
        gen_I_mu_bias_shape = (n_cluster, self.N_I)
        gen_I_C_bias_shape = (n_cluster, self.N_I, self.N_F)
        if gen_I is None:
            self.gen_I = nn.Parameter(torch.randn(gen_I_shape), requires_grad=True)
        else:
            assert gen_I.shape == gen_I_shape
            self.gen_I = nn.Parameter(gen_I, requires_grad=False)
        if gen_I_mu_bias is None:
            self.gen_I_mu_bias = nn.Parameter(torch.randn(gen_I_mu_bias_shape), requires_grad=True)
        else:
            assert gen_I_mu_bias.shape == gen_I_mu_bias_shape
            self.gen_I_mu_bias = nn.Parameter(gen_I_mu_bias, requires_grad=False)
        if gen_I_C_bias is None:
            self.gen_I_C_bias = nn.Parameter(torch.randn(gen_I_C_bias_shape), requires_grad=True)
        else:
            assert gen_I_C_bias.shape == gen_I_mu_bias_shape
            self.gen_I_C_bias = nn.Parameter(gen_I_C_bias, requires_grad=False)

    def set_gen_O(self, gen_O=None, gen_O_mu_bias=None, gen_O_C_bias=None):
        self.gen_I_from_R = True
        # number of functional clusters (different with population numbers, a
        # cluster is a total of populations with the same role of function
        n_cluster = len(self.pop_expand)
        gen_O_shape = (n_cluster, self.N_O, 2 * self.N_R)
        gen_O_mu_bias_shape = (n_cluster, self.N_O)
        gen_O_C_bias_shape = (n_cluster, self.N_O, self.N_F)
        if gen_O is None:
            self.gen_O = nn.Parameter(torch.randn(gen_O_shape), requires_grad=True)
        else:
            assert gen_O.shape == gen_O_shape
            self.gen_O = nn.Parameter(gen_O, requires_grad=False)
        if gen_O_mu_bias is None:
            self.gen_O_mu_bias = nn.Parameter(torch.randn(gen_O_mu_bias_shape), requires_grad=True)
        else:
            assert gen_O_mu_bias.shape == gen_O_mu_bias_shape
            self.gen_O_mu_bias = nn.Parameter(gen_O_mu_bias, requires_grad=False)
        if gen_O_C_bias is None:
            self.gen_O_C_bias = nn.Parameter(torch.randn(gen_O_C_bias_shape), requires_grad=True)
        else:
            assert gen_O_C_bias.shape == gen_O_mu_bias_shape
            self.gen_O_C_bias = nn.Parameter(gen_O_C_bias, requires_grad=False)

    def mod_population(self, M, device='cpu'):
        if M.dim() == 2:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        elif M.dim() == 3:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand, -1),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        else:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(M.shape[0])],
                              dim=0).to(device)
        return M_mod

    def mod_Mask(self, Mask, device='cpu'):
        if Mask.dim() > 1:
            Mask_mod = [Mask[i:i + 1].to(device).repeat_interleave(self.pop_expand[i], dim=0) for i in
                        range(len(self.pop_expand))]
        else:
            Mask_mod = torch.cat(
                [Mask[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(Mask.shape[0])], dim=0).to(
                device)
        return Mask_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, device=device)

    def mu_mod(self, device='cpu'):
        mu_RO = torch.cat([self.mu_R, self.mu_O], dim=1)
        # a list of clusters' Statistics
        mu_RO_ = self.mod_population(mu_RO, device=device)
        if self.gen_I_from_R:
            mu_I = [self.gen_mu_from_R(mu_RO_[i][:, :2 * self.N_R], self.gen_I[i], self.gen_I_mu_bias[i:i + 1],
                                       device=device) for i in range(len(self.pop_expand))]
        else:
            mu_I = self.mu_I.to(device)
        mu_RO_std = torch.cat(mu_RO_, dim=0)
        mu_I_std = torch.cat(mu_I, dim=0)
        mu = torch.cat([mu_I_std, mu_RO_std], dim=1)
        return [mu[:, self.I_index], mu[:, self.U_index + self.V_index], mu[:, self.O_index]]

    def C_mod(self, device='cpu'):
        C_RO = torch.cat([self.C_R, self.C_O], dim=1)
        C_RO_ = self.mod_population(C_RO, device=device)
        if self.gen_I_from_R:
            C_I = [
                self.gen_C_from_R(C_RO_[i][:, :2 * self.N_R], self.gen_I[i], self.gen_I_C_bias[i:i + 1], device=device)
                for i in range(len(self.pop_expand))]
        else:
            C_I = self.C_I.to(device)
        C_RO_std = torch.cat(C_RO_, dim=0)
        C_I_std = torch.cat(C_I, dim=0)
        C = torch.cat([C_I_std, C_RO_std], dim=1)
        return [C[:, self.I_index], C[:, self.U_index + self.V_index], C[:, self.O_index]]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_mu, device=device))

    def Mask_C_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_C, device=device))

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            2 * self.N_pop - 1, 1, self.N_F)
        return loadingvectors


# subspaces' indices [U1,U2,U3,V1,V2,V3,W1,W2]
N2_pop_mod1 = torch.ones([1, 8, 1, 1])
# Using G1xE gauge:{(E,E),(R,E)} rotations for (rank1,rank2) correlated subspaces
N2_pop_mod2 = torch.ones([2, 8, 1, 1])
N2_pop_mod2[1, :, 0, 0] = torch.tensor([-1, 1, -1., -1., 1., -1., -1., 1.])
N2_pop_mod = [N2_pop_mod1, N2_pop_mod2]


class hierarchy_repar_N2_normalI(reparameterlizedRNN_sample):
    pop_expand = [1, 2]
    dim_expand = 1

    def set_pop_mod(self, pop_mod):
        self.pop_mod = pop_mod

    def mod_population(self, M, device='cpu'):
        if M.dim() == 2:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        elif M.dim() == 3:
            M_mod = [torch.einsum('ry...,prxy->prx...',
                                  M[i].to(device).reshape(int(M.shape[1] / self.dim_expand), self.dim_expand, -1),
                                  self.pop_mod[i].to(device)).reshape(self.pop_mod[i].shape[0], *M.shape[1:]) for i in
                     range(len(self.pop_expand))]
        else:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(M.shape[0])],
                              dim=0).to(device)
        return M_mod

    def mod_Mask(self, Mask, device='cpu'):
        if Mask.dim() > 1:
            Mask_mod = [Mask[i:i + 1].to(device).repeat_interleave(self.pop_expand[i], dim=0) for i in
                        range(len(self.pop_expand))]
        else:
            Mask_mod = torch.cat(
                [Mask[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(Mask.shape[0])], dim=0).to(
                device)
        return Mask_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, device=device)

    def mu_mod(self, device='cpu'):
        mu_RO = torch.cat([self.mu_R, self.mu_O], dim=1)
        # a list of clusters' Statistics
        mu_RO_ = self.mod_population(mu_RO, device=device)
        mu_I = torch.cat(
            [self.mu_I[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(len(self.pop_expand))],
            dim=0).to(device)
        mu_RO_std = torch.cat(mu_RO_, dim=0)
        mu = torch.cat([mu_I, mu_RO_std], dim=1)
        return [mu[:, self.I_index], mu[:, self.U_index + self.V_index], mu[:, self.O_index]]

    def C_mod(self, device='cpu'):
        C_RO = torch.cat([self.C_R, self.C_O], dim=1)
        C_RO_ = self.mod_population(C_RO, device=device)
        C_I = torch.cat(
            [self.C_I[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0) for i in range(len(self.pop_expand))],
            dim=0).to(device)
        C_RO_std = torch.cat(C_RO_, dim=0)
        C = torch.cat([C_I, C_RO_std], dim=1)
        return [C[:, self.I_index], C[:, self.U_index + self.V_index], C[:, self.O_index]]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_mu, device=device))

    def Mask_C_mod(self, device='cpu'):
        return torch.cat(self.mod_Mask(self.Mask_C, device=device))

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(sum(self.pop_expand), reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            2 * self.N_pop - 1, 1, self.N_F)
        return loadingvectors


class hierarchy_repar_N6(reparameterlizedRNN_sample):
    dim_expand = 2
    pop_expand = 3
    rotation = [
        rotate_2d(0), rotate_2d(math.pi * 2 / 3), rotate_2d(math.pi * 4 / 3)
    ]

    # rotate
    def G_mod(self, device='cpu'):
        return self.G.to(device).repeat_interleave(hierarchy_repar_N6.pop_expand, dim=0)

    def mu_mod(self, device='cpu'):
        mu_I = torch.cat(sym_rotate(self.mu_I, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        mu_R = torch.cat(sym_rotate(self.mu_R, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        mu_O = torch.cat(sym_rotate(self.mu_O, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        return [mu_I, mu_R, mu_O]

    def C_mod(self, device='cpu'):
        C_I = torch.cat(sym_rotate(self.C_I, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        C_R = torch.cat(sym_rotate(self.C_R, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        C_O = torch.cat(sym_rotate(self.C_O, hierarchy_repar_N6.rotation, rotate_dim=1, device=device), dim=0)
        return [C_I, C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.Mask_G.to(device).repeat_interleave(hierarchy_repar_N6.pop_expand, dim=0)

    def Mask_mu_mod(self, device='cpu'):
        return self.Mask_mu.to(device).repeat_interleave(hierarchy_repar_N6.pop_expand, dim=0)

    def Mask_C_mod(self, device='cpu'):
        return self.Mask_C.to(device).repeat_interleave(hierarchy_repar_N6.pop_expand, dim=0)

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(self.N_pop * hierarchy_repar_N6.pop_expand,
                                         reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        '''
        :return: samples of shape (N_pop, n_samples, dynamical dimensions)
        '''
        return torch.randn(self.N_pop * hierarchy_repar_N6.pop_expand, reparameterlizedRNN_sample.randomsample,
                           self.N_F, device=device)

    def get_loading(self, device='cpu'):
        '''
        :return: loading vectors of multipopulations with shape (N_pop, n_samples, dynamical dimensions)
        '''
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            self.N_pop * hierarchy_repar_N6.pop_expand, 1, self.N_F)
        return loadingvectors

    def set_axes(self, device='cpu'):
        loadingvectors = self.get_loading(device=device)
        self._I = loadingvectors[:, :, self.I_index]
        self._U = loadingvectors[:, :, self.U_index]
        self._V = loadingvectors[:, :, self.V_index]
        self._O = loadingvectors[:, :, self.O_index]

    def ortho_loss(self, I=False, U=False, V=False, W=False, IU=False, IW=False, corr_only=False, d_I=1., d_R=0.9,
                   d_O=0., device='cpu'):
        def MSE0(x):
            L = nn.MSELoss(reduction='sum')
            return L(x, torch.zeros_like(x))

        Overlap = self.get_Overlap(reduction_p=True, device=device)

        # Overlap under hierarchical structure refer to overlap between subspaces rather than between axes
        dim_expand = hierarchy_repar_N6.dim_expand
        Overlap_hierarchy = Overlap.reshape(int(self.N_F / dim_expand), dim_expand, int(self.N_F / dim_expand),
                                            dim_expand).transpose(1, 2).det()

        # the number of subspaces by hierarchy
        N_I = int(self.N_I / dim_expand)
        N_R = int(self.N_R / dim_expand)
        N_O = int(self.N_O / dim_expand)

        # the hierarchical indices of subspaces
        I_index = list(range(N_I))
        U_index = list(range(N_I, N_I + N_R))
        V_index = list(range(N_I + N_R, N_I + 2 * N_R))
        O_index = list(range(N_I + 2 * N_R, N_I + 2 * N_R + N_O))

        # The I,U,V,W Overlap
        O_I = Overlap_hierarchy[I_index][:, I_index]
        O_I_nd = O_I - d_I * torch.diag(O_I.diag())
        O_U = Overlap_hierarchy[U_index][:, U_index]
        O_U_nd = O_U - d_R * torch.diag(O_U.diag())
        O_V = Overlap_hierarchy[V_index][:, V_index]
        O_V_nd = O_V - d_R * torch.diag(O_V.diag())
        O_W = Overlap_hierarchy[O_index][:, O_index]
        O_W_nd = O_W - d_O * torch.diag(O_W.diag())
        O_IU = Overlap_hierarchy[I_index][:, U_index]
        O_IW = Overlap_hierarchy[I_index][:, O_index]

        # compute Orthogonal loss with Masks
        keys = [I, U, V, W, IU, IW]
        if not corr_only:
            values = [O_I_nd, O_U_nd, O_V_nd, O_W_nd, O_IU, O_IW]
        else:
            # correlation only orthogonal loss, avoid diagonal amplitude decaying
            O_I_d = O_I.diag()
            O_U_d = O_U.diag()
            O_V_d = O_V.diag()
            O_W_d = O_W.diag()
            corr_I = O_I / torch.einsum('u,v->uv', O_I_d, O_I_d).sqrt()
            corr_I_nd = corr_I - torch.diag(corr_I.diag())
            corr_U = O_U / torch.einsum('u,v->uv', O_U_d, O_U_d).sqrt()
            corr_U_nd = corr_U - torch.diag(corr_U.diag())
            corr_V = O_V / torch.einsum('u,v->uv', O_V_d, O_V_d).sqrt()
            corr_V_nd = corr_V - torch.diag(corr_V.diag())
            corr_W = O_W / torch.einsum('u,v->uv', O_W_d, O_W_d).sqrt()
            corr_W_nd = corr_W - torch.diag(corr_W.diag())
            corr_IU = O_IU / torch.einsum('u,v->uv', O_I_d, O_U_d).sqrt()
            corr_IW = O_IW / torch.einsum('u,v->uv', O_I_d, O_W_d).sqrt()
            values = [corr_I_nd, corr_U_nd, corr_V_nd, corr_W_nd, corr_IU, corr_IW]
        Ortho_Loss = sum([float(keys[idx]) * MSE0(values[idx]) for idx in range(len(keys))])
        return Ortho_Loss


class hierarchy_repar(reparameterlizedRNN_sample):
    pop_expand = [3, 18]
    dim_expand = 2
    gen_I = 'default'
    no_z = True

    def set_no_z(self, no_z):
        assert (no_z or no_z == False)
        self.no_z = no_z

    def set_R_to_z(self, R_to_z, Mask_R_to_z):
        R_to_z_shape = (self.N_O, 2 * self.N_R)
        assert R_to_z.shape == R_to_z_shape
        assert Mask_R_to_z.shape == R_to_z_shape
        self.R_to_z = R_to_z
        self.Mask_R_to_z = Mask_R_to_z

    def set_pop_mod(self, pop_mod):
        # setup of population modulation for each clusters
        # Modulation of each population should be of shape:[n_pop,n_dim,dim_expand,dim_expand]
        assert len(pop_mod) == len(self.pop_expand)
        if self.no_z:
            n_subspace = int(2 * self.N_R / self.dim_expand)
        else:
            n_subspace = int((2 * self.N_R + self.N_O) / self.dim_expand)
        for p in range(len(pop_mod)):
            assert pop_mod[p].shape == (self.pop_expand[p], n_subspace, self.dim_expand, self.dim_expand)
        self.pop_mod = pop_mod
        self.total_pop = sum(self.pop_expand)

    def set_gen_I(self, gen_I):
        assert gen_I in ['default', 'repeat', 'R_to_I']
        self.gen_I = gen_I
        if self.gen_I == 'default':
            assert self.mu_I.shape == (sum(self.pop_expand), self.N_I)
            assert self.C_I.shape == (sum(self.pop_expand), self.N_I, self.N_F)
        elif self.gen_I == 'repeat':
            assert self.mu_I.shape == (len(self.pop_expand), self.N_I)
            assert self.C_I.shape == (len(self.pop_expand), self.N_I, self.N_F)
        else:
            pass

    def set_R_to_I(self, R_to_I, Mask_R_to_I):
        # setup of R to Input
        if self.gen_I == 'R_to_I':
            R_to_I_shape = (len(self.pop_expand), self.N_I, 2 * self.N_R)
            assert R_to_I.shape == R_to_I_shape
            assert Mask_R_to_I.shape == R_to_I_shape
            self.R_to_I = R_to_I
            self.Mask_R_to_I = Mask_R_to_I
        else:
            print("The 'gen_I' mode of network is not 'R_to_I', function 'set_R_to_I' not work!")

    def apply_gen_mu_I(self, mu_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert mu_R is not None
            mu_R = mu_R.to(device)
            R_to_I_Masked = self.R_to_I.to(device) * self.Mask_R_to_I.to(device)
            n_pops = [sum(self.pop_expand[:i]) for i in range(len(self.pop_expand) + 1)]
            return torch.cat([torch.einsum('ir,pr->pi', R_to_I_Masked[i], mu_R[n_pops[i]:n_pops[i + 1]])
                             for i in range(len(self.pop_expand))], dim=0)
        elif self.gen_I == 'repeat':
            return self.mod_population(self.mu_I, only_expand_pop=True, device=device)
        else:
            return self.mu_I.to(device)

    def apply_gen_C_I(self, C_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert C_R is not None
            C_R = C_R.to(device)
            R_to_I_Masked = self.R_to_I.to(device) * self.Mask_R_to_I.to(device)
            n_pops = [sum(self.pop_expand[:i]) for i in range(len(self.pop_expand) + 1)]
            return torch.cat([torch.einsum('ir,pry->piy', R_to_I_Masked[i], C_R[n_pops[i]:n_pops[i + 1]])
                             for i in range(len(self.pop_expand))], dim=0)
        elif self.gen_I == 'repeat':
            return self.mod_population(self.C_I, only_expand_pop=True, device=device)
        else:
            return self.C_I.to(device)

    def mod_dim(self, M, device='cpu'):
        return M.reshape(M.shape[0], int(M.shape[1] / self.dim_expand), self.dim_expand, *M.shape[2:]).to(device)

    def mod_population(self, M, only_expand_pop=False, device='cpu'):
        # default pop_dim=0, rotate_dim=1, and only expand population if
        # M.dim()==1 without regard of given key only_expand_pop=False
        if only_expand_pop or M.dim() == 1:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0)
                              for i in range(M.shape[0])], dim=0).to(device)
        else:
            M_mod = [torch.einsum('ry...,prxy->prx...', self.mod_dim(M, device=device)[p],
                                  self.pop_mod[p].to(device)).reshape(self.pop_expand[p],
                                                                      *M.shape[1:]) for p in range(len(self.pop_expand))]
            M_mod = torch.cat(M_mod, dim=0)
        return M_mod

    def get_subspace_Overlap(self, device='cpu'):
        Overlap = self.get_Overlap(reduction_p=True, device=device)
        Overlap_mod = Overlap.reshape(int(Overlap.shape[0] /
                                          self.dim_expand), self.dim_expand, int(Overlap.shape[1] /
                                                                                 self.dim_expand), self.dim_expand).transpose(1, 2).det()
        return Overlap_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, only_expand_pop=True, device=device)

    def mu_mod(self, device='cpu'):
        if self.no_z:
            mu_R = self.mod_population(self.mu_R, device=device)
            R_to_z_Masked = self.R_to_z.to(device) * self.Mask_R_to_z.to(device)
            mu_O = torch.einsum('or,pr->po', R_to_z_Masked, mu_R)
        else:
            mu_RO = self.mod_population(torch.cat([self.mu_R, self.mu_O], dim=1), device=device)
            mu_R = mu_RO[:, :2 * self.N_R]
            mu_O = mu_RO[:, 2 * self.N_R:]
        mu_I = self.apply_gen_mu_I(mu_R=mu_R, device=device)
        return [mu_I, mu_R, mu_O]

    def C_mod(self, device='cpu'):
        if self.no_z:
            C_R = self.mod_population(self.C_R, device=device)
            R_to_z_Masked = self.R_to_z.to(device) * self.Mask_R_to_z.to(device)
            C_O = torch.einsum('or,pry->poy', R_to_z_Masked, C_R)
        else:
            C_RO = self.mod_population(torch.cat([self.C_R, self.C_O], dim=1), device=device)
            C_R = C_RO[:, :2 * self.N_R]
            C_O = C_RO[:, 2 * self.N_R:]
        C_I = self.apply_gen_C_I(C_R=C_R, device=device)
        return [C_I, C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, only_expand_pop=True, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return self.mod_population(self.Mask_mu, only_expand_pop=True, device=device)

    def Mask_C_mod(self, device='cpu'):
        return self.mod_population(self.Mask_C, only_expand_pop=True, device=device)

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            self.total_pop, 1, self.N_F)
        return loadingvectors

    def ortho_loss(
            self,
            I=False,
            U=False,
            V=False,
            W=False,
            IU=False,
            IW=False,
            corr_only=False,
            d_I=1.,
            d_R=0.95,
            d_O=1,
            device='cpu', origin=False):
        if origin:
            return super().ortho_loss(I=I, U=U, V=V, IU=IU, IW=IW, corr_only=corr_only, d_I=d_I, d_R=d_R, d_O=d_O, device=device)
        else:
            def MSE0(x):
                L = nn.MSELoss(reduction='sum')
                return L(x, torch.zeros_like(x))

            Overlap = self.get_subspace_Overlap(device=device)
            N_I = int(self.N_I / self.dim_expand)
            N_R = int(self.N_R / self.dim_expand)
            N_O = int(self.N_O / self.dim_expand)
            N_F = int(self.N_F / self.dim_expand)
            I_index = list(range(N_I))
            U_index = list(range(N_I, N_I + N_R))
            V_index = list(range(N_I + N_R, N_I + 2 * N_R))
            O_index = list(range(N_I + 2 * N_R, N_F))

            # The I,U,V,W Overlap
            O_I = Overlap[I_index][:, I_index]
            O_I_od = d_I * (O_I - torch.eye(N_I, device=device))
            O_U = Overlap[U_index][:, U_index]
            O_U_od = O_U - d_R * torch.diag(O_U.diag())
            O_V = Overlap[V_index][:, V_index]
            O_V_od = O_V - d_R * torch.diag(O_V.diag())
            O_W = Overlap[O_index][:, O_index]
            O_W_od = d_O * (O_W - torch.eye(N_O, device=device))
            O_IU = Overlap[I_index][:, U_index]
            O_IW = Overlap[I_index][:, O_index]
            keys = [I, U, V, W, IU, IW]
            if not corr_only:
                values = [O_I_od, O_U_od, O_V_od, O_W_od, O_IU, O_IW]
            else:
                # correlation only orthogonal loss, avoid diagonal amplitude decaying
                O_I_d = O_I.diag()
                O_U_d = O_U.diag()
                O_V_d = O_V.diag()
                O_W_d = O_W.diag()
                corr_I = O_I / torch.einsum('u,v->uv', O_I_d, O_I_d).sqrt()
                corr_I_nd = corr_I - torch.diag(corr_I.diag())
                corr_U = O_U / torch.einsum('u,v->uv', O_U_d, O_U_d).sqrt()
                corr_U_nd = corr_U - torch.diag(corr_U.diag())
                corr_V = O_V / torch.einsum('u,v->uv', O_V_d, O_V_d).sqrt()
                corr_V_nd = corr_V - torch.diag(corr_V.diag())
                corr_W = O_W / torch.einsum('u,v->uv', O_W_d, O_W_d).sqrt()
                corr_W_nd = corr_W - torch.diag(corr_W.diag())
                corr_IU = O_IU / torch.einsum('u,v->uv', O_I_d, O_U_d).sqrt()
                corr_IW = O_IW / torch.einsum('u,v->uv', O_I_d, O_W_d).sqrt()
                values = [corr_I_nd, corr_U_nd, corr_V_nd, corr_W_nd, corr_IU, corr_IW]
            Ortho_Loss = sum([float(keys[idx]) * MSE0(values[idx]) for idx in range(len(keys))])
            return Ortho_Loss


class hierarchy_repar_mod_generation(reparameterlizedRNN_sample):
    pop_expand = [3, 18]
    dim_expand = 2
    gen_I = 'default'
    no_z = True

    def __init__(self, N_pop, N_I, N_R, N_O, act_func, tau=100):
        super().__init__(N_pop, N_I, N_R, N_O, act_func, tau=tau)
        self.n_pops = [sum(self.pop_expand[:i]) for i in range(len(self.pop_expand) + 1)]

    def set_no_z(self, no_z):
        assert (no_z or no_z == False)
        self.no_z = no_z

    def set_R_to_z(self, R_to_z_amp, R_to_z_phase, Mask_R_to_z):
        R_to_z_shape = (int(self.N_O / self.dim_expand), int(2 * self.N_R / self.dim_expand))
        assert R_to_z_amp.shape == R_to_z_shape
        assert R_to_z_phase.shape == R_to_z_shape
        assert Mask_R_to_z.shape == R_to_z_shape
        self.R_to_z_amp = R_to_z_amp
        self.R_to_z_phase = R_to_z_phase
        self.Mask_R_to_z = Mask_R_to_z

    def get_R_to_z(self, device='cpu'):
        assert self.no_z
        R_to_z_phase_flatten = self.R_to_z_phase.to(device).flatten()
        phase_matrix_flatten = tensor_rotates_2d(R_to_z_phase_flatten).to(device)
        phase_matrix = phase_matrix_flatten.reshape(*self.R_to_z_phase.shape, 2, 2)
        R_to_z_full = torch.einsum(
            'or,or,orxy->orxy',
            self.Mask_R_to_z.to(device),
            self.R_to_z_amp.to(device),
            phase_matrix)
        return R_to_z_full

    def apply_gen_mu_O(self, mu_R, device='cpu'):
        mu_R_expand = mu_R.to(device).reshape(mu_R.shape[0], int(2 * self.N_R / self.dim_expand), self.dim_expand)
        mu_O_expand = torch.einsum('orxy,pry->pox', self.get_R_to_z(device=device), mu_R_expand)
        return mu_O_expand.reshape(mu_R.shape[0], self.N_O)

    def apply_gen_C_O(self, C_R, device='cpu'):
        C_R_expand = C_R.to(device).reshape(C_R.shape[0], int(
            2 * self.N_R / self.dim_expand), self.dim_expand, self.N_F)
        C_O_expand = torch.einsum('orxy,pryk->poxk', self.get_R_to_z(device=device), C_R_expand)
        return C_O_expand.reshape(C_R.shape[0], self.N_O, self.N_F)

    def set_pop_mod(self, pop_mod):
        # setup of population modulation for each clusters
        # Modulation of each population should be of shape:[n_pop,n_dim,dim_expand,dim_expand]
        assert len(pop_mod) == len(self.pop_expand)
        if self.no_z:
            n_subspace = int(2 * self.N_R / self.dim_expand)
        else:
            n_subspace = int((2 * self.N_R + self.N_O) / self.dim_expand)
        for p in range(len(pop_mod)):
            assert pop_mod[p].shape == (self.pop_expand[p], n_subspace, self.dim_expand, self.dim_expand)
        self.pop_mod = pop_mod
        self.total_pop = sum(self.pop_expand)

    def set_gen_I(self, gen_I):
        assert gen_I in ['default', 'repeat', 'R_to_I']
        self.gen_I = gen_I
        if self.gen_I == 'default':
            assert self.mu_I.shape == (sum(self.pop_expand), self.N_I)
            assert self.C_I.shape == (sum(self.pop_expand), self.N_I, self.N_F)
        elif self.gen_I == 'repeat':
            assert self.mu_I.shape == (len(self.pop_expand), self.N_I)
            assert self.C_I.shape == (len(self.pop_expand), self.N_I, self.N_F)
        else:
            pass

    def set_R_to_I(self, R_to_I_amp, R_to_I_phase, Mask_R_to_I):
        # setup of R to Input
        if self.gen_I == 'R_to_I':
            R_to_I_shape = (self.N_pop, int(self.N_I / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            assert R_to_I_amp.shape == R_to_I_shape
            assert R_to_I_phase.shape == R_to_I_shape
            assert Mask_R_to_I.shape == R_to_I_shape
            self.R_to_I_amp = R_to_I_amp
            self.R_to_I_phase = R_to_I_phase
            self.Mask_R_to_I = Mask_R_to_I
        else:
            print("The 'gen_I' mode of network is not 'R_to_I', function 'set_R_to_I' not work!")

    def get_R_to_I(self, device='cpu'):
        assert self.gen_I == 'R_to_I'
        R_to_I_phase_flatten = self.R_to_I_phase.to(device).flatten()
        phase_matrix_flatten = tensor_rotates_2d(R_to_I_phase_flatten).to(device)
        phase_matrix = phase_matrix_flatten.reshape(*self.R_to_I_phase.shape, 2, 2)
        R_to_I_full = torch.einsum(
            'pir,pir,pirxy->pirxy',
            self.Mask_R_to_I.to(device),
            self.R_to_I_amp.to(device),
            phase_matrix)
        return R_to_I_full

    def apply_gen_mu_I(self, mu_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert mu_R is not None
            mu_R_expand = mu_R.to(
                device=device).reshape(
                mu_R.shape[0], int(
                    2 * self.N_R / self.dim_expand), self.dim_expand)
            mu_I = torch.cat([torch.einsum('irxy,pry->pix',
                                           self.get_R_to_I(device=device)[p],
                                           mu_R_expand[self.n_pops[p]:self.n_pops[p + 1]]).reshape(-1,
                                                                                                   self.N_I) for p in range(len(self.pop_expand))],
                             dim=0)
            return mu_I
        elif self.gen_I == 'repeat':
            return self.mod_population(self.mu_I, only_expand_pop=True, device=device)
        else:
            return self.mu_I.to(device)

    def apply_gen_C_I(self, C_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert C_R is not None
            C_R_expand = C_R.to(
                device=device).reshape(
                C_R.shape[0], int(
                    2 * self.N_R / self.dim_expand), self.dim_expand, self.N_F)
            C_I = torch.cat([torch.einsum('irxy,pryk->pixk',
                                          self.get_R_to_I(device=device)[p],
                                          C_R_expand[self.n_pops[p]:self.n_pops[p + 1]]).reshape(-1,
                                                                                                 self.N_I,
                                                                                                 self.N_F) for p in range(len(self.pop_expand))],
                            dim=0)
            return C_I
        elif self.gen_I == 'repeat':
            return self.mod_population(self.C_I, only_expand_pop=True, device=device)
        else:
            return self.C_I.to(device)

    def mod_dim(self, M, device='cpu'):
        return M.reshape(M.shape[0], int(M.shape[1] / self.dim_expand), self.dim_expand, *M.shape[2:]).to(device)

    def mod_population(self, M, only_expand_pop=False, device='cpu'):
        # default pop_dim=0, rotate_dim=1, and only expand population if
        # M.dim()==1 without regard of given key only_expand_pop=False
        if only_expand_pop or M.dim() == 1:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0)
                              for i in range(M.shape[0])], dim=0).to(device)
        else:
            M_mod = [torch.einsum('ry...,prxy->prx...', self.mod_dim(M, device=device)[p],
                                  self.pop_mod[p].to(device)).reshape(self.pop_expand[p],
                                                                      *M.shape[1:]) for p in range(len(self.pop_expand))]
            M_mod = torch.cat(M_mod, dim=0)
        return M_mod

    def get_subspace_Overlap(self, device='cpu'):
        Overlap = self.get_Overlap(reduction_p=True, device=device)
        Overlap_mod = Overlap.reshape(int(Overlap.shape[0] /
                                          self.dim_expand), self.dim_expand, int(Overlap.shape[1] /
                                                                                 self.dim_expand), self.dim_expand).transpose(1, 2).det()
        return Overlap_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, only_expand_pop=True, device=device)

    def mu_mod(self, device='cpu'):
        if self.no_z:
            mu_R = self.mod_population(self.mu_R, device=device)
            mu_O = self.apply_gen_mu_O(mu_R, device=device)
        else:
            mu_RO = self.mod_population(torch.cat([self.mu_R, self.mu_O], dim=1), device=device)
            mu_R = mu_RO[:, :2 * self.N_R]
            mu_O = mu_RO[:, 2 * self.N_R:]
        mu_I = self.apply_gen_mu_I(mu_R=mu_R, device=device)
        return [mu_I, mu_R, mu_O]

    def C_mod(self, device='cpu'):
        if self.no_z:
            C_R = self.mod_population(self.C_R, device=device)
            C_O = self.apply_gen_C_O(C_R, device=device)
        else:
            C_RO = self.mod_population(torch.cat([self.C_R, self.C_O], dim=1), device=device)
            C_R = C_RO[:, :2 * self.N_R]
            C_O = C_RO[:, 2 * self.N_R:]
        C_I = self.apply_gen_C_I(C_R=C_R, device=device)
        return [C_I, C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, only_expand_pop=True, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return self.mod_population(self.Mask_mu, only_expand_pop=True, device=device)

    def Mask_C_mod(self, device='cpu'):
        return self.mod_population(self.Mask_C, only_expand_pop=True, device=device)

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            self.total_pop, 1, self.N_F)
        return loadingvectors

    def ortho_loss(
            self,
            I=False,
            U=False,
            V=False,
            W=False,
            IU=False,
            IW=False,
            corr_only=False,
            d_I=1.,
            d_R=0.95,
            d_O=1,
            device='cpu', origin=False):
        if origin:
            return super().ortho_loss(I=I, U=U, V=V, IU=IU, IW=IW, corr_only=corr_only, d_I=d_I, d_R=d_R, d_O=d_O, device=device)
        else:
            def MSE0(x):
                L = nn.MSELoss(reduction='sum')
                return L(x, torch.zeros_like(x))

            Overlap = self.get_subspace_Overlap(device=device)
            N_I = int(self.N_I / self.dim_expand)
            N_R = int(self.N_R / self.dim_expand)
            N_O = int(self.N_O / self.dim_expand)
            N_F = int(self.N_F / self.dim_expand)
            I_index = list(range(N_I))
            U_index = list(range(N_I, N_I + N_R))
            V_index = list(range(N_I + N_R, N_I + 2 * N_R))
            O_index = list(range(N_I + 2 * N_R, N_F))

            # The I,U,V,W Overlap
            O_I = Overlap[I_index][:, I_index]
            O_I_od = d_I * (O_I - torch.eye(N_I, device=device))
            O_U = Overlap[U_index][:, U_index]
            O_U_od = O_U - d_R * torch.diag(O_U.diag())
            O_V = Overlap[V_index][:, V_index]
            O_V_od = O_V - d_R * torch.diag(O_V.diag())
            O_W = Overlap[O_index][:, O_index]
            O_W_od = d_O * (O_W - torch.eye(N_O, device=device))
            O_IU = Overlap[I_index][:, U_index]
            O_IW = Overlap[I_index][:, O_index]
            keys = [I, U, V, W, IU, IW]
            if not corr_only:
                values = [O_I_od, O_U_od, O_V_od, O_W_od, O_IU, O_IW]
            else:
                # correlation only orthogonal loss, avoid diagonal amplitude decaying
                O_I_d = O_I.diag()
                O_U_d = O_U.diag()
                O_V_d = O_V.diag()
                O_W_d = O_W.diag()
                corr_I = O_I / torch.einsum('u,v->uv', O_I_d, O_I_d).sqrt()
                corr_I_nd = corr_I - torch.diag(corr_I.diag())
                corr_U = O_U / torch.einsum('u,v->uv', O_U_d, O_U_d).sqrt()
                corr_U_nd = corr_U - torch.diag(corr_U.diag())
                corr_V = O_V / torch.einsum('u,v->uv', O_V_d, O_V_d).sqrt()
                corr_V_nd = corr_V - torch.diag(corr_V.diag())
                corr_W = O_W / torch.einsum('u,v->uv', O_W_d, O_W_d).sqrt()
                corr_W_nd = corr_W - torch.diag(corr_W.diag())
                corr_IU = O_IU / torch.einsum('u,v->uv', O_I_d, O_U_d).sqrt()
                corr_IW = O_IW / torch.einsum('u,v->uv', O_I_d, O_W_d).sqrt()
                values = [corr_I_nd, corr_U_nd, corr_V_nd, corr_W_nd, corr_IU, corr_IW]
            Ortho_Loss = sum([float(keys[idx]) * MSE0(values[idx]) for idx in range(len(keys))])
            return Ortho_Loss


class hierarchy_repar_mod_generation_mod_ortho(reparameterlizedRNN_sample):
    pop_expand = [3, 18]
    dim_expand = 2
    gen_I = 'default'
    no_z = True

    def __init__(self, N_pop, N_I, N_R, N_O, act_func, tau=100):
        super().__init__(N_pop, N_I, N_R, N_O, act_func, tau=tau)
        self.n_pops = [sum(self.pop_expand[:i]) for i in range(len(self.pop_expand) + 1)]

    def set_no_z(self, no_z):
        assert (no_z or no_z == False)
        self.no_z = no_z

    def set_R_to_z(self, R_to_z_amp, R_to_z_phase, Mask_R_to_z):
        R_to_z_shape = (int(self.N_O / self.dim_expand), int(2 * self.N_R / self.dim_expand))
        assert R_to_z_amp.shape == R_to_z_shape
        assert R_to_z_phase.shape == R_to_z_shape
        assert Mask_R_to_z.shape == R_to_z_shape
        self.R_to_z_amp = R_to_z_amp
        self.R_to_z_phase = R_to_z_phase
        self.Mask_R_to_z = Mask_R_to_z

    def get_R_to_z(self, device='cpu'):
        assert self.no_z
        R_to_z_phase_flatten = self.R_to_z_phase.to(device).flatten()
        phase_matrix_flatten = tensor_rotates_2d(R_to_z_phase_flatten).to(device)
        phase_matrix = phase_matrix_flatten.reshape(*self.R_to_z_phase.shape, 2, 2)
        R_to_z_full = torch.einsum(
            'or,or,orxy->orxy',
            self.Mask_R_to_z.to(device),
            self.R_to_z_amp.to(device),
            phase_matrix)
        return R_to_z_full

    def apply_gen_mu_O(self, mu_R, device='cpu'):
        mu_R_expand = mu_R.to(device).reshape(mu_R.shape[0], int(2 * self.N_R / self.dim_expand), self.dim_expand)
        mu_O_expand = torch.einsum('orxy,pry->pox', self.get_R_to_z(device=device), mu_R_expand)
        return mu_O_expand.reshape(mu_R.shape[0], self.N_O)

    def apply_gen_C_O(self, C_R, device='cpu'):
        C_R_expand = C_R.to(device).reshape(C_R.shape[0], int(
            2 * self.N_R / self.dim_expand), self.dim_expand, self.N_F)
        C_O_expand = torch.einsum('orxy,pryk->poxk', self.get_R_to_z(device=device), C_R_expand)
        return C_O_expand.reshape(C_R.shape[0], self.N_O, self.N_F)

    def set_pop_mod(self, pop_mod):
        # setup of population modulation for each clusters
        # Modulation of each population should be of shape:[n_pop,n_dim,dim_expand,dim_expand]
        assert len(pop_mod) == len(self.pop_expand)
        if self.no_z:
            n_subspace = int(2 * self.N_R / self.dim_expand)
        else:
            n_subspace = int((2 * self.N_R + self.N_O) / self.dim_expand)
        for p in range(len(pop_mod)):
            assert pop_mod[p].shape == (self.pop_expand[p], n_subspace, self.dim_expand, self.dim_expand)
        self.pop_mod = pop_mod
        self.total_pop = sum(self.pop_expand)

    def set_gen_I(self, gen_I):
        assert gen_I in ['default', 'repeat', 'R_to_I']
        self.gen_I = gen_I
        if self.gen_I == 'default':
            assert self.mu_I.shape == (sum(self.pop_expand), self.N_I)
            assert self.C_I.shape == (sum(self.pop_expand), self.N_I, self.N_F)
        elif self.gen_I == 'repeat':
            assert self.mu_I.shape == (len(self.pop_expand), self.N_I)
            assert self.C_I.shape == (len(self.pop_expand), self.N_I, self.N_F)
        else:
            pass

    def set_R_to_I(self, R_to_I_amp, R_to_I_phase, Mask_R_to_I):
        # setup of R to Input
        if self.gen_I == 'R_to_I':
            R_to_I_shape = (self.N_pop, int(self.N_I / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            assert R_to_I_amp.shape == R_to_I_shape
            assert R_to_I_phase.shape == R_to_I_shape
            assert Mask_R_to_I.shape == R_to_I_shape
            self.R_to_I_amp = R_to_I_amp
            self.R_to_I_phase = R_to_I_phase
            self.Mask_R_to_I = Mask_R_to_I
        else:
            print("The 'gen_I' mode of network is not 'R_to_I', function 'set_R_to_I' not work!")

    def get_R_to_I(self, device='cpu'):
        assert self.gen_I == 'R_to_I'
        R_to_I_phase_flatten = self.R_to_I_phase.to(device).flatten()
        phase_matrix_flatten = tensor_rotates_2d(R_to_I_phase_flatten).to(device)
        phase_matrix = phase_matrix_flatten.reshape(*self.R_to_I_phase.shape, 2, 2)
        R_to_I_full = torch.einsum(
            'pir,pir,pirxy->pirxy',
            self.Mask_R_to_I.to(device),
            self.R_to_I_amp.to(device),
            phase_matrix)
        return R_to_I_full

    def apply_gen_mu_I(self, mu_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert mu_R is not None
            mu_R_expand = mu_R.to(
                device=device).reshape(
                mu_R.shape[0], int(
                    2 * self.N_R / self.dim_expand), self.dim_expand)
            mu_I = torch.cat([torch.einsum('irxy,pry->pix',
                                           self.get_R_to_I(device=device)[p],
                                           mu_R_expand[self.n_pops[p]:self.n_pops[p + 1]]).reshape(-1,
                                                                                                   self.N_I) for p in range(len(self.pop_expand))],
                             dim=0)
            return mu_I
        elif self.gen_I == 'repeat':
            return self.mod_population(self.mu_I, only_expand_pop=True, device=device)
        else:
            return self.mu_I.to(device)

    def apply_gen_C_I(self, C_R=None, device='cpu'):
        if self.gen_I == 'R_to_I':
            assert C_R is not None
            C_R_expand = C_R.to(
                device=device).reshape(
                C_R.shape[0], int(
                    2 * self.N_R / self.dim_expand), self.dim_expand, self.N_F)
            C_I = torch.cat([torch.einsum('irxy,pryk->pixk',
                                          self.get_R_to_I(device=device)[p],
                                          C_R_expand[self.n_pops[p]:self.n_pops[p + 1]]).reshape(-1,
                                                                                                 self.N_I,
                                                                                                 self.N_F) for p in range(len(self.pop_expand))],
                            dim=0)
            return C_I
        elif self.gen_I == 'repeat':
            return self.mod_population(self.C_I, only_expand_pop=True, device=device)
        else:
            return self.C_I.to(device)

    def mod_dim(self, M, device='cpu'):
        return M.reshape(M.shape[0], int(M.shape[1] / self.dim_expand), self.dim_expand, *M.shape[2:]).to(device)

    def mod_population(self, M, only_expand_pop=False, device='cpu'):
        # default pop_dim=0, rotate_dim=1, and only expand population if
        # M.dim()==1 without regard of given key only_expand_pop=False
        if only_expand_pop or M.dim() == 1:
            M_mod = torch.cat([M[i:i + 1].repeat_interleave(self.pop_expand[i], dim=0)
                              for i in range(M.shape[0])], dim=0).to(device)
        else:
            M_mod = [torch.einsum('ry...,prxy->prx...', self.mod_dim(M, device=device)[p],
                                  self.pop_mod[p].to(device)).reshape(self.pop_expand[p],
                                                                      *M.shape[1:]) for p in range(len(self.pop_expand))]
            M_mod = torch.cat(M_mod, dim=0)
        return M_mod

    def get_subspace_Overlap(self, device='cpu'):
        Overlap = self.get_Overlap(reduction_p=True, device=device)
        Overlap_mod = Overlap.reshape(int(Overlap.shape[0] /
                                          self.dim_expand), self.dim_expand, int(Overlap.shape[1] /
                                                                                 self.dim_expand), self.dim_expand).transpose(1, 2).det()
        return Overlap_mod

    def G_mod(self, device='cpu'):
        return self.mod_population(self.G, only_expand_pop=True, device=device)

    def mu_mod(self, device='cpu'):
        if self.no_z:
            mu_R = self.mod_population(self.mu_R, device=device)
            mu_O = self.apply_gen_mu_O(mu_R, device=device)
        else:
            mu_RO = self.mod_population(torch.cat([self.mu_R, self.mu_O], dim=1), device=device)
            mu_R = mu_RO[:, :2 * self.N_R]
            mu_O = mu_RO[:, 2 * self.N_R:]
        mu_I = self.apply_gen_mu_I(mu_R=mu_R, device=device)
        return [mu_I, mu_R, mu_O]

    def C_mod(self, device='cpu'):
        if self.no_z:
            C_R = self.mod_population(self.C_R, device=device)
            C_O = self.apply_gen_C_O(C_R, device=device)
        else:
            C_RO = self.mod_population(torch.cat([self.C_R, self.C_O], dim=1), device=device)
            C_R = C_RO[:, :2 * self.N_R]
            C_O = C_RO[:, 2 * self.N_R:]
        C_I = self.apply_gen_C_I(C_R=C_R, device=device)
        return [C_I, C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.mod_population(self.Mask_G, only_expand_pop=True, device=device)

    def Mask_mu_mod(self, device='cpu'):
        return self.mod_population(self.Mask_mu, only_expand_pop=True, device=device)

    def Mask_C_mod(self, device='cpu'):
        return self.mod_population(self.Mask_C, only_expand_pop=True, device=device)

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        return torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    def get_loading(self, device='cpu'):
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(
            self.total_pop, 1, self.N_F)
        return loadingvectors

    def ortho_loss(
            self,
            I=False,
            U=False,
            V=False,
            W=False,
            IU=False,
            IW=False,
            corr_only=False,
            d_I=1.,
            d_R=0.95,
            d_O=1,
            device='cpu', origin=False,Mask_IV=None):
        if origin:
            return super().ortho_loss(I=I, U=U, V=V, IU=IU, IW=IW, corr_only=corr_only, d_I=d_I, d_R=d_R, d_O=d_O, device=device,Mask_IV=Mask_IV)
        else:
            def L1Loss0(x):
                L = nn.L1Loss(reduction='sum')
                return L(x, torch.zeros_like(x))

            Overlap = self.get_subspace_Overlap(device=device)
            N_I = int(self.N_I / self.dim_expand)
            N_R = int(self.N_R / self.dim_expand)
            N_O = int(self.N_O / self.dim_expand)
            N_F = int(self.N_F / self.dim_expand)
            I_index = list(range(N_I))
            U_index = list(range(N_I, N_I + N_R))
            V_index = list(range(N_I + N_R, N_I + 2 * N_R))
            O_index = list(range(N_I + 2 * N_R, N_F))

            # The I,U,V,W Overlap
            O_I = Overlap[I_index][:, I_index]
            O_I_od = d_I * (O_I - torch.eye(N_I, device=device))
            O_U = Overlap[U_index][:, U_index]
            O_U_od = O_U - d_R * torch.diag(O_U.diag())
            O_V = Overlap[V_index][:, V_index]
            O_V_od = O_V - d_R * torch.diag(O_V.diag())
            O_W = Overlap[O_index][:, O_index]
            O_W_od = d_O * (O_W - torch.eye(N_O, device=device))
            O_IU = Overlap[I_index][:, U_index]
            O_IW = Overlap[I_index][:, O_index]
            if Mask_IV is None:
                Mask_IV=torch.zeros(N_I,N_R).to(device)
            else:
                Mask_IV = Mask_IV.to(device)
            O_IV=Overlap[I_index][:,V_index]*Mask_IV
            keys = [I, U, V, W, IU, IW,True]
            if not corr_only:
                values = [O_I_od, O_U_od, O_V_od, O_W_od, O_IU, O_IW, O_IV]
            else:
                # correlation only orthogonal loss, avoid diagonal amplitude decaying
                O_I_d = O_I.diag()
                O_U_d = O_U.diag()
                O_V_d = O_V.diag()
                O_W_d = O_W.diag()
                corr_I = O_I / torch.einsum('u,v->uv', O_I_d, O_I_d).sqrt()
                corr_I_nd = corr_I - torch.diag(corr_I.diag())
                corr_U = O_U / torch.einsum('u,v->uv', O_U_d, O_U_d).sqrt()
                corr_U_nd = corr_U - torch.diag(corr_U.diag())
                corr_V = O_V / torch.einsum('u,v->uv', O_V_d, O_V_d).sqrt()
                corr_V_nd = corr_V - torch.diag(corr_V.diag())
                corr_W = O_W / torch.einsum('u,v->uv', O_W_d, O_W_d).sqrt()
                corr_W_nd = corr_W - torch.diag(corr_W.diag())
                corr_IU = O_IU / torch.einsum('u,v->uv', O_I_d, O_U_d).sqrt()
                corr_IW = O_IW / torch.einsum('u,v->uv', O_I_d, O_W_d).sqrt()
                corr_IV = O_IV / torch.einsum('u,v->uv', O_I_d, O_V_d).sqrt()
                values = [corr_I_nd, corr_U_nd, corr_V_nd, corr_W_nd, corr_IU, corr_IW, corr_IV]
            Ortho_Loss = sum([float(keys[idx]) * L1Loss0(values[idx]) for idx in range(len(keys))])
            return Ortho_Loss
