# -*- codeing = utf-8 -*-
# @time:2021/8/8 下午4:31
# Author:Xuewen Shen
# @File:network.py
# @Software:PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from rw import *
import copy
from torch.distributions import Categorical

default_param = dict(
    N_Neuron=128, N_rank=2, act_func='Tanh',
    in_Channel=2, out_Channel=1,
    tau=100, dt=10, g_in=0.01, g_rec=0.15, train_hidden_0=False

)


class lowRNN(nn.Module):
    def __init__(self, P):
        super(lowRNN, self).__init__()
        self.P = P
        self.N = self.P['N_Neuron']
        self.r = self.P['N_rank']
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.In = nn.Linear(in_features=self.P['in_Channel'], out_features=self.N, bias=False)
        self.V = nn.Linear(in_features=self.N, out_features=self.r, bias=False)
        self.U = nn.Linear(in_features=self.r, out_features=self.N, bias=False)
        self.Out = nn.Linear(in_features=self.N, out_features=self.P['out_Channel'], bias=False)
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=True)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=True)
        # train_hidden_0:True if initial hidden state requires training, else False and given zero initial state
        if self.P['train_hidden_0']:
            self.hidden_0 = nn.Parameter(torch.randn(self.N), requires_grad=True)
        else:
            self.hidden_0 = nn.Parameter(torch.zeros(self.N), requires_grad=False)

    def forward(self, Batch_Input, hidden_0=None, device='cpu'):
        # Batch_Input:(Batch_Size,T,channels)
        Batch_Size, T = Batch_Input.shape[0:2]
        # Input noise should be dealt with out of forward function
        Batch_Input = torch.mul(Batch_Input, self.in_strength).to(device)
        aligned_Batch_Input = self.In(Batch_Input)
        # hidden in one step is of shape(Batch_Size,N),decide if using default hidden
        if hidden_0 is None:
            hidden = self.hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1).to(device)
        else:
            hidden = hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1).to(device)

        hidden_t = torch.tensor([]).to(device)
        for frame in range(T):
            Noise_recurrent = math.sqrt(2 * self.P['tau']) * self.P['g_rec'] * torch.randn(
                [Batch_Size, self.N]).to(device)
            hidden = (1 - self.P['dt'] / self.P['tau']) * hidden + self.P['dt'] / self.P['tau'] * (
                self.U(self.V(self.act_func(hidden))) / self.N + aligned_Batch_Input[:, frame, :] + Noise_recurrent)

            # hidden_t=(Batch_Size,T,N)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = torch.mul(self.Out(self.act_func(hidden_t)), self.out_strength) / self.N
        return hidden_t, out

    def reinit(self, requires_grad=[True, True, True, True], W=[None, None, None, None]):
        layer = [self.In, self.V, self.U, self.Out]
        for i in range(4):
            layer[i].weight.requires_grad = requires_grad[i]
            if W[i] is not None:
                layer[i].weight.data = W[i]

    def set_strength(self, in_strength=None, out_strength=None):
        if in_strength is not None:
            self.in_strength.data = in_strength
        if out_strength is not None:
            self.out_strength.data = out_strength

    def ortho_reg_loss(self):
        '''
        The input channel should be orthogonal to m vectors in the low-rank network, and m
         vectors should be orthogonal to each other.
        :return:
        orthogonal regularization loss
        '''
        # vector(N_neuron,in_Channel+N_rank)
        vector = torch.cat((self.In.weight, self.U.weight), dim=1)
        norm = torch.sqrt((vector ** 2).sum(dim=0)).unsqueeze(dim=0)
        norm_matrix = norm.T @ norm
        # Q: covariance matrix of {I,m} with unital normalization;D: diagonal
        Q = vector.T @ vector
        norm_Q = (Q / norm_matrix) ** 2
        D = torch.diag(torch.diagonal(norm_Q))
        Loss = nn.L1Loss(reduction='sum')
        return Loss(norm_Q, D) / 2

    def connect_weight_loss(self):
        '''
        The norm of general connectivity of network is constrained
        :return:
        connectivity weight loss
        '''
        Loss = nn.MSELoss(reduction='mean')
        loss_n = Loss(self.V.weight ** 2, torch.zeros_like(self.V.weight))
        loss_m = Loss(self.U.weight ** 2, torch.zeros_like(self.U.weight))
        return loss_n + loss_m


default_Seq_param = dict(
    Embedding=None, N=4096, M=64, R=10, in_Channel=2, out_Channel=2, decoder_steps=3, g_in=0.1, g_rec=0.15,
    t_rest=20, Dt_rest=10, t_on=10, Dt_on=5, t_off=10, Dt_off=5, t_delay=30, Dt_delay=10,
)


# get N*N matrix with eigenvalues distributed within a ring of radius rho
def getJ(N, rho):
    J = np.random.normal(loc=0, scale=1 / np.sqrt(N), size=(N, N))
    rs = max(np.real(np.linalg.eigvals(J)))
    return rho * J / rs


# get N*N matrix with eigenvalues distributed around a ring of radius rho
def getJ_equa_radius(N, rho):
    J = np.random.normal(loc=0, scale=1 / np.sqrt(N), size=(N, N))
    w, v = np.linalg.eig(J)
    w_ = rho * w / np.sqrt(w * np.conjugate(w))
    return np.real(v @ np.diag(w_) @ np.linalg.inv(v))


def Batch_onehot(n_emb, Batch):
    Batch_Size, L = Batch.shape

    def onehot(n, i):
        l = [0, ] * n
        l[i] += 1
        return l

    return torch.tensor([[onehot(n_emb, Batch[i, j] - 1) for j in range(L)] for i in range(Batch_Size)]).float()


class low_decoder(nn.Module):
    def __init__(self, P):
        super(low_decoder, self).__init__()
        self.P = P
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        # default embedding(6,2) 6points of 2-dim rep of flat surface
        self.Embedding = P['Embedding']
        # Linear visual Embedding
        self.In = nn.Linear(in_features=P['in_Channel'], out_features=P['N'], bias=False)
        # Low rank network UV
        self.hidden_0 = torch.zeros(P['N'])
        self.J = nn.Linear(in_features=P['N'], out_features=P['N'], bias=False)
        # # In pre_trained network there is no linear projection and N=M
        # # Projection from low rank hidden layer to full rank decoder
        # self.W = nn.Linear(in_features=P['N'], out_features=P['M'], bias=True)
        # self.W.bias.data = 0. * self.W.bias.data
        # Fulle rank network Q
        self.V = nn.Linear(in_features=P['M'], out_features=P['R'], bias=False)
        self.U = nn.Linear(in_features=P['R'], out_features=P['M'], bias=False)
        # Linear readout
        self.Out = nn.Linear(in_features=P['M'], out_features=P['out_Channel'], bias=False)
        # input strength setting
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=False)

    def reinit(self):
        self.In.weight.requires_grad = self.P['require_grad'][0]
        self.In.weight.data = torch.randn(self.In.weight.data.shape) / math.sqrt(self.P['N'])
        self.Out.weight.requires_grad = self.P['require_grad'][1]
        self.Out.weight.data = torch.randn(self.Out.weight.data.shape) / math.sqrt(self.P['M'])
        # self.J.weight.data = tt(getJ_equa_radius(self.P['N'], self.P['rho']))
        # self.Q.weight.data = tt(getJ_equa_radius(self.P['M'], self.P['rho']))
        self.J.weight.data = tt(np.random.normal(loc=0, scale=1 / self.P['N'], size=(self.P['N'], self.P['N'])))

        self.V.weight.data = tt(
            np.random.normal(loc=0, scale=1 / np.sqrt(self.P['M']), size=(self.P['R'], self.P['M'])))
        self.U.weight.data = tt(
            np.random.normal(loc=0, scale=1 / np.sqrt(self.P['M']), size=(self.P['M'], self.P['R'])))

    def Batch2Input(self, Batch, add_param=None, device='cpu'):
        '''
        :param Batch: tensor of int (Bath_Size,length), must be of tensor type, not list
        :param add_param: extra option, keywords to replace default parameters of network
        :param device:device ran on,default='cpu'
        :return: Bacth of Input: tensor of float (Batch_Size,T,in_Channel),default In_Channel=2 (flat surface of visual input)
        '''
        Batch_Size, L = Batch.shape
        n_emb = self.Embedding.shape[0]
        P = self.P.copy()
        if add_param is not None:
            P.update(add_param)

        t_rest = P['t_rest'] + random.randint(0, P['Dt_rest'])
        t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for i in range(L)]
        t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for i in range(L - 1)]
        t_delay = P['t_delay'] + random.randint(0, P['Dt_delay'])
        t_total = t_rest + sum(t_on) + sum(t_off) + t_delay
        Input = math.sqrt(2 * self.P['tau']) * P['g_in'] * torch.randn((Batch_Size, t_total, P['in_Channel']))
        Batch_oh = Batch_onehot(n_emb, Batch)
        Batch_Input = Batch_oh @ self.Embedding
        pointer = t_rest
        for i in range(L):
            Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
            if i != L - 1:
                pointer += t_on[i] + t_off[i]
        return Input.to(device)

    def encode(self, Batch_Input, device='cpu', **kwargs):
        '''
        Input to Low rank RNN hidden state through encoder
        :param Batch_Input: Batch of Input tensor of float (Batch_Size,T,in_Channel),default In_Channel=2
        :param device: device ran on,default='cpu'
        :param kwargs: extra option,keywords to replace default parameters of network
        :return: hidden trajectory of low rank network
        '''
        Batch_Size, T = Batch_Input.shape[:2]

        # Embedding of Batch_Input to low rank network
        Input = (self.In(Batch_Input * self.in_strength)).to(device)

        # given initial hidden state or default hidden state
        if 'hidden_0' in kwargs:
            hidden = kwargs['hidden_0']
        else:
            hidden = self.hidden_0
        hidden = hidden.unsqueeze(dim=0).repeat(Batch_Size, 1).to(device)
        hidden_t = hidden.unsqueeze(dim=1)

        # recurrent hidden state computation
        alpha = self.P['dt'] / self.P['tau']
        if 'g_rec' in kwargs:
            g_rec = kwargs['g_rec']
        else:
            g_rec = self.P['g_rec']
        Noise_rec = math.sqrt(2 * self.P['tau']) * g_rec * torch.randn(Input.shape).to(device)
        for i in range(T):
            hidden = (1 - alpha) * hidden + alpha * (
                self.J(self.act_func(hidden)) + Noise_rec[:, i] + Input[:, i])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def decode(self, encoder_hidden, device='cpu', **kwargs):
        '''
        hidden to readout through full rank RNN decoder
        :param encoder_hidden: Batch of the last step hidden layer of encoder (Batch_Size,N)
        :param device: device ran on,default='cpu'
        :param kwargs:extra option,keywords to replace default parameters of network
        :return: Output of network with default steps=P['decoder_steps'] and can be replaced by kwargs['decoder_steps']
        '''

        # decoder steps,kwargs-param-defualt=3,
        if 'decoder_steps' in kwargs:
            decoder_steps = kwargs['decoder_steps']
        elif 'decoder_steps' in self.P:
            decoder_steps = self.P['decoder_steps']
        else:
            decoder_steps = 3

        hidden = encoder_hidden
        hidden_t = hidden.unsqueeze(dim=1)
        for i in range(decoder_steps):
            hidden = self.U(self.V(hidden))
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = (self.Out(hidden_t) * self.out_strength).to(device)
        return hidden_t, out


class pre_Seq2Seq(nn.Module):
    def __init__(self, P):
        super(pre_Seq2Seq, self).__init__()
        self.P = P
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        # default embedding(6,2) 6points of 2-dim rep of flat surface
        self.Embedding = P['Embedding']
        # Linear visual Embedding
        self.In = nn.Linear(in_features=P['in_Channel'], out_features=P['N'], bias=False)
        # Low rank network UV
        self.hidden_0 = torch.zeros(P['N'])
        self.J = nn.Linear(in_features=P['N'], out_features=P['N'], bias=False)
        # # In pre_trained network there is no linear projection and N=M
        # # Projection from low rank hidden layer to full rank decoder
        # self.W = nn.Linear(in_features=P['N'], out_features=P['M'], bias=True)
        # self.W.bias.data = 0. * self.W.bias.data
        # Fulle rank network Q
        self.Q = nn.Linear(in_features=P['M'], out_features=P['M'], bias=False)
        # Linear readout
        self.Out = nn.Linear(in_features=P['M'], out_features=P['out_Channel'], bias=False)
        # input strength setting
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=False)

    def reinit(self):
        self.In.weight.requires_grad = self.P['require_grad'][0]
        self.In.weight.data = torch.randn(self.In.weight.data.shape) / math.sqrt(self.P['N'])
        self.Out.weight.requires_grad = self.P['require_grad'][1]
        self.Out.weight.data = torch.randn(self.Out.weight.data.shape) / math.sqrt(self.P['M'])
        # self.J.weight.data = tt(getJ_equa_radius(self.P['N'], self.P['rho']))
        # self.Q.weight.data = tt(getJ_equa_radius(self.P['M'], self.P['rho']))
        self.J.weight.data = tt(
            np.random.normal(loc=0, scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['N'])))
        self.Q.weight.data = tt(
            np.random.normal(loc=0, scale=1 / self.P['M'], size=(self.P['M'], self.P['M'])))

    def Batch2Input(self, Batch, add_param=None, device='cpu'):
        '''
        :param Batch: tensor of int (Bath_Size,length), must be of tensor type, not list
        :param add_param: extra option, keywords to replace default parameters of network
        :param device:device ran on,default='cpu'
        :return: Bacth of Input: tensor of float (Batch_Size,T,in_Channel),default In_Channel=2 (flat surface of visual input)
        '''
        Batch_Size, L = Batch.shape
        n_emb = self.Embedding.shape[0]
        P = self.P.copy()
        if add_param is not None:
            P.update(add_param)

        t_rest = P['t_rest'] + random.randint(0, P['Dt_rest'])
        t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for i in range(L)]
        t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for i in range(L - 1)]
        t_delay = P['t_delay'] + random.randint(0, P['Dt_delay'])
        t_total = t_rest + sum(t_on) + sum(t_off) + t_delay
        Input = math.sqrt(2 * self.P['tau']) * P['g_in'] * torch.randn((Batch_Size, t_total, P['in_Channel']))
        Batch_oh = Batch_onehot(n_emb, Batch)
        Batch_Input = Batch_oh @ self.Embedding
        pointer = t_rest
        for i in range(L):
            Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
            if i != L - 1:
                pointer += t_on[i] + t_off[i]
        return Input.to(device)

    def encode(self, Batch_Input, device='cpu', **kwargs):
        '''
        Input to Low rank RNN hidden state through encoder
        :param Batch_Input: Batch of Input tensor of float (Batch_Size,T,in_Channel),default In_Channel=2
        :param device:
        :param kwargs: extra option,keywords to replace default parameters of network
        :return: hidden trajectory of low rank network
        '''
        Batch_Size, T = Batch_Input.shape[:2]

        # Embedding of Batch_Input to low rank network
        Input = (self.In(Batch_Input * self.in_strength)).to(device)

        # given initial hidden state or default hidden state
        if 'hidden_0' in kwargs:
            hidden = kwargs['hidden_0']
        else:
            hidden = self.hidden_0
        hidden = hidden.unsqueeze(dim=0).repeat(Batch_Size, 1).to(device)
        hidden_t = hidden.unsqueeze(dim=1)

        # recurrent hidden state computation
        alpha = self.P['dt'] / self.P['tau']
        if 'g_rec' in kwargs:
            g_rec = kwargs['g_rec']
        else:
            g_rec = self.P['g_rec']
        Noise_rec = math.sqrt(2 * self.P['tau']) * g_rec * torch.randn(Input.shape).to(device)
        for i in range(T):
            hidden = (1 - alpha) * hidden + alpha * (
                self.J(self.act_func(hidden)) + Noise_rec[:, i] + Input[:, i])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def decode(self, encoder_hidden, device='cpu', **kwargs):
        '''
        hidden to readout through full rank RNN decoder
        :param encoder_hidden: Batch of the last step hidden layer of encoder (Batch_Size,N)
        :param device:
        :param kwargs:extra option,keywords to replace default parameters of network
        :return: Output of network with default steps=P['decoder_steps'] and can be replaced by kwargs['decoder_steps']
        '''

        # decoder steps,kwargs-param-defualt=3,
        if 'decoder_steps' in kwargs:
            decoder_steps = kwargs['decoder_steps']
        elif 'decoder_steps' in self.P:
            decoder_steps = self.P['decoder_steps']
        else:
            decoder_steps = 3

        hidden = encoder_hidden
        hidden_t = hidden.unsqueeze(dim=1)
        for i in range(decoder_steps):
            hidden = self.Q(hidden)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = (self.Out(hidden_t) * self.out_strength).to(device)
        return hidden_t, out


# Full model of seq2seq model which include several parts:
# Embedding of input, Low rank working memory network,Linear projection of last step,full rank decoder network,readout
class Seq2SeqFull(nn.Module):
    def __init__(self, P):
        super(Seq2SeqFull, self).__init__()
        self.P = P
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        # default embedding(6,2) 6points of 2-dim rep of flat surface
        self.Embedding = P['Embedding']
        # Linear visual Embedding
        self.In = nn.Linear(in_features=P['in_Channel'], out_features=P['N'], bias=False)
        # Low rank network UV
        self.hidden_0 = torch.zeros(P['N'])
        self.V = nn.Linear(in_features=P['N'], out_features=P['R'], bias=False)
        self.U = nn.Linear(in_features=P['R'], out_features=P['N'], bias=False)
        # Projection from low rank hidden layer to full rank decoder
        self.W = nn.Linear(in_features=P['N'], out_features=P['M'], bias=True)
        self.W.bias.data = 0. * self.W.bias.data
        # Fulle rank network Q
        self.Q = nn.Linear(in_features=P['M'], out_features=P['M'], bias=False)
        # Linear readout
        self.Out = nn.Linear(in_features=P['M'], out_features=P['out_Channel'], bias=False)
        # input strength setting
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=False)

    def reinit(self):
        # re-initiate to form statistic distribution based on system hyperparameters and training process setup
        layer = [self.In, self.V, self.U, self.W, self.Out]
        for i in range(5):
            layer[i].weight.requires_grad = self.P['require_grad'][i]
            W = self.P['weight'][i] * torch.randn(layer[i].weight.data.shape)
            layer[i].weight.data = W
        # linear recurrent reinitialization
        self.Q.weight.data = tt(getJ(self.P['M'], self.P['rho']))

    def Batch2Input(self, Batch, add_param=None, device='cpu'):
        '''
        :param Batch: tensor of int (Bath_Size,length), must be of tensor type, not list
        :param add_param: extra option, keywords to replace default parameters of network
        :param device:device ran on,default='cpu'
        :return: Bacth of Input: tensor of float (Batch_Size,T,in_Channel),default In_Channel=2 (flat surface of visual input)
        '''
        Batch_Size, L = Batch.shape
        n_emb = self.Embedding.shape[0]
        P = self.P.copy()
        if add_param is not None:
            P.update(add_param)

        t_rest = P['t_rest'] + random.randint(0, P['Dt_rest'])
        t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for i in range(L)]
        t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for i in range(L - 1)]
        t_delay = P['t_delay'] + random.randint(0, P['Dt_delay'])
        t_total = t_rest + sum(t_on) + sum(t_off) + t_delay
        Input = math.sqrt(2 * self.P['tau']) * P['g_in'] * torch.randn((Batch_Size, t_total, P['in_Channel']))
        Batch_oh = Batch_onehot(n_emb, Batch)
        Batch_Input = Batch_oh @ self.Embedding
        pointer = t_rest
        for i in range(L):
            Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
            if i != L - 1:
                pointer += t_on[i] + t_off[i]
        return Input.to(device)

    def encode(self, Batch_Input, device='cpu', **kwargs):
        '''
        Input to Low rank RNN hidden state through encoder
        :param Batch_Input: Batch of Input tensor of float (Batch_Size,T,in_Channel),default In_Channel=2
        :param device: device ran on,default='cpu'
        :param kwargs: extra option,keywords to replace default parameters of network
        :return: hidden trajectory of low rank network
        '''
        Batch_Size, T = Batch_Input.shape[:2]

        # Embedding of Batch_Input to low rank network
        Input = (self.In(Batch_Input * self.in_strength)).to(device)

        # given initial hidden state or default hidden state
        if 'hidden_0' in kwargs:
            hidden = kwargs['hidden_0']
        else:
            hidden = self.hidden_0
        hidden = hidden.unsqueeze(dim=0).repeat(Batch_Size, 1).to(device)
        hidden_t = hidden.unsqueeze(dim=1)

        # recurrent hidden state computation
        alpha = self.P['dt'] / self.P['tau']
        if 'g_rec' in kwargs:
            g_rec = kwargs['g_rec']
        else:
            g_rec = self.P['g_rec']
        Noise_rec = math.sqrt(2 * self.P['tau']) * g_rec * torch.randn(Input.shape).to(device)
        for i in range(T):
            hidden = (1 - alpha) * hidden + alpha * (
                self.U(self.V(self.act_func(hidden))) / self.P['N'] + Noise_rec[:, i] + Input[:, i])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def decode(self, encoder_hidden, device='cpu', **kwargs):
        '''
        hidden to readout through full rank RNN decoder
        :param encoder_hidden: Batch of the last step hidden layer of encoder (Batch_Size,N)
        :param device: device ran on,default='cpu'
        :param kwargs:extra option,keywords to replace default parameters of network
        :return: Output of network with default steps=P['decoder_steps'] and can be replaced by kwargs['decoder_steps']
        '''

        # decoder steps,kwargs-param-defualt=3,
        if 'decoder_steps' in kwargs:
            decoder_steps = kwargs['decoder_steps']
        elif 'decoder_steps' in self.P:
            decoder_steps = self.P['decoder_steps']
        else:
            decoder_steps = 3

        hidden = (self.W(self.act_func(encoder_hidden)) / self.P['N']).to(device)
        hidden_t = hidden.unsqueeze(dim=1)
        for i in range(decoder_steps):
            hidden = self.Q(hidden)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = (self.Out(hidden_t) * self.out_strength / self.P['M']).to(device)
        return hidden_t, out


class Seq2SeqModel_simp(nn.Module):
    def __init__(self, P):
        super(Seq2SeqModel_simp, self).__init__()
        self.P = P
        self.device = P['device']
        if 'Embedding' in P:
            self.Embedding = P['Embedding'].to(self.device)
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.In = nn.Parameter(tt(np.random.normal(loc=0., scale=1., size=(self.P['N'], self.P['N_I']))),
                               requires_grad=self.P['grad_In'])
        self.Encoder_P = dict(N=P['N'], R=P['N_R'], act_func=P['act_func'], dt=P['dt'], tau=P['tau'],
                              g_rec=P['g_rec'], device=P['device'])
        self.Encoder = LeakyRNN(self.Encoder_P)
        self.Out = nn.Parameter(tt(np.random.normal(loc=0., scale=1., size=(self.P['N'], self.P['N_O']))),
                                requires_grad=self.P['grad_Out'])
        self.in_strength = nn.Parameter(torch.ones(self.P['N_I']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['N_O']), requires_grad=False)

    def train_strength(self, train_In_strength: bool, train_Out_strength: bool):
        self.in_strength.requires_grad = train_In_strength
        self.out_strength.requires_grad = train_Out_strength

    def reinit(self, g_En=1.):
        self.Encoder.reinit(g=g_En)

    def forward(self, Batch_Input, device='cpu', get_Encoder=False, **kwargs):
        Batch_Input.to(device)
        Input = (Batch_Input * self.in_strength) @ self.In.T
        Encoder_hidden = self.Encoder(Input, device=device, **kwargs)
        Output = self.out_strength * self.act_func(Encoder_hidden) @ self.Out / self.P['N']
        if not get_Encoder:
            return Output
        else:
            return Encoder_hidden, Output

    def Batch2Input(self, Batch, sync=True, device='cpu', add_param=None):
        if sync:
            Batch_Size, L = Batch.shape
            n_emb = self.Embedding.shape[0]
            P = self.P.copy()
            if add_param is not None:
                P.update(add_param)
            Input = math.sqrt(2 * self.P['tau'] / self.P['dt']) * P['g_in'] * torch.randn(
                (Batch_Size, self.P['t_upb'][L - 1], P['N_I']))
            t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for _ in range(L)]
            t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for _ in range(L - 1)]
            Batch_oh = Batch_onehot(n_emb, Batch)
            Batch_Input = Batch_oh @ self.Embedding
            pointer = 0
            for i in range(L):
                Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
                if i != L - 1:
                    pointer += t_on[i] + t_off[i]
            return Input.to(device)
        else:
            return torch.cat([self.Batch2Input(Batch[i], sync=True, device=device) for i in
                              range(Batch.shape[0])], dim=0)

    def from_vector(self, vector, N_I, N_R, N_O):
        self.In.data = vector[:, :N_I]
        self.Encoder.U.data = vector[:, N_I:N_I + N_R]
        self.Encoder.V.data = vector[:, N_I + N_R:N_I + 2 * N_R]
        self.Out.data = vector[:, N_I + 2 * N_R:N_I + 2 * N_R + N_O]

    @staticmethod
    def from_loading(LoadingVector: LoadingVector, add_param=dict(), device='cpu'):
        P = dict(N=LoadingVector.N, N_I=LoadingVector.N_I, N_R=LoadingVector.N_R, N_O=LoadingVector.N_O, gain=[1., 1.],
                 require_grad=[False, False], )
        P.update(add_param)
        model = Seq2SeqModel_simp(P)
        model.In = nn.Parameter(tt(LoadingVector.I, device=device))
        model.Encoder.setJ(V=tt(LoadingVector.V.T / math.sqrt(model.P['N']), device=device),
                           U=tt(LoadingVector.U / math.sqrt(model.P['N']), device=device))
        model.Out = nn.Parameter(tt(LoadingVector.W.T, device=device))
        return model

    def reg_loss(self):
        Eloss = self.Encoder.reg_loss()
        L = nn.MSELoss(reduction='sum')
        Proj_In = self.In.T @ self.In / self.P['N']
        Proj_Out = self.Out.T @ self.Out / self.P['N']
        if self.In.requires_grad:
            reg_I = L(Proj_In, torch.eye(self.P['N_I'], device=self.device))
        else:
            reg_I = torch.zeros([1], device=self.device)
        if self.Out.requires_grad:
            reg_O = L(Proj_Out, torch.eye(self.P['N_O'], device=self.device))
        else:
            reg_O = torch.zeros([1], device=self.device)

        return [reg_I, reg_O] + Eloss


# Seq2Seq model with class encoder and decoder
class Seq2SeqModel(nn.Module):
    def __init__(self, P):
        super(Seq2SeqModel, self).__init__()
        self.P = P
        self.Embedding = P['Embedding']
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.In = nn.Linear(in_features=P['in_Channel'], out_features=P['N_Encoder'], bias=False)
        self.Encoder_P = dict(N=P['N_Encoder'], R=P['R_Encoder'], act_func=P['act_func'], dt=P['dt'], tau=P['tau'],
                              g_rec=P['g_rec'], gain=P['gain'])
        self.Encoder = LeakyRNN(self.Encoder_P)
        if P['N_Encoder'] != P['N_Decoder']:
            self.W = nn.Linear(in_features=P['N_Encoder'], out_features=P['N_Decoder'], bias=self.P['Wb'])
            self.W.weight.data = tt(
                np.random.normal(loc=0, scale=1 / self.P['N_Encoder'], size=(self.P['N_Decoder'], self.P['N_Encoder'])))
        self.Decoder_P = dict(N=P['N_Decoder'])
        self.Decoder = VanillaRNN(self.Decoder_P)
        self.Out = nn.Linear(in_features=P['N_Decoder'], out_features=P['out_Channel'], bias=False)
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=False)

    def reinit(self, g_En=1., g_De=1., g_W=1.):
        self.In.weight.requires_grad = self.P['require_grad'][0]
        self.Out.weight.requires_grad = self.P['require_grad'][1]
        self.In.weight.data = tt(np.random.normal(loc=0., scale=1., size=(self.P['N_Encoder'], self.P['in_Channel'])))
        self.Encoder.reinit(g=g_En)
        self.Decoder.reinit(g=g_De)
        self.Out.weight.data = tt(
            np.random.normal(loc=0., scale=1. / self.P['N_Decoder'], size=(self.P['out_Channel'], self.P['N_Decoder'])))
        self.W.weight.data = g_W * self.W.weight.data
        if self.W.bias is not None:
            self.W.bias.data = g_W * self.W.bias.data

    def forward(self, Batch_Input, Batch_T=None, decoder_steps=3, device='cpu', get_Encoder=False, **kwargs):
        Input = self.In(Batch_Input * self.in_strength).to(device)
        Encoder_hidden = self.Encoder(Input, device=device, **kwargs)
        if Batch_T is not None:
            last_hidden = torch.cat([Encoder_hidden[i:i + 1, Batch_T[i]] for i in range(Input.shape[0])], dim=0).to(
                device)
        else:
            last_hidden = Encoder_hidden[:, -1].to(device)
        if self.P['N_Encoder'] != self.P['N_Decoder']:
            last_hidden = self.W(self.act_func(last_hidden))
        Decoder_hidden = self.Decoder(n_steps=decoder_steps, hidden_0=last_hidden, device=device)
        Output = self.out_strength * self.Out(Decoder_hidden).to(device)
        if not get_Encoder:
            return Decoder_hidden, Output
        else:
            if not Batch_T:
                return Encoder_hidden, Decoder_hidden, Output
            else:
                return Batch_T, Encoder_hidden, Decoder_hidden, Output

    def Batch2Input(self, Batch, add_param=None, device='cpu'):
        '''
        :param Batch: tensor of int (Bath_Size,length), must be of tensor type, not list
        :param add_param: extra option, keywords to replace default parameters of network
        :param device:device ran on,default='cpu'
        :return: Bacth of Input: tensor of float (Batch_Size,T,in_Channel),default In_Channel=2 (flat surface of visual input)
        '''
        Batch_Size, L = Batch.shape
        n_emb = self.Embedding.shape[0]
        P = self.P.copy()
        if add_param is not None:
            P.update(add_param)

        t_rest = P['t_rest'] + random.randint(0, P['Dt_rest'])
        t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for i in range(L)]
        t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for i in range(L - 1)]
        t_delay = P['t_delay'] + random.randint(0, P['Dt_delay'])
        t_total = t_rest + sum(t_on) + sum(t_off) + t_delay
        Input = math.sqrt(2 * self.P['tau'] / self.P['dt']) * P['g_in'] * torch.randn(
            (Batch_Size, t_total, P['in_Channel']))
        Batch_oh = Batch_onehot(n_emb, Batch)
        Batch_Input = Batch_oh @ self.Embedding
        pointer = t_rest
        for i in range(L):
            Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
            if i != L - 1:
                pointer += t_on[i] + t_off[i]
        return Input.to(device)

    def from_loading(self, LoadingVector: LoadingVector, U_W, S_W, device='cpu'):
        model = copy.deepcopy(self).to(device)
        model.P['N_Encoder'] = LoadingVector.I.shape[0]
        model.In = nn.Linear(in_features=model.P['in_Channel'], out_features=model.P['N_Encoder'], bias=False,
                             device=device)
        model.In.weight.data = tt(LoadingVector.I, device=device)
        if model.P['R_Encoder'] == -1:
            model.Encoder.setJ(J=tt(LoadingVector.U @ LoadingVector.V.T / model.P['N_Encoder'], device=device))
        else:
            model.Encoder.setJ(V=tt(LoadingVector.V.T / math.sqrt(model.P['N_Encoder']), device=device),
                               U=tt(LoadingVector.U / math.sqrt(model.P['N_Encoder']), device=device))
        model.W = nn.Linear(in_features=model.P['N_Encoder'], out_features=model.P['N_Decoder'], bias=False,
                            device=device)
        model.W.weight.data = tt(U_W, device=device) @ torch.diag(tt(S_W, device=device)
                                                                  ) @ tt(LoadingVector.W.T, device=device) / math.sqrt(U_W.shape[0])
        return model


class LeakyRNN(nn.Module):
    def __init__(self, P):
        super(LeakyRNN, self).__init__()
        self.P = P
        self.device = P['device']
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        if self.P['R'] == -1:
            self.J = nn.Parameter(torch.randn(P['N'], P['N']), requires_grad=True)
        else:
            self.V = nn.Parameter(torch.randn(P['N'], P['R']), requires_grad=True)
            self.U = nn.Parameter(torch.randn(P['N'], P['R']), requires_grad=True)
        self.hidden_0 = nn.Parameter(torch.zeros(P['N'], dtype=torch.float), requires_grad=False)

    def reinit(self, g=1.):
        if self.P['R'] == -1:
            self.J.data = g * tt(
                np.random.normal(loc=0., scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['N'])))
        else:
            self.V.data = g * tt(
                np.random.normal(loc=0, scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['R'])))
            self.U.data = g * tt(
                np.random.normal(loc=0, scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['R'])))

    def gradient(self, hidden: torch.Tensor, Input=None, Noise_rec=None, device='cpu') -> torch.Tensor:
        '''
        computation of gradient under given hidden state and Input series
        :param hidden: given hidden state,should be of shape((shape_),N_Encoder)
        :param Input: None if not given, should be of shape((shape_),N_Encoder)
        :return:
        grad of Encoder under Input ((shape_),N_Encoder)
        '''
        if Input is None:
            Input = torch.zeros_like(hidden).to(device)
        if Noise_rec is None:
            Noise_rec = torch.zeros_like(hidden).to(device)
        # gradient is computed with nondimensionalized t_=t/tau
        # alpha = Encoder.P['dt'] / Encoder.P['tau']
        if self.P['R'] == -1:
            gradient = -hidden + self.act_func(hidden) @ self.J.T + Input + Noise_rec
        else:
            gradient = -hidden + self.act_func(hidden) @ (self.V @ self.U.T) / self.P['N'] + Input + Noise_rec
        return gradient

    def forward(self, Input, device='cpu', **kwargs):
        '''
        Input to Low rank RNN hidden state through encoder
        :param Batch_Input: Batch of Input tensor of float (Batch_Size,T,in_Channel),default In_Channel=2
        :param device: device ran on,default='cpu'
        :param kwargs: extra option,keywords to replace default parameters of network
        :return: hidden trajectory of low rank network
        '''
        Input = Input.to(device)
        # given initial hidden state or default hidden state
        if 'hidden_0' in kwargs:
            hidden = kwargs['hidden_0']
        else:
            hidden = self.hidden_0
        hidden = hidden.unsqueeze(dim=0).repeat(Input.shape[0], 1).to(device)
        hidden_t = hidden.unsqueeze(dim=1)

        # recurrent hidden state computation
        alpha = self.P['dt'] / self.P['tau']
        if 'g_rec' in kwargs:
            g_rec = kwargs['g_rec']
        else:
            g_rec = self.P['g_rec']
        Noise_rec = math.sqrt(2 * self.P['tau'] / self.P['dt']) * g_rec * torch.randn(Input.shape).to(device)
        for i in range(Input.shape[1]):
            hidden += alpha * self.gradient(hidden, Input=Input[:, i], Noise_rec=Noise_rec[:, i], device=device)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    # def __getattr__(self, item):
    #     return None

    def setJ(self, J=None, V=None, U=None, device='cpu'):
        if J is not None:
            self.J.data = J.to(device)
        else:
            self.V.data = V.to(device)
            self.U.data = U.to(device)

    def test(self):
        if self.P['R'] == -1:
            self.J.requires_grad = False
        else:
            self.V.requires_grad = False
            self.U.requires_grad = False

    def reg_loss(self):
        L = nn.MSELoss(reduction='sum')

        if self.P['R'] == -1:
            if self.J.requires_grad:
                return [L(self.J, torch.zeros_like(self.J))]
            else:
                return [torch.zeros([1])]
        else:
            Proj_V = self.V.T @ self.V / self.P['N']
            Proj_U = self.U.T @ self.U / self.P['N']
            if self.V.requires_grad:
                reg_V = L(Proj_V, torch.eye(self.P['R'], device=self.device))
            else:
                reg_V = torch.zeros([1], device=self.device)
            if self.U.requires_grad:
                reg_U = L(Proj_U, torch.eye(self.P['R'], device=self.device))
            else:
                reg_U = torch.zeros([1], device=self.device)
            return [reg_U, reg_V]


class VanillaRNN(nn.Module):
    def __init__(self, P):
        super(VanillaRNN, self).__init__()
        self.P = P
        if 'act_func' in self.P:
            self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.Q = nn.Linear(in_features=P['N'], out_features=P['N'], bias=False)
        self.hidden_0 = torch.zeros(P['N'], dtype=torch.float)

    def reinit(self, g=1.):
        self.Q.weight.data = g * tt(
            np.random.normal(loc=0, scale=1 / math.sqrt(self.P['N']), size=(self.P['N'], self.P['N'])))

    def forward(self, Input=None, n_steps=None, hidden_0=None, device='cpu'):
        '''
        :param Input: external input (Batch_Size,T,N) if given
        :param n_steps: trial length if given
        :param hidden_0: initial hidden state if given
        :param device:
        :return:
        '''

        if Input is None:
            hidden = hidden_0.to(device)
            Batch_Input = torch.zeros(hidden.shape[0], n_steps, hidden.shape[1]).to(device)
        else:
            if hidden_0 is None:
                hidden = self.hidden_0.unsqueeze(dim=0).repeat(Input.shape[0], 1).to(device)
            else:
                hidden = hidden_0.to(device)
            Batch_Input = Input.to(device)
        hidden_t = hidden.unsqueeze(dim=1)
        for i in range(Batch_Input.shape[1]):
            if 'act_func' in self.P:
                hidden = self.Q(self.act_func(hidden + Batch_Input[:, i]))
            else:
                hidden = self.Q(hidden + Batch_Input[:, i])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def test(self):
        self.Q.weight.requires_grad = False


##########################################################
#                                                        #
#               Mean Field Dynamics                      #
#                                                        #
##########################################################

class reparameterlizedRNN_sample(nn.Module):
    randomsample = 1024

    def __init__(self, N_pop, N_I, N_R, N_O, act_func, tau=100):
        super(reparameterlizedRNN_sample, self).__init__()
        self.act_func = eval('torch.nn.' + act_func + '()')
        self.tau = tau
        # Initialization of dimensionality and population numbers
        self.N_pop = N_pop
        self.total_pop = self.N_pop
        self.N_I = N_I
        self.N_R = N_R
        self.N_O = N_O
        self.N_F = self.N_I + 2 * self.N_R + self.N_O
        # Initialization of Statistics hyperparameters
        self.G = nn.Parameter(torch.ones(self.N_pop))
        self.mu_I = nn.Parameter(torch.zeros(self.N_pop, self.N_I), requires_grad=False)
        self.mu_R = nn.Parameter(torch.zeros(self.N_pop, 2 * self.N_R), requires_grad=False)
        self.mu_O = nn.Parameter(torch.zeros(self.N_pop, self.N_O), requires_grad=False)
        self.C_I = nn.Parameter(torch.zeros(self.N_pop, self.N_I, self.N_F), requires_grad=False)
        self.C_R = nn.Parameter(torch.zeros(self.N_pop, 2 * self.N_R, self.N_F), requires_grad=False)
        self.C_O = nn.Parameter(torch.zeros(self.N_pop, self.N_O, self.N_F), requires_grad=False)
        # Initialization of adjacent Mask variables
        self.Mask_G = torch.ones(self.N_pop)
        self.Mask_mu = torch.ones(self.N_pop, self.N_F)
        self.Mask_C = torch.ones(self.N_pop, self.N_F, self.N_F)
        # get axes indices
        self.I_index = list(range(self.N_I))
        self.U_index = list(range(self.N_I, self.N_I + self.N_R))
        self.V_index = list(range(self.N_I + self.N_R, self.N_I + 2 * self.N_R))
        self.O_index = list(range(self.N_F - self.N_O, self.N_F))
        # get readout Amplification
        self.Out_Amplification = torch.ones(self.N_O)
        # To generate z from recurrent selective vector V
        self.no_z = False

    def get_total_pop(self):
        return self.total_pop

    def set_no_z(self):
        self.no_z = True
        self.gen_O = torch.zeros(self.N_O, 2 * self.N_R)
        self.gen_O[:, self.N_R:self.N_R + self.N_O] = torch.eye(self.N_O)

    def Mask_to(self, device):
        self.Mask_G.to(device)
        self.Mask_mu.to(device)
        self.Mask_C.to(device)

    # reinitialization of network through additional hyperparameters for specific task requirements
    def reinit(self, **kwargs):
        keys = ['G', 'mu_I', 'mu_R', 'mu_O', 'C_I', 'C_R', 'C_O']
        n_keys = len(keys)
        layers = [self.G, self.mu_I, self.mu_R, self.mu_O, self.C_I, self.C_R, self.C_O]
        # reinitialization of trainable layers
        trainable_keys = [key + '_train' for key in keys]
        for idx in range(n_keys):
            if trainable_keys[idx] in kwargs:
                layers[idx].requires_grad = bool(kwargs[trainable_keys[idx]])
        # reinitialization of weights in trainable layers
        sigma_keys = ['g_' + key for key in keys]
        for idx in range(n_keys):
            if sigma_keys[idx] in kwargs:
                layers[idx].data = kwargs[sigma_keys[idx]] * torch.randn(layers[idx].shape)
        # reinitialization of weights through direct given values
        value_keys = ['w_' + key for key in keys]
        for idx in range(n_keys):
            if value_keys[idx] in kwargs:
                layers[idx].data = kwargs[value_keys[idx]]
        # reinitialization of Mask values:
        Mask_keys = ['Mask_' + key for key in ['G', 'mu', 'C']]
        Masks = [self.Mask_G, self.Mask_mu, self.Mask_C]
        for idx in range(3):
            if Mask_keys[idx] in kwargs:
                Masks[idx] = kwargs[Mask_keys[idx]]
        # reinitialization of readout Amplification
        if 'Out_Amplification' in kwargs:
            self.Out_Amplification = kwargs['Out_Amplification']

    def get_alpha_p(self, device='cpu'):
        alpha_p = torch.softmax(self.G_mod(device=device), dim=0)
        return alpha_p * self.Mask_G_mod(device=device) / (sum(alpha_p * self.Mask_G_mod(device=device)))

    def get_mu(self, device='cpu'):
        mu = torch.cat(self.mu_mod(device=device), dim=1)
        return mu * self.Mask_mu_mod(device=device)

    def get_C(self, device='cpu'):
        C = torch.cat(self.C_mod(device=device), dim=1)
        return C * self.Mask_C_mod(device=device)

    def get_cov(self, device='cpu'):
        cov = torch.einsum('pij,pkj->pik', self.get_C(device=device), self.get_C(device=device))
        return cov

    def get_Overlap(self, reduction_p=True, device='cpu'):
        Overlap_p = torch.einsum('pu,pv->puv', self.get_mu(device=device), self.get_mu(device=device)) + self.get_cov(
            device=device)
        if reduction_p:
            return torch.einsum('p,puv->uv', self.get_alpha_p(device=device), Overlap_p)
        else:
            return Overlap_p

    def G_mod(self, device='cpu'):
        return self.G.to(device)

    def mu_mod(self, device='cpu'):
        mu_O = self.mu_O.to(device)
        if hasattr(self, 'no_z'):
            if self.no_z:
                mu_O = torch.einsum('or,pr->po', self.gen_O.to(device), self.mu_R.to(device))
        return [self.mu_I.to(device), self.mu_R.to(device), mu_O]

    def C_mod(self, device='cpu'):
        C_O = self.C_O.to(device)
        if hasattr(self, 'no_z'):
            if self.no_z:
                C_O = torch.einsum('or,pry->poy', self.gen_O.to(device), self.C_R.to(device))
        return [self.C_I.to(device), self.C_R.to(device), C_O]

    def Mask_G_mod(self, device='cpu'):
        return self.Mask_G.to(device)

    def Mask_mu_mod(self, device='cpu'):
        return self.Mask_mu.to(device)

    def Mask_C_mod(self, device='cpu'):
        return self.Mask_C.to(device)

    def reset_noise_loading(self, device='cpu'):
        self.noise_loading = torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F,
                                         device=device)

    def get_noise_loading(self, device='cpu'):
        '''
        :return: samples of shape (N_pop, n_samples, dynamical dimensions)
        '''
        return torch.randn(self.total_pop, reparameterlizedRNN_sample.randomsample, self.N_F, device=device)

    # def forward(self, hidden: torch.Tensor, Input_t: torch.Tensor, g_rec=None, device='cpu'):
    #     Batch_Size,T=Input_t.shape[:2]
    #     if hidden is None: hidden=

    def get_loading(self, device='cpu'):
        '''
        :return: loading vectors of multipopulations with shape (N_pop, n_samples, dynamical dimensions)
        '''
        mu = self.get_mu(device=device)
        C = self.get_C(device=device)
        loadingvectors = torch.einsum('pnr,pkr->pnk', self.noise_loading.to(device), C) + mu.view(self.total_pop, 1,
                                                                                                  self.N_F)
        return loadingvectors

    def set_axes(self, device='cpu'):
        loadingvectors = self.get_loading(device=device)
        self._I = loadingvectors[:, :, self.I_index]
        self._U = loadingvectors[:, :, self.U_index]
        self._V = loadingvectors[:, :, self.V_index]
        self._O = loadingvectors[:, :, self.O_index]

    def get_Statistics(self):
        alpha_p = self.get_alpha_p()
        mu = self.get_mu()
        cov = self.get_cov()
        return table(I_index=self.I_index, U_index=self.U_index, V_index=self.V_index, O_index=self.O_index,
                     alpha_p=alpha_p.detach().clone(), mu=mu.detach().clone(), cov=cov.detach().clone())

    def forward(self, Input_t: torch.Tensor, hidden=None, dt=10, g_rec=0.02, get_Encoder=False, device='cpu'):
        Batch_Size, T = Input_t.shape[:2]
        Input_t = Input_t.to(device)
        alpha_p = self.get_alpha_p(device=device)
        if hidden is None:
            hidden = torch.zeros(self.total_pop, reparameterlizedRNN_sample.randomsample, device=device)
        # set axes of I,U,V,O with shape (N_pop,n_samples,n_ranks)
        self.set_axes(device=device)

        hidden_t = torch.zeros(Batch_Size, T + 1, self.total_pop, reparameterlizedRNN_sample.randomsample,
                               device=device)
        hidden_t[:, 0] = hidden

        alpha = dt / self.tau
        Noise_rec = math.sqrt(2 / alpha) * g_rec * torch.randn(hidden_t.shape, device=device)
        for i in range(T):
            External = torch.einsum('bi,pni->bpn', Input_t[:, i], self._I)
            Selection = torch.einsum('p,pnr,bpn->br', alpha_p, self._V,
                                     self.act_func(hidden_t[:, i])) / self.randomsample
            Recurrence = torch.einsum('br,pnr->bpn', Selection, self._U)
            hidden_t[:, i + 1] = (1 - alpha) * hidden_t[:, i] + alpha * (
                Recurrence + External + Noise_rec[:, i])

        out_t = torch.einsum('p,pno,btpn->bto', alpha_p, self._O, self.act_func(hidden_t)) * self.Out_Amplification.to(
            device) / self.randomsample
        if not get_Encoder:
            return out_t
        else:
            return hidden_t, out_t

    def hidden_align_U(self, hidden, with_kappa_I=False, normalize=False, device='cpu'):
        '''
        :param normalize: True->change to normalized axes
        '''
        # notica that in this function,hidden must be reshaped as '...pn' to fit _U
        weights = self.get_alpha_p(device=device)
        Norm = 1
        if normalize:
            Norm = 1 / 2
        U_pow2_norm_inv = (torch.einsum('p,pnr,pnr->r', weights, self._U.to(device),
                           self._U.to(device)) / self.randomsample)**(-Norm)
        kappa = torch.einsum(
            'p,pnr,...pn,r->...r',
            weights,
            self._U.to(device),
            hidden.to(device),
            U_pow2_norm_inv) / self.randomsample
        if not with_kappa_I:
            return kappa
        else:
            I_pow2_norm_inv = (torch.einsum('p,pni,pni->i', weights, self._I.to(device),
                               self._I.to(device)) / self.randomsample) ** (-Norm)
            kappa_I = torch.einsum(
                'p,pnr,...pn,r->...r',
                weights,
                self._I.to(device),
                hidden.to(device),
                I_pow2_norm_inv) / self.randomsample
            return kappa, kappa_I

    def hidden_align_U_non_ortho(self, hidden, device='cpu'):
        '''
        A non-orthogonal solution of collective value for (kappa_I,kappa_r) [batch_size,T,dim]
        '''
        _UI = torch.cat((self._I, self._U), dim=-1).to(device)
        pinv_UI = torch.linalg.pinv(_UI.reshape(-1, _UI.shape[-1]))
        kappa_f = hidden.to(device).reshape(*hidden.shape[:-2], -1) @ pinv_UI.T
        return kappa_f

    def kappa_align_U(self, kappa, kappa_I=None, normalize=False, device='cpu'):
        Norm = 0
        if normalize:
            Norm = 1 / 2
        U_pow2_norm_inv = (torch.einsum('p,pnr,pnr->r', weights, self._U.to(device),
                           self._U.to(device)) / self.randomsample)**(-Norm)
        hidden = torch.einsum('...r,pnr,r->...pn', kappa.to(device), self._U.to(device), U_pow2_norm_inv)
        if kappa_I is None:
            return hidden
        else:
            I_pow2_norm_inv = (torch.einsum('p,pni,pni->i', weights, self._I.to(device),
                               self._I.to(device)) / self.randomsample) ** (-Norm)
            hidden += torch.einsum('...i,pni,i->...pn', kappa_I.to(device), self._I.to(device), I_pow2_norm_inv)
            return hidden

    def gradient(self, hidden, Input=None, device='cpu'):
        # hidden size(Batch_Size,population,N)
        Selection = torch.einsum('p,pnr,...pn->...r', self.get_alpha_p(device=device), self._V.to(device),
                                 self.act_func(hidden.to(device))) / self.randomsample
        Recurrence = torch.einsum('...r,pnr->...pn', Selection, self._U.to(device))
        if Input is None:
            gradient = Recurrence
        else:
            External = torch.einsum('...i,pni->...pn', Input.to(device), self._I.to(device))
            gradient = Recurrence + External
        return -hidden.to(device) + gradient

    # duprecated
    def reg_loss_p(self, device='cpu'):
        # regularization for overlap of multi-population
        def MSE0(x):
            L = nn.MSELoss(reduction='sum')
            return L(x, torch.zeros_like(x))

        def diag3d(x):
            return torch.cat([torch.diag(y.diag()).unsqueeze(dim=0) for y in x], dim=0)

        def eye3d(x):
            return torch.eye(x.shape[1], device=x.device).unsqueeze(dim=0).repeat(x.shape[0], 1, 1)

        Overlap_p = self.get_Overlap(reduction_p=False, device=device)
        Overlap = self.get_Overlap(reduction_p=True, device=device)

        O_I_p = Overlap_p[:, :self.N_I, :self.N_I]
        O_W_p = Overlap_p[:, self.N_I + 2 * self.N_R:, self.N_I + 2 * self.N_R:]
        O_IU_p = Overlap_p[:, self.N_I, self.N_I:self.N_I + self.N_R]
        O_IW_p = Overlap_p[:self.N_I, self.N_I + 2 * self.N_R:]
        O_U = Overlap[self.N_I:self.N_I + self.N_R, self.N_I:self.N_I + self.N_R]
        O_V = Overlap[self.N_I + self.N_R:self.N_I + 2 * self.N_R, self.N_I + self.N_R:self.N_I + 2 * self.N_R]

        # regularization loss computation for I,U,V,W respectively
        if self.P['reg_I'] == 'ortho':
            reg_I = torch.einsum('p,puv->puv', self.get_alpha_p(), O_I_p - diag3d(O_I_p))
        elif self.P['reg_I'] == 'norm_ortho':
            reg_I = torch.einsum('p,puv->puv', self.get_alpha_p(), O_I_p - eye3d(O_I_p))
        else:
            reg_I = torch.zeros(O_I_p.shape, device=device)

        reg_U = torch.einsum('uv,uv->uv', O_U, 1 - self.P['reg_U'] * torch.eye(O_U.shape[0], device=device))
        reg_V = torch.einsum('uv,uv->uv', O_V, 1 - self.P['reg_V'] * torch.eye(O_V.shape[0], device=device))
        if self.P['reg_W'] == 'ortho':
            reg_W = torch.einsum('p,puv->puv', self.get_alpha_p(), O_W_p - diag3d(O_W_p))
        elif self.P['reg_I'] == 'norm_ortho':
            reg_W = torch.einsum('p,puv->puv', self.get_alpha_p(), O_W_p - eye3d(O_W_p))
        else:
            reg_W = torch.zeros(O_W_p.shape, device=device)
        reg_IU = torch.einsum('p,puv->puv', self.get_alpha_p(), O_IU_p)
        reg_IW = torch.einsum('p,puv->puv', self.get_alpha_p(), O_IW_p)
        return [MSE0(reg) for reg in [reg_I, reg_U, reg_V, reg_W, reg_IU, reg_IW]]

    def reg_loss(self, type='L2', device='cpu'):
        layers = [layer.to(device) for layer in [self.mu_I, self.mu_R, self.mu_O, self.C_I, self.C_R, self.C_O]]
        if type == 'L2':
            Loss = nn.MSELoss(reduction='sum')
        elif type == 'L1':
            Loss = nn.L1Loss(reduction='sum')

        def Loss0(x):
            return Loss(x, torch.zeros_like(x))

        return sum([Loss0(param) for param in layers])

    # We should get all U vectors orthogonal
    def ortho_loss(self, I=False, U=False, V=False, W=False, IU=False, IW=False, corr_only=False, d_I=1., d_R=0.95,
                   d_O=1., device='cpu', Mask_IV=None):
        '''
        :param I:
        :param U:
        :param V:
        :param W:
        :param IU:
        :param IW:
        :param corr_only: If consider correlation only and ignore amplitude,
                        while corr_only=False means constrain amplitude using d_I,d_R,d_O,
                        d=1 means no constrain and d=0 means equivalent constrain on diagonal elemnts with off-diagonal element.
        :param d_I:
        :param d_R:
        :param d_O:
        :param device:
        :return:
        '''

        def MSE0(x):
            L = nn.MSELoss(reduction='sum')
            return L(x, torch.zeros_like(x))

        Overlap = self.get_Overlap(reduction_p=True, device=device)

        # The I,U,V,W Overlap
        O_I = Overlap[self.I_index][:, self.I_index]
        O_I_od = d_I * (O_I - torch.eye(self.N_I, device=device))
        O_U = Overlap[self.U_index][:, self.U_index]
        O_U_od = O_U - d_R * torch.diag(O_U.diag())
        O_V = Overlap[self.V_index][:, self.V_index]
        O_V_od = O_V - d_R * torch.diag(O_V.diag())
        O_W = Overlap[self.O_index][:, self.O_index]
        O_W_od = d_O * (O_W - torch.eye(self.N_O, device=device))
        O_IU = Overlap[self.I_index][:, self.U_index]
        O_IW = Overlap[self.I_index][:, self.O_index]
        if Mask_IV is None:
            Mask_IV = torch.zeros(self.N_I, self.N_R).to(device)
        else:
            Mask_IV = Mask_IV.to(device)
        O_IV = Overlap[self.I_index][:, self.V_index] * Mask_IV
        keys = [I, U, V, W, IU, IW, True]
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
        Ortho_Loss = sum([float(keys[idx]) * MSE0(values[idx]) for idx in range(len(keys))])

        # Ortho_Loss = sum(
        #     [torch.tensor(keys[idx], dtype=torch.long, device=device) * MSE0(values[idx]) for idx in range(len(keys))])
        return Ortho_Loss

    def entropy(self):
        # add entropy loss to reduce population numbers
        # sub entropy loss to increase population numbers
        return Categorical(probs=self.get_alpha_p()).entropy()


def expand_class(classes, pop_expand, device='cpu'):
    return torch.cat([classes[i:i + 1].repeat_interleave(pop_expand[i], dim=0)
                      for i in range(len(pop_expand))], dim=0).to(device)


def mod_dim(M, n_dim, device='cpu'):
    return M.reshape(M.shape[0], int(M.shape[1] / n_dim), n_dim, *M.shape[2:]).to(device)


def mod_population(statistic, n_dim, pop_mod, device='cpu'):
    statistic_mod = torch.einsum('ry...,prxy->prx...', mod_dim(statistic, n_dim, device=device),
                                 pop_mod.to(device)).reshape(pop_mod.shape[0], *statistic.shape[1:])
    return statistic_mod


def assign_withNone(value, default):
    if value is None:
        return default
    else:
        return value


g_init = 0.001


def uns(x, dim=0):
    return x.unsqueeze(dim=dim)


def list_proj_2d(amp_list, angle_list, device='cpu'):
    angle_cos = uns(angle_list.cos(), -1).to(device)
    angle_sin = uns(angle_list.sin(), -1).to(device)
    phase_matrix = torch.cat([angle_cos, angle_sin, -angle_sin, angle_cos], dim=-1)
    return torch.einsum('...,...x->...x', amp_list.to(device), phase_matrix).reshape(*angle_list.shape, 2, 2)


class reparameterlizedRNN_sample_hierarchy(reparameterlizedRNN_sample):

    # by defining a hierachy network,first initializing, then reinit scale, finally reinit statistic

    def reinit_scale(self, dim_expand=None, pop_expand=None):
        if dim_expand is not None:
            self.set_dim_expand(dim_expand)
        if pop_expand is not None:
            self.set_pop_expand(pop_expand)

    def set_dim_expand(self, dim_expand):
        self.dim_expand = dim_expand

    def set_pop_expand(self, pop_expand):
        self.pop_expand = pop_expand
        self.total_classes = len(self.pop_expand)
        self.total_pop = sum(self.pop_expand)

    def reinit_statistic(self, R_param=None, I_param=None, O_param=None):
        '''
        :param R_param: a dict containing keys: 'pop_mod_R','mu_R','C_R'
        :param I_param: a dict containing keys: 'pop_mod_I','mu_I','C_I'...
        :param O_param: a dict containing keys: 'pop_mod_O','mu_O','C_O'...
        '''
        if R_param is not None:
            self.set_get_R_mode(R_param['pop_mod_R'], R_param['mu_R'], R_param['C_R'])
        if I_param is not None:
            self.set_get_I_mode(I_param['get_I_mode'], param=I_param)
        if O_param is not None:
            self.set_get_O_mode(O_param['get_O_mode'], param=O_param)

    def set_get_R_mode(self, pop_mod_R, mu_R, C_R):
        assert len(pop_mod_R) == self.total_classes
        for c in range(self.total_classes):
            assert pop_mod_R[c].shape == (self.pop_expand[c], int(
                2 * self.N_R / self.dim_expand), self.dim_expand, self.dim_expand)
        self.pop_mod_R = pop_mod_R
        self.mu_R = nn.Parameter(assign_withNone(mu_R, g_init * torch.randn(self.total_classes, 2 * self.N_R)))
        self.C_R = nn.Parameter(assign_withNone(C_R, g_init * torch.randn(self.total_classes, 2 * self.N_R, self.N_F)))

    def set_get_I_mode(self, get_I_mode, param=None):
        '''
        :param get_I_mode: in 'default','from_R','symmetry'
        in 'default' mode, mu_I,C_I should be of shape (n_pops,...),using key 'mu_I','C_I'
        in 'from_R' mode, a matrix R_to_I should be given,using key 'Mask_R_to_I','R_to_I' for dim_expand=1,'R_to_I_amp/phase' for dim_expand=2
        in 'symmetry' mode, a symmetry matrix pop_mod_O should be given,using key 'pop_mod_I','mu_I','C_I'
        '''
        self.get_I_mode = get_I_mode
        if self.get_I_mode == 'default':
            self.mu_I = nn.Parameter(assign_withNone(param['mu_I'], torch.randn(self.total_pop, self.N_I)))
            self.C_I = nn.Parameter(assign_withNone(param['C_I'], torch.randn(self.total_pop, self.N_I, self.N_F)))
        elif (self.get_I_mode == 'from_R') or (self.get_I_mode == 'from_pR'):
            # transformation from recurrent sta to Input are the same/different among classes:
            assert param is not None
            if self.get_I_mode == 'from_R':
                shape = (int(self.N_I / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            else:
                shape = (len(self.total_classes), int(self.N_I / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            self.Mask_R_to_I = assign_withNone(param['Mask_R_to_I'], torch.randn(*shape))
            if self.dim_expand == 1:
                assert param['R_to_I'].shape == shape
                self.R_to_I = nn.Parameter(param['R_to_I'])
            elif self.dim_expand == 2:
                assert param['R_to_I_amp'].shape == shape
                assert param['R_to_I_phase'].shape == shape
                self.R_to_I_amp = nn.Parameter(assign_withNone(param['R_to_I_amp'], torch.randn(*shape)))
                self.R_to_I_phase = nn.Parameter(assign_withNone(param['R_to_I_phase'], torch.randn(*shape)))
            else:
                raise ValueError(f'get_I_mode={self.get_I_mode}: The method for expanding dimensions is not defined!!!')
        elif self.get_I_mode == 'symmetry':
            assert param is not None
            assert 'pop_mod_O' in param
            self.mu_I = nn.Parameter(assign_withNone(param['mu_I'], torch.randn(self.total_classes, self.N_I)))
            self.C_I = nn.Parameter(assign_withNone(param['C_I'], torch.randn(self.total_classes, self.N_I, self.N_F)))
            assert len(param['pop_mod_I']) == self.total_classes
            for c in range(self.total_classes):
                assert param['pop_mod_I'][c].shape == (self.pop_expand[c], int(
                    self.N_I / self.dim_expand), self.dim_expand, self.dim_expand)
            self.pop_mod_I = param['pop_mod_I']
        else:
            raise KeyError('Correct key not found in get_I_mode!!!')

    def set_get_O_mode(self, get_O_mode, param=None):
        '''
        :param get_O_mode: in 'default','from_R','symmetry'
        in 'default' mode, mu_O,C_O should be of shape (n_pops,...),using key 'mu_O','C_O'
        in 'from_R' mode, a matrix R_to_O should be given,using key 'Mask_R_to_O','R_to_O' for dim_expand=1,'R_to_O_amp/phase' for dim_expand=2
        in 'symmetry' mode, a symmetry matrix pop_mod_O should be given,using key 'pop_mod_O','mu_O','C_O'
        '''
        self.get_O_mode = get_O_mode
        if self.get_O_mode == 'default':
            self.mu_O = nn.Parameter(assign_withNone(param['mu_O'], torch.randn(self.total_pop, self.N_O)))
            self.C_O = nn.Parameter(assign_withNone(param['C_O'], torch.randn(self.total_pop, self.N_O, self.N_F)))
        elif (self.get_O_mode == 'from_R') or (self.get_O_mode == 'from_pR'):
            # transformation from recurrent sta to readout are the same/different among classes:
            assert param is not None
            if self.get_O_mode == 'from_R':
                shape = (int(self.N_O / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            else:
                shape = (len(self.total_classes), int(self.N_O / self.dim_expand), int(2 * self.N_R / self.dim_expand))
            self.Mask_R_to_O = assign_withNone(param['Mask_R_to_O'], torch.randn(*shape))
            if self.dim_expand == 1:
                assert param['R_to_O'].shape == shape
                self.R_to_O = nn.Parameter(param['R_to_O'])
            elif self.dim_expand == 2:
                assert param['R_to_O_amp'].shape == shape
                assert param['R_to_O_phase'].shape == shape
                self.R_to_O_amp = nn.Parameter(assign_withNone(param['R_to_O_amp'], torch.randn(*shape)))
                self.R_to_O_phase = nn.Parameter(assign_withNone(param['R_to_O_phase'], torch.randn(*shape)))
            else:
                raise ValueError(f'get_O_mode={self.get_O_mode}: The method for expanding dimensions is not defined!!!')

        elif self.get_O_mode == 'symmetry':
            assert param is not None
            assert 'pop_mod_O' in param
            self.mu_O = nn.Parameter(assign_withNone(param['mu_O'], torch.randn(self.total_classes, self.N_O)))
            self.C_O = nn.Parameter(assign_withNone(param['C_O'], torch.randn(self.total_classes, self.N_O, self.N_F)))
            assert len(param['pop_mod_O']) == self.total_classes
            for c in range(self.total_classes):
                assert param['pop_mod_O'][c].shape == (self.pop_expand[c], int(
                    self.N_O / self.dim_expand), self.dim_expand, self.dim_expand)
            self.pop_mod_O = param['pop_mod_O']
        else:
            raise KeyError('Correct key not found in get_O_mode!!!')

    def get_total_pop(self):
        return self.total_pop

    def G_mod(self, device='cpu'):
        G_mod = expand_class(self.G, self.pop_expand, device=device)
        return G_mod

    def mu_mod(self, device='cpu'):
        mu_R_div = [mod_population(self.mu_R[c], self.dim_expand, self.pop_mod_R[c], device=device)
                    for c in range(self.total_classes)]
        mu_R = torch.cat(mu_R_div, dim=0)
        # generate input channel means
        if self.get_I_mode == 'default':
            mu_I = self.mu_I.to(device)
        elif self.get_I_mode == 'from_R':
            if self.dim_expand == 1:
                R_to_I = self.Mask_R_to_I.to(device) * self.R_to_I.to(device)
            elif self.dim_expand == 2:
                R_to_I = list_proj_2d(self.Mask_R_to_I * self.R_to_I_amp, self.R_to_I_phase, device=device)
            else:
                raise ValueError('mu_I_mod: The method for expanding dimensions is not defined!!!')
            mu_I = torch.einsum('ir,pr->pi', R_to_I.to(device), mu_R)
        elif self.get_I_mode == 'from_pR':
            if self.dim_expand == 1:
                R_to_I = [self.Mask_R_to_I[c].to(device) * self.R_to_I[c].to(device) for c in range(self.total_classes)]
            elif self.dim_expand == 2:
                R_to_I = [
                    list_proj_2d(
                        self.Mask_R_to_I[c].to(device) *
                        self.R_to_I_amp[c],
                        self.R_to_I_phase[c],
                        device=device) for c in range(
                        self.total_classes)]
            else:
                raise ValueError('mu_I_mod: The method for expanding dimensions is not defined!!!')

            mu_I_div = [torch.einsum('ir,pr->pi', R_to_I[c].to(device), mu_R_div[c]) for c in range(self.total_classes)]
            mu_I = torch.cat(mu_I_div, dim=0)
        else:  # symmetry
            mu_I_div = [
                mod_population(
                    self.mu_I[c],
                    self.dim_expand,
                    self.pop_mod_I[c],
                    device=device) for c in range(
                    self.total_classes)]
            mu_I = torch.cat(mu_I_div, dim=0)

        # generate output channel means
        if self.get_O_mode == 'default':
            mu_O = self.mu_O.to(device)
        elif self.get_O_mode == 'from_R':
            if self.dim_expand == 1:
                R_to_O = self.Mask_R_to_O.to(device) * self.R_to_O.to(device)
            elif self.dim_expand == 2:
                R_to_O = list_proj_2d(self.Mask_R_to_O * self.R_to_O_amp, self.R_to_O_phase, device=device)
            else:
                raise ValueError('mu_O_mod: The method for expanding dimensions is not defined!!!')
            mu_O = torch.einsum('or,pr->po', R_to_O.to(device), mu_R)
        elif self.get_O_mode == 'from_pR':
            if self.dim_expand == 1:
                R_to_O = [self.Mask_R_to_O[c].to(device) * self.R_to_O[c].to(device) for c in range(self.total_classes)]
            elif self.dim_expand == 2:
                R_to_O = [
                    list_proj_2d(
                        self.Mask_R_to_O[c].to(device) *
                        self.R_to_O_amp[c],
                        self.R_to_O_phase[c],
                        device=device) for c in range(
                        self.total_classes)]
            else:
                raise ValueError('mu_O_mod: The method for expanding dimensions is not defined!!!')

            mu_O_div = [torch.einsum('or,pr->po', R_to_O[c].to(device), mu_R_div[c]) for c in range(self.total_classes)]
            mu_O = torch.cat(mu_O_div, dim=0)
        else:  # symmetry
            mu_O_div = [
                mod_population(
                    self.mu_O[c],
                    self.dim_expand,
                    self.pop_mod_O[c],
                    device=device) for c in range(
                    self.total_classes)]
            mu_O = torch.cat(mu_O_div, dim=0)

        return [mu_I, mu_R, mu_O]

    def C_mod(self, device='cpu'):
        C_R_div = [mod_population(self.C_R[c], self.dim_expand, self.pop_mod_R[c], device=device)
                    for c in range(self.total_classes)]
        C_R = torch.cat(C_R_div, dim=0)
        # generate input channel means
        if self.get_I_mode == 'default':
            C_I = self.C_I.to(device)
        elif self.get_I_mode == 'from_R':
            if self.dim_expand == 1:
                R_to_I = self.Mask_R_to_I.to(device) * self.R_to_I.to(device)
            elif self.dim_expand == 2:
                R_to_I = list_proj_2d(self.Mask_R_to_I * self.R_to_I_amp, self.R_to_I_phase, device=device)
            else:
                raise ValueError('C_I_mod: The method for expanding dimensions is not defined!!!')
            C_I = torch.einsum('ir,pry->piy', R_to_I.to(device), C_R)
        elif self.get_I_mode == 'from_pR':
            if self.dim_expand == 1:
                R_to_I = [self.Mask_R_to_I[c].to(device) * self.R_to_I[c].to(device) for c in range(self.total_classes)]
            elif self.dim_expand == 2:
                R_to_I = [
                    list_proj_2d(
                        self.Mask_R_to_I[c].to(device) *
                        self.R_to_I_amp[c],
                        self.R_to_I_phase[c],
                        device=device) for c in range(
                        self.total_classes)]
            else:
                raise ValueError('mu_I_mod: The method for expanding dimensions is not defined!!!')

            C_I_div = [torch.einsum('ir,pry->piy', R_to_I[c].to(device), C_R_div[c]) for c in range(self.total_classes)]
            C_I = torch.cat(C_I_div, dim=0)
        else:  # symmetry
            C_I_div = [
                mod_population(
                    self.C_I[c],
                    self.dim_expand,
                    self.pop_mod_I[c],
                    device=device) for c in range(
                    self.total_classes)]
            C_I = torch.cat(C_I_div, dim=0)

        # generate output channel means
        if self.get_O_mode == 'default':
            C_O = self.C_O.to(device)
        elif self.get_O_mode == 'from_R':
            if self.dim_expand == 1:
                R_to_O = self.Mask_R_to_O.to(device) * self.R_to_O.to(device)
            elif self.dim_expand == 2:
                R_to_O = list_proj_2d(self.Mask_R_to_O * self.R_to_O_amp, self.R_to_O_phase, device=device)
            else:
                raise ValueError('mu_O_mod: The method for expanding dimensions is not defined!!!')
            C_O = torch.einsum('or,pry->poy', R_to_O.to(device), C_R)
        elif self.get_O_mode == 'from_pR':
            if self.dim_expand == 1:
                R_to_O = [self.Mask_R_to_O[c].to(device) * self.R_to_O[c].to(device) for c in range(self.total_classes)]
            elif self.dim_expand == 2:
                R_to_O = [
                    list_proj_2d(
                        self.Mask_R_to_O[c].to(device) *
                        self.R_to_O_amp[c],
                        self.R_to_O_phase[c],
                        device=device) for c in range(
                        self.total_classes)]
            else:
                raise ValueError('mu_O_mod: The method for expanding dimensions is not defined!!!')

            C_O_div = [torch.einsum('or,pry->poy', R_to_O[c].to(device), C_R_div[c]) for c in range(self.total_classes)]
            C_O = torch.cat(C_O_div, dim=0)
        else:  # symmetry
            C_O_div = [
                mod_population(
                    self.C_O[c],
                    self.dim_expand,
                    self.pop_mod_O[c],
                    device=device) for c in range(
                    self.total_classes)]
            C_O = torch.cat(C_O_div, dim=0)

        return [C_I, C_R, C_O]

    def Mask_G_mod(self, device='cpu'):
        Mask_G_mod = expand_class(self.Mask_G, self.pop_expand, device=device)
        return Mask_G_mod

    def Mask_mu_mod(self, device='cpu'):
        Mask_mu_mod = expand_class(self.Mask_mu, self.pop_expand, device=device)
        return Mask_mu_mod

    def Mask_C_mod(self, device='cpu'):
        Mask_C_mod = expand_class(self.Mask_C, self.pop_expand, device=device)
        return Mask_C_mod

    def get_alpha_p(self, device='cpu'):
        alpha_p = torch.softmax(self.G_mod(device=device), dim=0)
        return alpha_p * self.Mask_G_mod(device=device) / (sum(alpha_p * self.Mask_G_mod(device=device)))

    def get_mu(self, device='cpu'):
        mu = torch.cat(self.mu_mod(device=device), dim=1)
        return mu * self.Mask_mu_mod(device=device)

    def get_C(self, device='cpu'):
        C = torch.cat(self.C_mod(device=device), dim=1)
        return C * self.Mask_C_mod(device=device)



# train collective variable dynamics with sampling in each epochs


# Train collective variable dynamics
# duprecated in 2022.8.23
# class reparameterlizedRNN(nn.Module):
#     randomsample = 1000
#     default_param = dict(
#         act_func='Tanh',
#         N_I=1,
#         N_R=5,
#         N_O=3,
#         N_pop=6,
#         dt=10,
#         tau=100,
#         g_rec=0.15
#     )

#     def __init__(self, P):
#         super(reparameterlizedRNN, self).__init__()
#         if P == None:
#             self.P = reparameterlizedRNN.default_param
#         else:
#             self.P = P
#         self.Embedding = P['Embedding']
#         self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
#         self.N_pop = self.P['N_pop']
#         # Using original input channel so that
#         self.N_I = self.P['N_I']
#         self.N_R = self.P['N_R']
#         self.N_O = self.P['N_O']
#         self.N_F = self.N_I + 2 * self.N_R + self.N_O
#         # adding softmax nonlinear projection to generate weights of populations,[I,U,V,W]
#         self.G = nn.Parameter(torch.ones(self.N_pop))
#         # means
#         self.mu_I = nn.Parameter(torch.zeros(self.N_pop, self.N_I), requires_grad=False)
#         self.mu_R = nn.Parameter(self.P['g_mu'] * torch.randn(self.N_pop, 2 * self.N_R))
#         self.mu_O = nn.Parameter(torch.zeros(self.N_pop, self.N_O), requires_grad=False)
#         if 'mu_I_train' in self.P and self.P['mu_I_train']:
#             self.mu_I.requires_grad = True
#         if 'mu_O_train' in self.P and self.P['mu_O_train']:
#             self.mu_O.requires_grad = True
#         # triangle lower fomrs of cholesky decomposition, in training use cov=torch.tril(C)@torch.tril(C).T
#         self.C_I = nn.Parameter(torch.zeros(self.N_pop, self.N_I, self.N_F), requires_grad=False)
#         self.C_R = nn.Parameter(self.P['g_C'] * torch.randn(self.N_pop, 2 * self.N_R, self.N_F))
#         self.C_O = nn.Parameter(torch.zeros(self.N_pop, self.N_O, self.N_F), requires_grad=False)
#         if 'C_I_train' in self.P and self.P['C_I_train']:
#             self.C_I.requires_grad = True
#         if 'C_O_train' in self.P and self.P['C_O_train']:
#             self.C_O.requires_grad = True
#         self.C_I.data[:, :, :self.N_I] = torch.eye(self.N_I).unsqueeze(dim=0).repeat(self.N_pop, 1, 1)
#         self.C_O.data[:, :, -self.N_O:] = torch.eye(self.N_O).unsqueeze(dim=0).repeat(self.N_pop, 1, 1)
#         # Initializing Direct input connection weights with 1. (for I to I) and 0. (for I to U)
#         self.D_UI = nn.Parameter(torch.zeros(self.N_R, self.N_I))
#         self.D_II = nn.Parameter(torch.ones(self.N_I))
#         # Names of parameters
#         self.param_names = ['G', 'mu', 'C', 'D_UI', 'D_II']
#         self.SNDsample = torch.randn(reparameterlizedRNN.randomsample)
#         # initialize hidden and kappa_I initial state
#         self.hidden_0 = torch.zeros(self.N_R)
#         self.kappa_I_0 = torch.zeros(self.N_I)
#         # initialize nonlinear functions
#         self.G2alpha_p = nn.Softmax(dim=0)

#     @staticmethod
#     def from_model_sample(RNN_sample):
#         model = reparameterlizedRNN(RNN_sample.P)
#         mu = RNN_sample.get_mu()
#         cov = RNN_sample.get_cov()
#         alpha_p = RNN_sample.get_alpha_p()

#     def attribute_update_to_device(self, device):
#         self.Embedding = self.Embedding.to(device)
#         self.SNDsample = self.SNDsample.to(device)
#         self.hidden_0 = self.hidden_0.to(device)
#         self.kappa_I_0 = self.kappa_I_0.to(device)

#     def reinit(self, **kwargs):
#         # reinitializing parameters for given keywords' stream of layer name and data
#         for param_name in self.param_names:
#             if param_name in kwargs:
#                 self[param_name] = kwargs[param_name]

#     def __setitem__(self, key, value):
#         assert key in self.keys()
#         super().__setattr__(key, value)

#     def Batch2Input(self, Batch, sync=True, g_in=None, device='cpu'):
#         # sync defines if the training trial has the same length
#         if sync:
#             Batch_Size, L = Batch.shape
#             n_emb = self.Embedding.shape[0]
#             P = self.P.copy()
#             if g_in is not None:
#                 P['g_in'] = g_in
#             Input = math.sqrt(2 * self.P['tau'] / self.P['dt']) * P['g_in'] * torch.randn(
#                 (Batch_Size, self.P['t_upb'][L - 1], P['N_I']), device=device)
#             t_on = [P['t_on'] + random.randint(0, P['Dt_on']) for _ in range(L)]
#             t_off = [P['t_off'] + random.randint(0, P['Dt_off']) for _ in range(L - 1)]
#             Batch_oh = Batch_onehot(n_emb, Batch).to(device)
#             Batch_Input = Batch_oh @ self.Embedding
#             pointer = 0
#             for i in range(L):
#                 Input[:, pointer:pointer + t_on[i], :] += Batch_Input[:, i, :].unsqueeze(dim=1).repeat(1, t_on[i], 1)
#                 if i != L - 1:
#                     pointer += t_on[i] + t_off[i]
#             return Input
#         else:
#             return torch.cat([self.Batch2Input(Batch[i].unsqueeze(dim=0), sync=True, g_in=g_in, device=device) for i in
#                               range(Batch.shape[0])], dim=0)

#     # U vector normalization to convient analysis
#     def normalization(self):
#         model = copy.deepcopy(self)
#         Overlap_p = torch.einsum('pu,pv->puv', self.get_mu(), self.get_mu()) + self.get_cov()
#         Overlap = torch.einsum('p,puv->uv', self.get_alpha_p(), Overlap_p).detach().clone()
#         # I normalization
#         norm_I = torch.sqrt(Overlap.diag()[:self.N_I])
#         model.mu.data[:, :self.N_I] = self.mu.data[:, :self.N_I].clone() / norm_I
#         model.C.data[:, :self.N_I] = self.C.data[:, :self.N_I].clone() / norm_I
#         model.D_II.data = (self.D_II.T * norm_I).T
#         # U normalization
#         norm_U = torch.sqrt(Overlap.diag()[self.N_I:self.N_I + self.N_R])

#         model.mu.data[:, self.N_I:self.N_I + self.N_R] = self.mu.data[:, self.N_I:self.N_I + self.N_R].clone() / norm_U
#         model.mu.data[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R] = self.mu.data[:,
#                                                                         self.N_I + self.N_R:self.N_I + 2 * self.N_R].clone() * norm_U
#         model.C.data[:, self.N_I:self.N_I + self.N_R] = self.C.data[:, self.N_I:self.N_I + self.N_R].clone() / norm_U
#         model.C.data[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R] = self.C.data[:,
#                                                                        self.N_I + self.N_R:self.N_I + 2 * self.N_R].clone() * norm_U
#         model.D_UI.data = (self.D_UI.data.T * norm_U).T
#         return model

#     # generate LoadingVector from statistics
#     def sampling(self, n_samples, multinomial=False):
#         '''
#         :param n_samples: sample numbers
#         :param multinomial: If using multinomial: samples are generated by multinomial distribution of weights alpha_p,
#                             else: samples are generated by exact ratios of alpha_p for corresponding populations, which is not recommended.
#         :return:
#         '''
#         alpha_p = self.get_alpha_p()
#         mu = self.get_mu()
#         cov = self.get_cov()
#         if multinomial:
#             loading, labels = GaussMixResample(n_samples, tn(alpha_p), tn(mu), tn(cov))
#         else:
#             n_clusters = alpha_p.shape[0]
#             clusters_num = tn((n_samples * alpha_p).long())
#             loading, labels = np.array([]), np.array([])
#             for idx in range(n_clusters):
#                 loading_idx, labels_idx = GaussMixResample(clusters_num[idx], np.array([1.], ), tn(mu[idx:idx + 1]),
#                                                            tn(cov[idx:idx + 1]))
#                 loading = np.concatenate([loading, loading_idx], axis=0)
#                 labels = np.concatenate([labels, labels_idx + idx], axis=0)
#         loading[:, :self.N_I] = loading[:, :self.N_I] @ tn(torch.diag(self.D_II.data)) + loading[:,
#                                                                                          self.N_I:self.N_I + self.N_R] @ tn(
#             self.D_UI.data)
#         return loading, labels

#     def sampling2LoadingVector(self, n_samples, multinomial=False):
#         loading, labels = self.sampling(n_samples, multinomial=multinomial)
#         return LoadingVector(loading[:, :self.N_I], loading[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R],
#                              loading[:, self.N_I:self.N_I + self.N_R], W=loading[:, self.N_I + 2 * self.N_R:],
#                              labels=labels)

#     # get structural parameters

#     def from_direct_I(self):
#         return torch.cat([torch.diag(self.D_II), self.D_UI], dim=0)

#     def get_alpha_p(self):
#         return torch.softmax(self.G, dim=0)

#     def get_mu(self):
#         mu = torch.cat([self.mu_I, self.mu_R, self.mu_O], dim=1)
#         return mu

#     def compute_mu(self):
#         self.mu = torch.cat([self.mu_I, self.mu_R, self.mu_O], dim=1)

#     def get_mu_I(self):
#         return self.mu[:, :self.N_I]

#     def get_mu_U(self):
#         return self.mu[:, self.N_I:self.N_I + self.N_R]

#     def get_mu_V(self):
#         return self.mu[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R]

#     def get_mu_W(self):
#         return self.mu[:, self.N_I + 2 * self.N_R:]

#     def compute_cov(self):
#         C = torch.cat([self.C_I, self.C_R, self.C_O], dim=1)
#         self.cov = torch.cat([(C[i] @ C[i].T).unsqueeze(dim=0) for i in range(C.shape[0])], dim=0)

#     def get_cov(self):
#         C = torch.cat([self.C_I, self.C_R, self.C_O], dim=1)
#         cov = torch.cat([(C[i] @ C[i].T).unsqueeze(dim=0) for i in range(C.shape[0])], dim=0)
#         return cov

#     def get_cov_nm(self):
#         return self.cov[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R, self.N_I:self.N_I + self.N_R].clone()

#     def get_cov_nI(self):
#         return self.cov[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R, :self.N_I].clone()

#     def get_cov_wm(self):
#         return self.cov[:, self.N_I + 2 * self.N_R:, self.N_I:self.N_I + self.N_R].clone()

#     def get_cov_wI(self):
#         return self.cov[:, self.N_I + 2 * self.N_R:, :self.N_I].clone()

#     # MT sample and population statistics

#     def mu_sigma_from_hidden(self, hidden_f: torch.Tensor, mu=None, cov=None):
#         assert len(hidden_f.shape) == 2 or len(hidden_f.shape) == 3
#         if mu == None:
#             mu = self.get_mu()
#         if cov == None:
#             cov = self.get_cov()
#         if len(hidden_f.shape) == 2:
#             mu_p = torch.einsum('by,py->pb', hidden_f, mu[:, :self.N_I + self.N_R])
#             sigma_2_p = torch.einsum('by,pyz,bz->pb', hidden_f,
#                                      cov[:, :self.N_I + self.N_R, :self.N_I + self.N_R],
#                                      hidden_f)
#             return mu_p, sigma_2_p
#         elif len(hidden_f.shape) == 3:
#             mu_p = torch.einsum('bty,py->pbt', hidden_f, mu[:, :self.N_I + self.N_R])
#             sigma_2_p = torch.einsum('bty,pyz,btz->pbt', hidden_f,
#                                      cov[:, :self.N_I + self.N_R, :self.N_I + self.N_R],
#                                      hidden_f)
#             return mu_p, sigma_2_p

#     def phi(self, x: torch.Tensor) -> torch.Tensor:
#         '''
#         :param x:hidden state
#         :return: effective act_func(x) with respect to model activation functions
#         '''
#         return self.act_func(x)

#     def Dphi(self, x: torch.Tensor) -> torch.Tensor:
#         key = self.P['act_func']
#         if key == 'Tanh':
#             Dphi = 1 - self.act_func(x) ** 2
#             return Dphi
#         else:
#             raise TypeError('key value not matched in Dphi computation!!!')

#     def rho(self, x: torch.Tensor) -> torch.Tensor:
#         '''
#         :param x: hidden state
#         :return: effective act_func(x)/x with respect to model activation functions
#         '''
#         key = self.P['act_func']
#         if key == 'Tanh':
#             rho = self.act_func(x) / x
#             Drho = 1 - x ** 2 / 3
#             return torch.where(rho <= 1, rho, Drho)
#         else:
#             raise TypeError('key value not matched in rho computation!!!')

#     def MT_mean_f(self, f, mu_p, sigma_2_p):
#         n_samples = self.SNDsample.shape[0]
#         Var_g = mu_p.repeat((n_samples,) + (1,) * len(mu_p.shape)) + torch.einsum('s,...->s...', self.SNDsample.clone(),
#                                                                                   torch.sqrt(sigma_2_p + 1e-6))
#         return f(Var_g).mean(dim=0)

#     # Calculating gradients and forward update

#     # def gradient(self, hidden_f: torch.Tensor, Input: torch.Tensor, alpha_p: torch.Tensor,
#     #              Noise_rec=0.):
#     #     # return gradient of (hidden,kappa_I) under hidden state and given Input
#     #     mu_p, sigma_2_p = self.mu_sigma_from_hidden(hidden_f)
#     #     Ephi = self.MT_mean_f(self.phi, mu_p.clone(), sigma_2_p.clone())
#     #     EDphi = self.MT_mean_f(self.Dphi, mu_p.clone(), sigma_2_p.clone())
#     #     return -hidden.clone() + Noise_rec \
#     #            + self.eff_direct_UI(Input.clone(), reduction='i') \
#     #            + self.eff_mean(alpha_p.clone(), Ephi.clone(), reduction='p') \
#     #            + self.eff_cov_rec(hidden.clone(), alpha_p.clone(), EDphi.clone(), reduction='full') \
#     #            + self.eff_cov_I(kappa_I.clone(), alpha_p.clone(), EDphi.clone(), reduction='full'), \
#     #            - kappa_I.clone() + self.eff_direct_II(Input.clone())

#     def forward(self, hidden: torch.Tensor, kappa_I: torch.Tensor, Input_t: torch.Tensor, g_rec=None, device='cpu'):
#         Batch_Size, T = Input_t.shape[:2]
#         if hidden is None:
#             hidden = self.hidden_0.unsqueeze(dim=0).repeat((Batch_Size, 1))
#         if kappa_I is None:
#             kappa_I = self.kappa_I_0.unsqueeze(dim=0).repeat((Batch_Size, 1))
#         # copy to device
#         hidden_f = torch.cat([kappa_I, hidden], dim=-1).to(device)
#         Input_t = Input_t.to(device)
#         # generate population weights
#         alpha_p = self.G2alpha_p(self.G)
#         # generate means structure
#         mu = self.get_mu()
#         # generate covariance structure
#         cov = self.get_cov()

#         # generate effective direct input
#         eff_direct_I_t = torch.einsum('ui,bti->btu', self.from_direct_I(), Input_t)

#         # saving trajectories of kappa_I and hidden states
#         hidden_t = torch.zeros(Batch_Size, T + 1, self.N_I + self.N_R, device=device)
#         hidden_t[:, 0] = hidden_f
#         # initialization of phi_t and Dphi_t depend on activation functions, the following setup suit for Tanh
#         Ephi_t = torch.zeros(self.N_pop, Batch_Size, T + 1, device=device)
#         EDphi_t = torch.zeros(self.N_pop, Batch_Size, T + 1, device=device)
#         mu_p, sigma_2_p = self.mu_sigma_from_hidden(hidden_t[:, 0].clone(), mu=mu.clone(), cov=cov.clone())
#         Ephi_t[:, :, 0] = self.MT_mean_f(self.phi, mu_p.clone(), sigma_2_p.clone())
#         EDphi_t[:, :, 0] = self.MT_mean_f(self.Dphi, mu_p.clone(), sigma_2_p.clone())

#         # generate updating ratio and recurrent noise copy to device
#         P = self.P.copy()
#         if g_rec is not None:
#             P['g_rec'] = g_rec
#         alpha = P['dt'] / P['tau']
#         Noise_rec = math.sqrt(2 * P['tau'] / P['dt']) * P['g_rec'] * torch.randn(hidden_t.shape, device=device)

#         for i in range(T):
#             # H, I = self.gradient(hidden_t[:, i], kappa_I_t[:, i], Input_t[:, i], alpha_p, Noise_rec=Noise_rec[:, i])
#             # H:-hidden+noise_rec+directUI+mean+cov_rec+cov_I,I:-kappa_I+directII

#             hidden_eff_mean_p = torch.einsum('pb,pv->pbv', Ephi_t[:, :, i].clone(),
#                                              mu[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R].clone())
#             hidden_eff_cov_p = torch.einsum('pb,pvu,bu->pbv', EDphi_t[:, :, i].clone(),
#                                             cov[:, self.N_I + self.N_R:self.N_I + 2 * self.N_R,
#                                             :self.N_I + self.N_R].clone(), hidden_t[:, i].clone())
#             hidden_eff = torch.einsum('p,pbv->bv', alpha_p.clone(),
#                                       hidden_eff_mean_p.clone() + hidden_eff_cov_p.clone())
#             hidden_t[:, i + 1] = (1 - alpha) * hidden_t[:, i] + alpha * (eff_direct_I_t[:, i] + Noise_rec[:, i])
#             hidden_t[:, i + 1, self.N_I:] = hidden_t[:, i + 1, self.N_I:].clone() + alpha * hidden_eff
#             mu_p, sigma_2_p = self.mu_sigma_from_hidden(hidden_t[:, i + 1].clone(), cov=cov.clone())
#             Ephi_t[:, :, i + 1] = self.MT_mean_f(self.phi, mu_p.clone(), sigma_2_p.clone())
#             EDphi_t[:, :, i + 1] = self.MT_mean_f(self.Dphi, mu_p.clone(), sigma_2_p.clone())

#         out_eff_mean_p = torch.einsum('pbt,pw->pbtw', Ephi_t.clone(), mu[:, self.N_I + 2 * self.N_R:].clone())
#         out_eff_cov_p = torch.einsum('pbt,pwu,btu->pbtw', EDphi_t.clone(),
#                                      cov[:, self.N_I + 2 * self.N_R:, :self.N_I + self.N_R].clone(), hidden_t.clone())
#         out_t = torch.einsum('p,pbtw->btw', alpha_p.clone(), out_eff_mean_p.clone() + out_eff_cov_p.clone())
#         return hidden_t, out_t

#     # def readout(self, hidden_t: torch.Tensor, kappa_I_t: torch.Tensor):
#     #     alpha_p = self.get_alpha_p()
#     #     mu_p, sigma_2_p = self.mu_sigma_from_hidden(hidden_t.clone(), kappa_I_t.clone())
#     #     Ephi = self.MT_mean_f(self.phi, mu_p.clone(), sigma_2_p.clone())
#     #     EDphi = self.MT_mean_f(self.Dphi, mu_p.clone(), sigma_2_p.clone())
#     #     return self.out_mean(alpha_p.clone(), Ephi.clone(), reduction='p') \
#     #            + self.out_cov_rec(hidden_t.clone(), alpha_p.clone(), EDphi.clone(), reduction='full') \
#     #            + self.out_cov_I(kappa_I_t.clone(), alpha_p.clone(), EDphi.clone(), reduction='full')

#     # Calculating regularization loss

#     def reg_loss(self, device='cpu'):
#         # regularization for overlap of loading space [I,U,V,W]
#         # return regularization loss for IU,V,W respectively in a list
#         def MSE0(x):
#             L = nn.MSELoss(reduction='sum')
#             return L(x, torch.zeros_like(x))

#         mu = self.get_mu()
#         cov = self.get_cov()
#         Overlap_p = torch.einsum('pu,pv->puv', mu.clone(), mu.clone()) + cov
#         Overlap = torch.einsum('p,puv->uv', self.get_alpha_p(), Overlap_p)

#         # The IU,V,W overlap respectively
#         O_IU = Overlap[:self.N_I + self.N_R, :self.N_I + self.N_R]
#         O_V = Overlap[self.N_I + self.N_R:self.N_I + 2 * self.N_R, self.N_I + self.N_R:self.N_I + 2 * self.N_R]
#         O_W = Overlap[self.N_I + 2 * self.N_R:, self.N_I + 2 * self.N_R:]

#         # regularization loss computation for IU,V,W respectively
#         reg_IU = torch.einsum('uv,uv->uv', O_IU, 1 - self.P['reg_IU'] * torch.eye(O_IU.shape[0], device=device))
#         reg_V = torch.einsum('uv,uv->uv', O_V, 1 - self.P['reg_V'] * torch.eye(O_V.shape[0], device=device))
#         reg_W = O_W - torch.eye(O_W.shape[0], device=device)

#         return [MSE0(reg) for reg in [reg_IU, reg_V, reg_W]]

#     def direct_I_reg_loss(self):
#         # regularization for input channel overlap D
#         # return regularization loss for D_UI and D_II
#         def MSE0(x):
#             L = nn.MSELoss(reduction='sum')
#             return L(x, torch.zeros_like(x))

#         reg_direct_UI = MSE0(self.D_UI)
#         # As norm of input channel may raise , -1 will be replaced by 0 in proper time.
#         reg_direct_II = MSE0(self.D_II - 1)
#         return reg_direct_UI, reg_direct_II

#     # calculating contribution to gradients

#     def eff_mean(self, alpha_p: torch.Tensor, phi: torch.Tensor, reduction=None) -> torch.Tensor:
#         mu_V = self.get_mu_V()
#         if reduction == None:
#             return torch.einsum('p,pb...,pv->pb...v', alpha_p, phi, mu_V)
#         elif reduction == 'p':
#             return self.eff_mean(alpha_p, phi).sum(dim=0)
#         elif reduction == 'full':
#             return self.eff_mean(alpha_p, phi, reduction='p')

#     def eff_cov_rec(self, hidden: torch.Tensor, alpha_p: torch.Tensor, Dphi: torch.Tensor,
#                     reduction=None) -> torch.Tensor:
#         cov_nm = self.get_cov_nm()
#         if reduction == None:
#             return torch.einsum('p,pb...,pvu,b...u->pb...vu', alpha_p, Dphi, cov_nm, hidden)
#         elif reduction == 'p':
#             return self.eff_cov_rec(hidden, alpha_p, Dphi).sum(dim=0)
#         elif reduction == 'u':
#             return self.eff_cov_rec(hidden, alpha_p, Dphi).sum(dim=-1)
#         elif reduction == 'full':
#             return self.eff_cov_rec(hidden, alpha_p, Dphi).sum(dim=0).sum(dim=-1)

#     def eff_cov_I(self, kappa_I: torch.Tensor, alpha_p: torch.Tensor, Dphi: torch.Tensor,
#                   reduction=None) -> torch.Tensor:
#         cov_nI = self.get_cov_nI()
#         if reduction == None:
#             return torch.einsum('p,pb...,pvi,b...i->pb...vi', alpha_p, Dphi, cov_nI, kappa_I)
#         elif reduction == 'p':
#             return self.eff_cov_I(kappa_I, alpha_p, Dphi).sum(dim=0)
#         elif reduction == 'i':
#             return self.eff_cov_I(kappa_I, alpha_p, Dphi).sum(dim=-1)
#         elif reduction == 'full':
#             return self.eff_cov_I(kappa_I, alpha_p, Dphi).sum(dim=0).sum(dim=-1)

#     def eff_direct_UI(self, Input: torch.Tensor, reduction=None) -> torch.Tensor:
#         if reduction == None:
#             return torch.einsum('ui,b...i->b...ui', self.D_UI, Input)
#         elif reduction == 'i':
#             return self.eff_direct_UI(Input).sum(dim=-1)
#         elif reduction == 'full':
#             return self.eff_direct_UI(Input, reduction='i')

#     def eff_direct_II(self, Input: torch.Tensor) -> torch.Tensor:
#         return torch.einsum('i,b...i->b...i', self.D_II, Input)

#     # Calculating contribution to readout

#     def out_mean(self, alpha_p: torch.Tensor, phi: torch.Tensor, reduction=None) -> torch.Tensor:
#         mu_W = self.get_mu_W()
#         if reduction == None:
#             return torch.einsum('p,pb...,pw->pb...w', alpha_p, phi, mu_W)
#         elif reduction == 'p':
#             return self.out_mean(alpha_p, phi).sum(dim=0)
#         elif reduction == 'full':
#             return self.out_mean(alpha_p, phi, reduction='p')

#     def out_cov_rec(self, hidden: torch.Tensor, alpha_p: torch.Tensor, Dphi: torch.Tensor,
#                     reduction=None) -> torch.Tensor:
#         cov_wm = self.get_cov_wm()
#         if reduction == None:
#             return torch.einsum('p,pb...,pwu,b...u->pb...wu', alpha_p, Dphi, cov_wm, hidden)
#         elif reduction == 'p':
#             return self.out_cov_rec(hidden, alpha_p, Dphi).sum(dim=0)
#         elif reduction == 'u':
#             return self.out_cov_rec(hidden, alpha_p, Dphi).sum(dim=-1)
#         elif reduction == 'full':
#             return self.out_cov_rec(hidden, alpha_p, Dphi).sum(dim=0).sum(dim=-1)

#     def out_cov_I(self, kappa_I: torch.Tensor, alpha_p: torch.Tensor, Dphi: torch.Tensor,
#                   reduction=None) -> torch.Tensor:
#         cov_wI = self.get_cov_wI()
#         if reduction == None:
#             return torch.einsum('p,pb...,pwi,b...i->pb...wi', alpha_p, Dphi, cov_wI, kappa_I)
#         elif reduction == 'p':
#             return self.out_cov_I(kappa_I, alpha_p, Dphi).sum(dim=0)
#         elif reduction == 'i':
#             return self.out_cov_I(kappa_I, alpha_p, Dphi).sum(dim=-1)
#         elif reduction == 'full':
#             return self.out_cov_I(kappa_I, alpha_p, Dphi).sum(dim=0).sum(dim=-1)
