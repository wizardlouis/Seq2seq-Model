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
        if hidden_0 == None:
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
            self.W = nn.Linear(in_features=P['N_Encoder'], out_features=P['N_Decoder'], bias=True)
            self.W.weight.data = tt(
                np.random.normal(loc=0, scale=1 / self.P['N_Encoder'], size=(self.P['N_Decoder'], self.P['N_Encoder'])))
        self.Decoder_P = dict(N=P['N_Decoder'])
        self.Decoder = VanillaRNN(self.Decoder_P)
        self.Out = nn.Linear(in_features=P['N_Decoder'], out_features=P['out_Channel'], bias=False)
        self.in_strength = nn.Parameter(torch.ones(self.P['in_Channel']), requires_grad=False)
        self.out_strength = nn.Parameter(torch.ones(self.P['out_Channel']), requires_grad=False)

    def reinit(self,g_En=1.,g_De=1.,g_W=1.):
        self.In.weight.requires_grad = self.P['require_grad'][0]
        self.Out.weight.requires_grad = self.P['require_grad'][1]
        self.In.weight.data = tt(np.random.normal(loc=0., scale=1., size=(self.P['N_Encoder'], self.P['in_Channel'])))
        self.Encoder.reinit(g=g_En)
        self.Decoder.reinit(g=g_De)
        self.Out.weight.data = tt(
            np.random.normal(loc=0., scale=1. / self.P['N_Decoder'], size=(self.P['out_Channel'], self.P['N_Decoder'])))
        self.W.weight.data=g_W*self.W.weight.data
        if self.W.bias!=None:
            self.W.bias.data=g_W*self.W.bias.data

    def forward(self, Batch_Input, Batch_T, decoder_steps=3, device='cpu', **kwargs):
        Input = self.In(Batch_Input * self.in_strength).to(device)
        Encoder_hidden = self.Encoder(Input, device=device, **kwargs)
        last_hidden = torch.cat([Encoder_hidden[i:i + 1, Batch_T[i]] for i in range(Input.shape[0])], dim=0).to(device)
        if self.P['N_Encoder'] != self.P['N_Decoder']:
            last_hidden = self.W(self.act_func(last_hidden))
        Decoder_hidden = self.Decoder(n_steps=decoder_steps, hidden_0=last_hidden, device=device)
        Output = self.out_strength * self.Out(Decoder_hidden).to(device)
        return Decoder_hidden, Output

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


class LeakyRNN(nn.Module):
    def __init__(self, P):
        super(LeakyRNN, self).__init__()
        self.P = P
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        if self.P['R'] == -1:
            self.J = nn.Linear(in_features=P['N'], out_features=P['N'], bias=False)
        else:
            self.V = nn.Linear(in_features=P['N'], out_features=P['R'], bias=False)
            self.U = nn.Linear(in_features=P['R'], out_features=P['N'], bias=False)
        self.hidden_0 = torch.zeros(P['N'], dtype=torch.float)

    def reinit(self,g=1.):
        if self.P['R'] == -1:
            self.J.weight.data = g*tt(
                np.random.normal(loc=0., scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['N'])))

        else:
            self.V.weight.data = g*self.P['gain'][0] * tt(
                np.random.normal(loc=0, scale=1 / np.sqrt(self.P['N']), size=(self.P['R'], self.P['N'])))
            self.U.weight.data = g*self.P['gain'][1] * tt(
                np.random.normal(loc=0, scale=1 / np.sqrt(self.P['N']), size=(self.P['N'], self.P['R'])))


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
        Noise_rec = math.sqrt(2 * self.P['tau']) * g_rec * torch.randn(Input.shape).to(device)
        for i in range(Input.shape[1]):
            if self.P['R'] == -1:
                hidden = (1 - alpha) * hidden + alpha * (
                        self.J(self.act_func(hidden)) + Noise_rec[:, i] + Input[:, i])
            else:
                hidden = (1 - alpha) * hidden + alpha * (
                        self.U(self.V(self.act_func(hidden))) + Noise_rec[:, i] + Input[:, i])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def Traj(self,hidden_0,T,Input=None,device='cpu',**kwargs):
        '''
        Trajectory of hidden state under given Input with time steps T
        :param hidden_0: (Batch_Size,N)
        :param T: int
        :param Input: (N,)
        :param device:
        :param kwargs:
        :return:
        '''
        hidden=hidden_0.to(device)
        hidden_t=hidden.unsqueeze(dim=-2)
        if Input==None:
            Input=torch.zeros(hidden.shape[-1])
        Input=Input.to(device)
        alpha = self.P['dt'] / self.P['tau']
        for i in range(T):
            if self.P['R'] == -1:
                hidden = (1 - alpha) * hidden + alpha * (
                        self.J(self.act_func(hidden)) + Input)
            else:
                hidden = (1 - alpha) * hidden + alpha * (
                        self.U(self.V(self.act_func(hidden))) + Input)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        return hidden_t

    def gradient(self, hidden, Input=None):
        '''
        computation of gradient under given hidden state and Input series
        :param hidden: given hidden state,should be of shape((shape_),N_Encoder)
        :param Input: None if not given, should be of shape((shape_),N_Encoder)
        :return:
        grad of Encoder under Input ((shape_),N_Encoder)
        '''
        if Input == None:
            Input = torch.zeros_like(hidden)
        # gradient is computed with nondimensionalized t_=t/tau
        # alpha = Encoder.P['dt'] / Encoder.P['tau']
        if self.P['R'] == -1:
            gradient = -hidden + self.J(self.act_func(hidden)) + Input
        else:
            gradient = -hidden + self.U(self.V(self.act_func(hidden))) + Input
        return gradient





class VanillaRNN(nn.Module):
    def __init__(self, P):
        super(VanillaRNN, self).__init__()
        self.P = P
        if 'act_func' in self.P:
            self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.Q = nn.Linear(in_features=P['N'], out_features=P['N'], bias=False)
        self.hidden_0 = torch.zeros(P['N'], dtype=torch.float)

    def reinit(self,g=1.):
        self.Q.weight.data = g*tt(
            np.random.normal(loc=0, scale=1 / math.sqrt(self.P['N']), size=(self.P['N'], self.P['N'])))

    def forward(self, Input=None, n_steps=None, hidden_0=None, device='cpu'):
        '''
        :param Input: external input (Batch_Size,T,N) if given
        :param n_steps: trial length if given
        :param hidden_0: initial hidden state if given
        :param device:
        :return:
        '''

        if Input == None:
            hidden = hidden_0.to(device)
            Batch_Input = torch.zeros(hidden.shape[0], n_steps, hidden.shape[1]).to(device)
        else:
            if hidden_0 == None:
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
