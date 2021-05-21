import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# reverse=False: In_Channel=3,Out_Channel=2
# reverse=True: In_Channel=5,Out_Channel=2
# Dt_on/Dt_delay: flexibility of target/delay steps
# default t_off:30-50 delay:40-80
default_param = dict(train_io=False, train_hidden_0=False, act_func='Tanh',
                     in_Channel=3, out_Channel=2,
                     dt=20, tau=100, g_in=0.01, g_rec=0.15,
                     N_Neuron=1000, t_on=20, Dt_on=10, t_item=60,
                     t_delay=40, Dt_delay=40, t_retrieve=40,
                     t_cue=10, t_ron=10, t_fix=10, t_final=40)

#Reference scaling of network with input strength u_0
N_0=128

class fullRNN(nn.Module):
    # initialization of recurrent
    # parameters: P;
    # train/evaluation: train
    # actication_function:act_func
    # Variables build-up by P:
    # Input layer: In; Recurrent layer: W; Output layer: Out
    # initial hidden state: hidden_0;
    def __init__(self, P):
        super(fullRNN, self).__init__()
        self.P = P
        self.N = self.P['N_Neuron']
        self.training = True
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.In = nn.Linear(in_features=self.P['in_Channel'], out_features=self.N, bias=True)
        self.W = nn.Linear(in_features=self.N, out_features=self.N, bias=False)
        self.Out = nn.Linear(in_features=self.N, out_features=self.P['out_Channel'], bias=True)
        G = torch.randn(P['N_Neuron'], P['N_Neuron'])
        u, _, _ = torch.svd(G)
        # Weight of input is orthogonal and scaled with sqrt(N/N_0) with N_0=128
        self.In.weight = nn.Parameter(math.sqrt(self.N/N_0)*u[:self.P['in_Channel']].transpose(1, 0), requires_grad=True)
        self.Out.weight = nn.Parameter(u[self.P['in_Channel']:self.P['out_Channel'] + self.P['in_Channel']],
                                       requires_grad=True)
        if not self.P['train_io']:
            self.In.weight.requires_grad = False
            self.Out.weight.requires_grad = False
        # if hidden_0==True, we should initialization hidden_0 as parameter requires_grad=True
        # else hidden_0=default initial hidden state [0]
        if self.P['train_hidden_0']:
            self.hidden_0 = nn.Parameter(torch.randn(self.N), requires_grad=True)
        else:
            self.hidden_0 = nn.Parameter(torch.zeros(self.N), requires_grad=False)

    def setW(self, W):
        self.W.weight = nn.Parameter(W, requires_grad=True)

    def Train(self):
        self.training = True

    def Eval(self):
        self.training = False

    # hidden state wrt t: hidden_t;
    # output wrt t: out
    def forward(self, Batch_Input, hidden_0=None, device='cpu'):
        # Batch_Input:(Batch_Size,T,channels:x,y,cue)
        Batch_Size, T = Batch_Input.shape[0:2]
        # Intrinsic noise is off if evaluation is on
        Batch_Input = Batch_Input.to(device)
        if self.training:
            Noise_Input = math.sqrt(2 * self.P['tau']) * self.P['g_in'] * torch.randn(Batch_Input.shape).to(device)
            Input = self.In(Batch_Input + Noise_Input)
        else:
            Input = self.In(Batch_Input)
        # hidden in one step is of shape(Batch_Size,N),decide if using default hidden
        if hidden_0 == None:
            hidden = self.hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)
        else:
            hidden = hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)

        hidden_t = torch.tensor([]).to(device)
        for frame in range(T):
            if self.training:
                Noise_recurrent = math.sqrt(2 * self.P['tau']) * self.P['g_rec'] * torch.randn(
                    [Batch_Size, self.N]).to(device)
                hidden = (1 - self.P['dt'] / self.P['tau']) * hidden + self.P['dt'] / self.P['tau'] * (
                        self.W(self.act_func(hidden)) + Input[:, frame, :] + Noise_recurrent)
            else:
                hidden = (1 - self.P['dt'] / self.P['tau']) * hidden + self.P['dt'] / self.P['tau'] * (
                        self.W(self.act_func(hidden)) + Input[:, frame, :])
            # hidden_t=(Batch_Size,T,N)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = self.Out(self.act_func(hidden_t))
        return hidden_t, out


class lowRNN(nn.Module):
    def __init__(self, P):
        super(lowRNN, self).__init__()
        self.P = P
        self.N = self.P['N_Neuron']
        self.r = self.P['N_rank']
        self.training = True
        self.act_func = eval('torch.nn.' + self.P['act_func'] + '()')
        self.In = nn.Linear(in_features=self.P['in_Channel'], out_features=self.N, bias=False)
        self.V = nn.Linear(in_features=self.N, out_features=self.r, bias=False)
        self.U = nn.Linear(in_features=self.r, out_features=self.N, bias=False)
        self.Out = nn.Linear(in_features=self.N, out_features=self.P['out_Channel'], bias=False)
        G = torch.randn(P['N_Neuron'], P['N_Neuron'])
        u, _, _ = torch.svd(G)
        self.In.weight = nn.Parameter(math.sqrt(self.N/N_0)*u[:self.P['in_Channel']].transpose(1, 0), requires_grad=True)
        self.Out.weight = nn.Parameter(u[self.P['in_Channel']:self.P['out_Channel'] + self.P['in_Channel']],
                                       requires_grad=True)
        if not self.P['train_io']:
            self.In.weight.requires_grad = False
            self.Out.weight.requires_grad = False
        # if hidden_0==True, we should initialization hidden_0 as parameter requires_grad=True
        # else hidden_0=default initial hidden state [0]
        if self.P['train_hidden_0']:
            self.hidden_0 = nn.Parameter(torch.randn(self.N), requires_grad=True)
        else:
            self.hidden_0 = nn.Parameter(torch.zeros(self.N), requires_grad=False)

    def Train(self):
        self.training = True

    def Eval(self):
        self.training = False

    # hidden state wrt t: hidden_t;
    # output wrt t: out
    def forward(self, Batch_Input, hidden_0=None, device='cpu'):
        # Batch_Input:(Batch_Size,T,channels:x,y,cue)
        Batch_Size, T = Batch_Input.shape[0:2]
        # Intrinsic noise is off if evaluation is on
        Batch_Input = Batch_Input.to(device)
        if self.training:
            Noise_Input = math.sqrt(2 * self.P['tau']) * self.P['g_in'] * torch.randn(Batch_Input.shape).to(device)
            Input = self.In(Batch_Input + Noise_Input)
        else:
            Input = self.In(Batch_Input)
        # hidden in one step is of shape(Batch_Size,N),decide if using default hidden
        if hidden_0 == None:
            hidden = self.hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)
        else:
            hidden = hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)

        hidden_t = torch.tensor([]).to(device)
        for frame in range(T):
            if self.training:
                Noise_recurrent = math.sqrt(2 * self.P['tau']) * self.P['g_rec'] * torch.randn(
                    [Batch_Size, self.N]).to(device)
                hidden = (1 - self.P['dt'] / self.P['tau']) * hidden + self.P['dt'] / self.P['tau'] * (
                        self.U(self.V(self.act_func(hidden))) + Input[:, frame, :] + Noise_recurrent)
            else:
                hidden = (1 - self.P['dt'] / self.P['tau']) * hidden + self.P['dt'] / self.P['tau'] * (
                        self.U(self.V(self.act_func(hidden))) + Input[:, frame, :])
            # hidden_t=(Batch_Size,T,N)
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=1)), dim=1)
        out = self.Out(self.act_func(hidden_t))
        return hidden_t, out
