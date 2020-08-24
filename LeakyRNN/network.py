import torch
import torch.nn as nn
import torch.nn.functional as F
from rw import *
from gene_seq import *

Direction = [
    [math.cos(0), math.sin(0)],
    [math.cos(math.pi / 3), math.sin(math.pi / 3)],
    [math.cos(math.pi * 2 / 3), math.sin(math.pi * 2 / 3)],
    [math.cos(math.pi), math.sin(math.pi)],
    [math.cos(math.pi * 4 / 3), math.sin(math.pi * 4 / 3)],
    [math.cos(math.pi * 5 / 3), math.sin(math.pi * 5 / 3)]
]


class myRNN(nn.Module):
    def __init__(self, N_Neuron, W, g_D, act_func, Vec_embedding, Vectorrep,
                 t_rest, t_on, t_off, t_ron, t_roff, t_delay, t_retrieve, t_cue,
                 decay=0.1, training=True, train_geometry=False):
        super(myRNN, self).__init__()
        self.N = N_Neuron
        self.W = W
        self.d = decay
        self.act_func = act_func
        self.Vec_embedding = Vec_embedding
        self.Vectorrep = Vectorrep
        self.g_D = g_D
        self.t_rest = t_rest
        self.t_on = t_on
        self.t_off = t_off
        self.t_ron = t_ron
        self.t_roff = t_roff
        self.t_delay = t_delay
        self.t_retrieve = t_retrieve
        self.t_cue = t_cue
        self.training = training
        self.use_scale_loss = False
        self.train_geometry = train_geometry

        self.Geometry = nn.Linear(in_features=self.N, out_features=2, bias=True)
        self.Readout = nn.Linear(in_features=2, out_features=6, bias=False)
        if not self.train_geometry:
            D = nn.Parameter(torch.tensor(Direction), requires_grad=False)
            self.Readout.weight = D
            self.Readout.weight.requires_grad = False

    def init_rand_hidden(self):
        # initialize hidden state with(time,N)
        return torch.randn(self.N, requires_grad=True)

    def reset_hidden(self):
        return torch.zeros(self.N, requires_grad=True)

    def set_additional_delay(self, min, max):
        self.ad_delay_min = min
        self.ad_delay_max = max
        self.delay_fixed = False

    def forward(self, hidden_0, Batch_Seq):
        # transform Seq to Input
        if self.delay_fixed:
            Input = Batch_Seq2Input(Batch_Seq, self.Vectorrep, self.Vec_embedding, self.g_D,
                                    self.t_rest, self.t_on, self.t_off, self.t_delay, self.t_retrieve, self.t_cue)
        else:
            Input = Batch_Seq2Input_vary_delay(Batch_Seq, self.Vectorrep, self.Vec_embedding, self.g_D,
                                               self.t_rest, self.t_on, self.t_off, self.t_delay, self.t_retrieve,
                                               self.t_cue, self.ad_delay_min, self.ad_delay_max)
        # initialize hidden state and hidden state time course
        hidden_t = torch.tensor([])
        # input frames in Input one by one
        Batch_Size, T = Input.shape[0], Input.shape[1]
        # hidden (Batch_Size,N)
        hidden = hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)
        for frame in range(T):
            hidden = (1 - self.d) * hidden + self.d * (
                    torch.einsum('ab,cb->ca', [self.W, self.act_func(hidden)]) + Input[:, frame, :])
            hidden_t = torch.cat((hidden_t, hidden.unsqueeze(dim=0)), dim=0)
        geometry = self.Geometry(self.act_func(hidden_t))
        readout = self.Readout(geometry)
        return hidden_t, geometry, readout

    def getDelay(self, trial, tolerate=0):
        return trial[-self.t_retrieve - self.t_delay + tolerate:-self.t_retrieve]

    # not necessary readout, it can be geometry
    def getTarget(self, trial, n_item):
        return torch.cat([trial[-self.t_retrieve + self.t_cue + i * (self.t_ron + self.t_roff):
                                -self.t_retrieve + self.t_cue + self.t_ron + i * (self.t_ron + self.t_roff)] for i in
                          range(n_item)], dim=0)

    # input geometry should be in critical period cause no additional slicing operation in the function
    # scale determine the raduis trained, if 1:normalize,if 0:fixed at zero point
    def scale_loss(self, geometry, scale, temp):
        if not self.use_scale_loss:
            return 0
        else:
            L = len(geometry)
            norm = geometry[:, 0] ** 2 + geometry[:, 1] ** 2
            abs_norm = temp * sum(abs(norm - scale)) / L
            return abs_norm

    # go forward a single step with given input and hidden
    def get_one_step_fun(self):

        def fun(hidden, input):
            hidden_t = (1 - self.d) * hidden + self.d * (self.W @ self.act_func(hidden) + input)
            geometry_t = self.Geometry(self.act_func(hidden_t))
            readout_t = self.Readout(geometry_t)
            return hidden_t, geometry_t, readout_t

        return fun

    # def regularization_loss(self,geometry):
    #     MSE=sum([torch.abs(g[0]**2+g[1]**2-1) for g in geometry])
    #     return MSE


class lowrank_RNN(nn.Module):
    def __init__(self, N_Neuron, N_rank, left_Vector, right_Vector, g_C, g_D, act_func, Vec_embedding, Vectorrep,
                 t_rest, t_on, t_off, t_delay, t_retrieve, t_cue,
                 Vector_train=True, unscaled_noise_M=None, noise_Train=False,
                 decay=0.1):
        super(lowrank_RNN, self).__init__()
        # delay period dixed
        self.delay_fixed = True
        # number of neurons and ranks in the network
        self.N = N_Neuron
        self.r = N_rank
        # set decay parameter/activation functions/seq_embedding
        # sequence embedding contained 3dimensions:x,y and cue
        self.d = decay
        self.act_func = act_func
        self.Vec_embedding = Vec_embedding;
        self.Vectorrep = Vectorrep;
        self.g_D = g_D
        # set noise with train&evaluation mode
        self.g_C = g_C
        self.noise_Train = noise_Train
        self.Geometry = nn.Linear(in_features=self.N, out_features=2, bias=True)
        self.Readout = nn.Linear(in_features=2, out_features=6, bias=False)
        # setting timescale of sequence
        self.t_rest = t_rest;
        self.t_on = t_on;
        self.t_off = t_off;
        self.t_delay = t_delay;
        self.t_retrieve = t_retrieve;
        self.t_cue = t_cue

        self.unscaled_noise_M = unscaled_noise_M
        if unscaled_noise_M.shape == torch.Size([self.N, self.N]):
            self.noise_M = self.g_C * unscaled_noise_M
            self.noise_M.requires_grad = self.noise_Train
        else:
            print('Noise shape not fit')
        # set left and right vector with train&evaluation mode
        self.Vector_train = Vector_train
        if left_Vector.shape == right_Vector.shape == torch.Size([self.r, self.N]):
            self.l_Vector = left_Vector
            self.l_Vector.requires_grad = self.Vector_train
            self.r_Vector = right_Vector
            self.r_Vector.requires_grad = self.Vector_train
        else:
            print('Vector size not fit')

    def set_additional_delay(self, min, max):
        self.ad_delay_min = min
        self.ad_delay_max = max
        self.delay_fixed = False

    def init_rand_hidden(self):
        # initialize hidden state with(time,N)
        return torch.randn(self.N, requires_grad=True)

    def reset_hidden(self):
        return torch.zeros(self.N, requires_grad=True)

    # #input shape is (time,N)
    # #return ouput time course (time,N), all time course were stored as act_func(hidden_state)
    # #geometry (time,2) read_out (time,6) based on activities of networks
    # def forward(self,hidden_0,Seq):
    #     #transform Seq to Input
    #     if self.delay_fixed:
    #         Input=Seq2Input(Seq,self.Vectorrep,self.Vec_embedding,self.g_D,
    #                         self.t_rest,self.t_on,self.t_off,self.t_delay,self.t_retrieve,self.t_cue)
    #     else:
    #         Input=Seq2Input_vary_delay(Seq,self.Vectorrep,self.Vec_embedding,self.g_D,
    #                         self.t_rest,self.t_on,self.t_off,self.t_delay,self.t_retrieve,self.t_cue,self.ad_delay_min,self.ad_delay_max)
    #
    #     #initialize hidden state and hidden state time course
    #     hidden_t=torch.tensor([])
    #     #get network recurrent connection
    #     W = self.r_Vector.transpose(0, 1) @ self.l_Vector/self.N + self.noise_M
    #     #input frames in Input one by one
    #     hidden=hidden_0
    #     for frame_input in Input:
    #         hidden=(1-self.d)*hidden+self.d*(W@self.act_func(hidden)+frame_input)
    #         hidden_t=torch.cat((hidden_t,self.act_func(hidden).unsqueeze(dim=0)),dim=0)
    #     geometry=self.Geometry(self.act_func(hidden_t))
    #     readout=self.Readout(geometry)
    #     return hidden_t,geometry,readout

    # return hidden_t(T, Batch_Size, N),geometry(T,Batch_Size,2),readout(T,Batch_Size,6)
    def forward(self, hidden_0, Batch_Seq):
        # transform Seq to Input
        # Input size(Batch_Size,T,N)
        if self.delay_fixed:
            Input = Batch_Seq2Input(Batch_Seq, self.Vectorrep, self.Vec_embedding, self.g_D,
                                    self.t_rest, self.t_on, self.t_off, self.t_delay, self.t_retrieve, self.t_cue)
        else:
            Input = Batch_Seq2Input_vary_delay(Batch_Seq, self.Vectorrep, self.Vec_embedding, self.g_D,
                                               self.t_rest, self.t_on, self.t_off, self.t_delay, self.t_retrieve,
                                               self.t_cue, self.ad_delay_min, self.ad_delay_max)

        # initialize hidden state and hidden state time course
        hidden_t = torch.tensor([])
        # get network recurrent connection
        W = self.r_Vector.transpose(0, 1) @ self.l_Vector / self.N + self.noise_M
        # input frames in Input one by one
        Batch_Size, T = len(Batch_Seq), len(Batch_Seq[0])
        # hidden (Batch_Size,N)
        hidden = hidden_0.unsqueeze(dim=0).repeat(Batch_Size, 1)
        for frame in range(T):
            hidden = (1 - self.d) * hidden + self.d * (
                    torch.einsum('ab,cb->ca', [W, self.act_func(hidden)]) + Input[:, frame, :])
            hidden_t = torch.cat((hidden_t, self.act_func(hidden).unsqueeze(dim=0)), dim=0)
        geometry = self.Geometry(self.act_func(hidden_t))
        readout = self.Readout(geometry)
        return hidden_t, geometry, readout

    # return periods of readout with (t_on*n_item,6)
    def getTarget(self, readout, n_item):
        return torch.cat([readout[-self.t_retrieve + self.t_cue + i * (self.t_on + self.t_off):
                                  -self.t_retrieve + self.t_cue + self.t_on + i * (self.t_on + self.t_off)] for i in
                          range(n_item)], dim=0)

    def regularization_loss(self, Seq, readout):
        return torch.tensor(0., dtype=torch.float)

    def geteigenvalue(self):
        pass

    # save network parameters in a dictionary
    def get_param(self):
        dic = {'N_Neuron': self.N, 'N_rank': self.r, 'left_Vector': self.l_Vector, 'right_Vector': self.r_Vector,
               'g_C': self.g_C, 'g_D': self.g_D, 'act_func': self.act_func, 'Vec_embedding': self.Vec_embedding,
               'Vectorrep': self.Vectorrep,
               't_rest': self.t_rest, 't_on': self.t_on, 't_off': self.t_off, 't_delay': self.t_delay,
               't_retrieve': self.t_retrieve, 't_cue': self.t_cue,
               'Vector_train': self.Vector_train, 'unscaled_noise_M': self.unscaled_noise_M,
               'noise_Train': self.noise_Train, 'decay': self.d}
        return dic

    # build lowrank network based on a dictionary of parameters
    @staticmethod
    def loadparam(paramdic):
        P = paramdic
        return lowrank_RNN(P['N_Neuron'], P['N_rank'], P['left_Vector'], P['right_Vector'],
                           P['g_C'], P['g_D'], P['act_func'], P['Vec_embedding'], P['Vectorrep'],
                           P['t_rest'], P['t_on'], P['t_off'], P['t_delay'], P['t_retrieve'], P['t_cue'],
                           P['Vector_train'], P['unscaled_noise_M'], P['noise_Train'], P['decay'])
