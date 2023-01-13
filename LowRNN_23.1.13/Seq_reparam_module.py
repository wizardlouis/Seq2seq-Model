# -*- codeing = utf-8 -*-
# @time:2022/8/20 下午10:13
# Author:Xuewen Shen
# @File:Seq_reparam_module.py
# @Software:PyCharm

import torch
import torch.nn as nn
import math
import numpy as np
import random
import time
import os
from network import reparameterlizedRNN_sample
from rw import tn, tt

Cross_Entropy_Loss = nn.CrossEntropyLoss(reduction='mean')

Hill_threshold = 0.1


def logcosh(x):
    return x.cosh().log()


def L1soft(x, theta=Hill_threshold):
    return logcosh(x ** 3 / (x ** 2 + theta ** 2)).sum()


def MSE0(x):
    L = nn.MSELoss(reduction='sum')
    return L(x, torch.zeros_like(x))


# Fixed_Loss = MSE0
Fixed_Loss = L1soft

dt = 20
tau = 100
t_rest = 200
t_stim_min = 160
t_stim_max = 240
t_interval = 400
t_delay_min = 200
t_delay_max = 2000
# readout from last_time point-out_duration
out_duration = 200
# fixation from last time point-fixation_duration
fix_duration = 200

# Durations for different events
time_information = dict(dt=dt, tau=tau, t_rest=t_rest, t_stim_min=t_stim_min, t_stim_max=t_stim_max,
                        t_interval=t_interval, t_delay_min=t_delay_min, t_delay_max=t_delay_max)

T_rest = math.floor(t_rest / dt)
T_stim_min = math.floor(t_stim_min / dt)
T_stim_max = math.floor(t_stim_max / dt)
T_interval = math.floor(t_interval / dt)
T_delay_min = math.floor(t_delay_min / dt)
T_delay_max = math.floor(t_delay_max / dt)


def get_max_t(L):
    return t_rest + t_stim_max * L + t_interval * (L - 1) + t_delay_max


# generate sequence dataset

# generate all possible sequence with length n in dataset items,output list (seq index,item ranks)
def Gen_Seq(items, n, repeat=False):
    if repeat:
        return Gen_Seq_REPEAT(items, n)
    else:
        return Gen_Seq_NOREPEAT(items, n)


def Gen_Seq_NOREPEAT(items, n):
    if n == 1:
        return [[item] for item in items]
    else:
        fseq = []
        for item in items:
            restitems = list(filter(lambda x: x != item, items))
            fseq.extend([[item] + seq for seq in Gen_Seq_NOREPEAT(restitems, n - 1)])
        return fseq


def Gen_Seq_REPEAT(items, n):
    if n == 1:
        return [[item] for item in items]
    else:
        fseq = []
        for item in items:
            fseq.extend([[item] + seq for seq in Gen_Seq_REPEAT(items, n - 1)])
        return fseq


# generate mixed length Sequence set
def Gen_Seq_Mix_Length(items, lengths, repeat=False):
    seq = []
    for length in lengths:
        seq += Gen_Seq(items, length, repeat=repeat)
    return seq


# randomly generate Dataset from a Sequence set
def Data_generator(Seq_set, Batch_Size=20):
    Data = [random.choice(Seq_set) for i in range(Batch_Size)]
    return Data


# generate Input from a Seq_set which is not necessary of the same length
def Batch2Input_generator(Data, N_out, Embedding, max_t=2440, g_in=0., device='cpu', fix_all=True):
    '''
    :param Data:
    :param N_out: number of readout channels
    :param Embedding:
    :param max_t:
    :param g_in:
    :param device:
    :param fix_all:
    :return:
    '''
    max_T = math.floor(max_t / dt)
    Batch_Size = len(Data)
    n_dim = len(Embedding[0])
    Input = g_in * math.sqrt(2 * tau / dt) * torch.randn(Batch_Size, max_T, n_dim)
    Target = Data
    Mask_pos = []
    Mask_fix = torch.zeros(Batch_Size, max_T, N_out * n_dim)
    T_fix = math.floor(fix_duration / dt)
    for idx in range(Batch_Size):
        positions = []
        duration_delay = random.choice(range(T_delay_min, T_delay_max + 1))
        seq = Data[idx]
        l = len(seq)
        pointer = T_rest
        for r in range(l):
            duration_stim = random.choice(range(T_stim_min, T_stim_max + 1))
            Input[idx, pointer:pointer + duration_stim] = torch.tensor(Embedding[seq[r]], device=device)
            pointer += duration_stim
            if r != l - 1:
                pointer += T_interval
            else:
                pointer += duration_delay
        # readout point to the last time point in delay
        for r in range(l):
            positions.append([pointer, r])
        Mask_pos.append(positions)
        for r in range(N_out):
            if r >= l:
                if fix_all:
                    Mask_fix[idx, :pointer, r * n_dim:(r + 1) * n_dim] = 1.
                else:
                    Mask_fix[idx, pointer - T_fix:pointer, r * n_dim:(r + 1) * n_dim] = 1.
    return Input.to(device), Target, Mask_pos, Mask_fix.to(device)


def Batch2Input_generator_art(Data, N_out, Embedding, max_t=2440, g_in=0., device='cpu', fix_all=True,
                              duration_stim=math.floor((T_stim_min + T_stim_max) / 2),
                              duration_delay=math.floor((T_delay_min + T_delay_max) / 2)):
    '''
    :param Data:
    :param N_out: number of readout channels
    :param Embedding:
    :param out_duration:
    :param max_T:
    :param g_in:
    :param device:
    :return:
    '''
    max_T = math.floor(max_t / dt)
    Batch_Size = len(Data)
    n_dim = len(Embedding[0])
    Input = g_in * math.sqrt(2 * tau / dt) * torch.randn(Batch_Size, max_T, n_dim)
    Target = Data
    Mask_pos = []
    Mask_fix = torch.zeros(Batch_Size, max_T, N_out * n_dim)
    T_fix = math.floor(fix_duration / dt)
    for idx in range(Batch_Size):
        positions = []
        seq = Data[idx]
        l = len(seq)
        pointer = T_rest
        for r in range(l):
            Input[idx, pointer:pointer + duration_stim] = torch.tensor(Embedding[seq[r]], device=device)
            pointer += duration_stim
            if r != l - 1:
                pointer += T_interval
            else:
                pointer += duration_delay
        for r in range(l):
            positions.append([pointer, r])
        Mask_pos.append(positions)
        for r in range(N_out):
            if r >= l:
                if fix_all:
                    Mask_fix[idx, :pointer, r * n_dim:(r + 1) * n_dim] = 1.
                else:
                    Mask_fix[idx, pointer - T_fix:pointer, r * n_dim:(r + 1) * n_dim] = 1.
    return Input.to(device), Target, Mask_pos, Mask_fix.to(device)


# train network with given input, Target, Msk_pos, Mask_fix information
def train_reparameter_seq(model, Embedding, optimizer, scheduler, Input, Target, Mask_pos, Mask_fix, g_rec=0.,
                          loss_weight=[1., 0., 1e-5, 1e-2], reg_type='L1',
                          ortho_type=dict(I=False, U=False, V=False, W=False, IU=False, IW=False, d_I=1., d_R=0.9,
                                          d_O=0.), clip_gradient=True,
                          device='cpu'):
    Batch_Size = Input.shape[0]
    n_dim = len(Embedding[0])
    T_out = math.floor(out_duration / dt)
    optimizer.zero_grad()
    Output = model(Input, g_rec=g_rec, dt=dt, device=device)[:, 1:]
    # readout cross entropy loss
    Loss_p = []
    for trial in range(Batch_Size):
        for r in range(len(Target[trial])):
            pos = Mask_pos[trial][r]
            prediction = Output[trial, pos[0] - T_out:pos[0], pos[1] * n_dim:(pos[1] + 1) * n_dim] @ torch.tensor(
                Embedding,
                device=device).T
            target = torch.tensor([Target[trial][r]], device=device).repeat_interleave(prediction.shape[0], dim=0)
            loss_p = Cross_Entropy_Loss(prediction, target)
            Loss_p.append(loss_p)
    Loss_p = sum(Loss_p) / len(Loss_p)
    # fixation loss
    if Mask_fix.sum() == 0.:
        Loss_f = torch.zeros(1, device=device)
    else:
        Loss_f = Fixed_Loss(Output * Mask_fix) / Mask_fix.sum()
    Loss_reg = model.reg_loss(type=reg_type)
    Loss_ortho = model.ortho_loss(**ortho_type, device=device)
    Loss = Loss_p * loss_weight[0] + Loss_f * loss_weight[1] + Loss_reg * loss_weight[2] + Loss_ortho * loss_weight[3]
    Loss.backward()
    if clip_gradient:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if scheduler is not None:
        scheduler.step(Loss)
    Loss_name = ['Total', 'Positional', 'Fixation', 'Regularization', 'Orthogonal']
    return Loss_name, np.array([loss.data.item() for loss in [Loss, Loss_p, Loss_f, Loss_reg, Loss_ortho]])


converge_loss = 0.002


# training process for reparameterlized model
def reporter_train_reparamter_seq(model, trainset, Embedding, optimizer, scheduler, Batch_Size=20, max_t=2440,
                                  g_in=0., g_rec=0., loss_weight=[1., 1e-3, 1e-3, 1e-2], reg_type='L1',
                                  ortho_type=dict(I=False, U=False, V=False, W=False, IU=False, IW=False, d_I=1.,
                                                  d_R=0.9, d_O=0.), fix_all=True,
                                  clip_gradient=True, device='cpu', savepath=None, n_epochs=2000, freq_report=10,
                                  freq_save=20):
    n_dim = len(Embedding[0])
    n_converge = 0
    start0 = time.time()
    save_count = 0

    f = open(savepath + '//report.txt', 'a')
    f.write('Report of simulation:\n')
    f.close()

    # save initial model
    torch.save(model, savepath + '//model_0.pth')
    save_count += 1

    for epoch in range(n_epochs):
        start = time.time()
        model.reset_noise_loading(device=device)

        Data = Data_generator(trainset, Batch_Size=Batch_Size)
        Input, Target, Mask_pos, Mask_fix = Batch2Input_generator(Data,
                                                                  int(model.N_O / n_dim),
                                                                  Embedding, max_t=max_t, g_in=g_in,
                                                                  device=device, fix_all=fix_all)
        loss_name, Epoch_loss = train_reparameter_seq(model, Embedding, optimizer, scheduler, Input, Target, Mask_pos,
                                                      Mask_fix, g_rec=g_rec, loss_weight=loss_weight, reg_type=reg_type,
                                                      ortho_type=ortho_type, clip_gradient=clip_gradient, device=device)

        if epoch % freq_report == freq_report - 1:
            end = time.time()
            f = open(savepath + '//report.txt', 'a')
            f.write(f'\nEpoch {epoch + 1}:\n')
            for idx in range(len(loss_name)):
                f.write(f'{loss_name[idx]} Loss = {str(Epoch_loss[idx])}\n')

            f.write(
                f'This Epoch takes:{str(end - start)} seconds.\nThe whole process takes:{str(end - start0)} seconds\n')
            f.close()
        # save model
        if epoch % freq_save == freq_save - 1:
            torch.save(model, savepath + '//model_' + str(save_count) + '.pth')
            save_count += 1
        if not os.path.exists(savepath + '//loss.npz'):
            np.savez(savepath + '//loss.npz', name=np.array(['Loss', 'Loss_p', 'Loss_f', 'Loss_r']),
                     data=np.array([Epoch_loss]))
        else:
            Lossfile = np.load(savepath + '//loss.npz')
            np.savez(savepath + '//loss.npz', name=Lossfile['name'],
                     data=np.concatenate((Lossfile['data'], np.array([Epoch_loss])), axis=0))
        if Epoch_loss[0] > converge_loss:
            n_converge = 0
        else:
            n_converge += 1
        if n_converge == 3:
            break

    end0 = time.time()
    f = open(savepath + '//report.txt', 'a')
    f.write('\nThe whole Training Process finished in {} seconds!!!'.format(str(end0 - start0)))
    return True


# training process for reparamterlized model with separate blocks of different lengths of sequences
def reporter_train_reparamter_seq_separate(model, mixed_trainset, Embedding, optimizer, scheduler,
                                           mixed_Batch_Size=[8, 16], mixed_max_t=[2440, 3080],
                                           g_in=0., g_rec=0., loss_weight=[1., 1e-3, 1e-3, 1e-2], reg_type='L1',
                                           ortho_type=dict(I=False, U=False, V=False, W=False, IU=False, IW=False,
                                                           d_I=1., d_R=0.9, d_O=0.), fix_all=True,
                                           clip_gradient=True, device='cpu', savepath=None, n_epochs=2000,
                                           freq_report=10, freq_save=20):
    n_dim = len(Embedding[0])
    n_converge = 0
    start0 = time.time()
    save_count = 0

    f = open(savepath + '//report.txt', 'a')
    f.write('Report of simulation:\n')
    f.close()

    # save initial model
    torch.save(model, savepath + '//model_0.pth')
    save_count += 1

    for epoch in range(n_epochs):
        start = time.time()
        model.reset_noise_loading(device=device)
        random.shuffle(mixed_trainset)
        Epoch_loss = 0.
        loss_name = []
        for trainset in mixed_trainset:
            L = len(trainset[0])
            Batch_Size = mixed_Batch_Size[L - 1]
            Data = Data_generator(trainset, Batch_Size=Batch_Size)
            Input, Target, Mask_pos, Mask_fix = Batch2Input_generator(Data,
                                                                      int(model.N_O / n_dim),
                                                                      Embedding, max_t=mixed_max_t[L - 1], g_in=g_in,
                                                                      device=device, fix_all=fix_all)
            loss_name, Batch_loss = train_reparameter_seq(model, Embedding, optimizer, scheduler, Input, Target,
                                                          Mask_pos,
                                                          Mask_fix, g_rec=g_rec, loss_weight=loss_weight,
                                                          reg_type=reg_type,
                                                          ortho_type=ortho_type, clip_gradient=clip_gradient,
                                                          device=device)
            Epoch_loss += Batch_loss

        if epoch % freq_report == freq_report - 1:
            end = time.time()
            f = open(savepath + '//report.txt', 'a')
            f.write(f'\nEpoch {epoch + 1}:\n')
            for idx in range(len(loss_name)):
                f.write(f'{loss_name[idx]} Loss = {str(Epoch_loss[idx])}\n')

            f.write(
                f'This Epoch takes:{str(end - start)} seconds.\nThe whole process takes:{str(end - start0)} seconds\n')
            f.close()
        # save model
        if epoch % freq_save == freq_save - 1:
            torch.save(model, savepath + '//model_' + str(save_count) + '.pth')
            save_count += 1
        if not os.path.exists(savepath + '//loss.npz'):
            np.savez(savepath + '//loss.npz', name=np.array(['Loss', 'Loss_p', 'Loss_f', 'Loss_r']),
                     data=np.array([Epoch_loss]))
        else:
            Lossfile = np.load(savepath + '//loss.npz')
            np.savez(savepath + '//loss.npz', name=Lossfile['name'],
                     data=np.concatenate((Lossfile['data'], np.array([Epoch_loss])), axis=0))
        if Epoch_loss[0] > converge_loss:
            n_converge = 0
        else:
            n_converge += 1
        if n_converge == 3:
            break

    end0 = time.time()
    f = open(savepath + '//report.txt', 'a')
    f.write('\nThe whole Training Process finished in {} seconds!!!'.format(str(end0 - start0)))
    return True


hidden_trajectory_colorset = [f'C{i}' for i in range(6)]


def plot_vectorfield(ax, model, aligned_axes_idices, hidden, samples, h_lim, v_lim, Input=None, full_gradient=True,
                     hidden_colors=hidden_trajectory_colorset, device='cpu', arrow_kwargs=dict(), traj_wargs=dict()):
    # plotting mesh gradients
    plot_mesh_gradient(ax, model, aligned_axes_idices, h_lim, v_lim, Input=Input, full_gradient=full_gradient,
                       device=device)
    # plotting arrow trajectories from samples
    plot_adjacent_arrow(ax, model, aligned_axes_idices, samples, Input=Input, device=device, **arrow_kwargs)
    # plotting trajectories
    plot_trajectory_2d(ax, model, aligned_axes_idices, hidden, colorset=hidden_colors, device=device, **traj_wargs)


def plot_mesh_gradient(ax, model: reparameterlizedRNN_sample, aligned_axes, h_lim, v_lim, Input=None, dpi=100,
                       full_gradient=True, device='cpu'):
    # get meshgrid
    dh = (h_lim[1] - h_lim[0]) / dpi
    h_space = np.arange(h_lim[0], h_lim[1] + dh, dh)
    dv = (v_lim[1] - v_lim[0]) / dpi
    v_space = np.arange(v_lim[0], v_lim[1] + dv, dv)
    xv, yv = np.meshgrid(h_space, v_space)

    # tranform coordinates to hidden states
    coordinate = torch.cat([tt(xv, device=device).unsqueeze(dim=-1), tt(yv, device=device).unsqueeze(dim=-1)], dim=-1)
    kappa = torch.zeros(*coordinate.shape[:-1], model.N_R, device=device)
    kappa[:, :, aligned_axes] = coordinate
    hidden = model.kappa_align_U(kappa, device=device)

    # get gradient norms
    if Input is not None:
        Input = Input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(*xv.shape[:2], 1).to(device)
    gradient_aligned = model.hidden_align_U(model.gradient(hidden, Input=Input, device=device), device=device)
    if full_gradient:
        gradient_norm = gradient_aligned.norm(dim=-1)
    else:
        gradient_norm = gradient_aligned[:, :, aligned_axes].norm(dim=-1)

    # plot gradient based on full_gradient or gradient within subspaces
    ax.imshow(np.flip(tn(gradient_norm), 0), extent=h_lim + v_lim)


def plot_adjacent_arrow(ax, model: reparameterlizedRNN_sample, aligned_axes, samples, Input=None, t_simulation=4000,
                        t_arrow=340, device='cpu', scaling=1., **kwargs):
    # transform sample starting points to hidden states
    samples = torch.tensor(samples, dtype=torch.float).to(device)
    kappa = torch.zeros(*samples.shape[:-1], model.N_R, device=device)
    kappa[:, aligned_axes] = samples
    hidden = model.kappa_align_U(kappa, device=device)

    # get trajectories within subspace from starting points
    n_steps = math.floor(t_simulation / dt)
    if Input is None:
        Input = torch.zeros(samples.shape[0], n_steps, model.N_I, device=device)
    else:
        Input = Input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(samples.shape[0], n_steps, 1).to(device)

    Eh, _ = model(Input, hidden=hidden, g_rec=0., get_Encoder=True, device=device)
    kappa = tn(model.hidden_align_U(Eh, device=device)[:, :, aligned_axes])

    # plot trajectories and arrows
    if type(t_arrow) == int:
        t_arrow = [t_arrow for i in range(kappa.shape[0])]
    n_arrow_step = [math.floor(t / dt) for t in t_arrow]

    # shape arrows
    if 'head_width' in kwargs:
        head_width = kwargs['head_width']
    else:
        head_width = 5
    if 'head_length' in kwargs:
        head_length = kwargs['head_length']
    else:
        head_length = 5
    for trial in range(kappa.shape[0]):
        ax.plot(kappa[trial, :, 0], kappa[trial, :, 1], color='w')
        ax.arrow(kappa[trial, n_arrow_step[trial], 0], kappa[trial, n_arrow_step[trial], 1],
                 (kappa[trial, n_arrow_step[trial] + 1, 0] - kappa[trial, n_arrow_step[trial], 0]) * scaling,
                 (kappa[trial, n_arrow_step[trial] + 1, 1] - kappa[trial, n_arrow_step[trial], 1]) * scaling,
                 fc='w', ec='w', head_width=head_width, head_length=head_length)


def plot_trajectory_2d(ax, model: reparameterlizedRNN_sample, aligned_axes, hidden, colorset=hidden_trajectory_colorset,
                       device='cpu', **kwargs):
    aligned_hidden = model.hidden_align_U(hidden, device=device).detach().cpu().numpy()[:, :, aligned_axes]
    for trial in range(aligned_hidden.shape[0]):
        ax.plot(aligned_hidden[trial, :, 0], aligned_hidden[trial, :, 1], color=colorset[trial], **kwargs)
