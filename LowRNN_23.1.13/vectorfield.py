# -*- codeing = utf-8 -*-
# @time:2021/12/14 下午3:17
# Author:Xuewen Shen
# @File:vectorfield.py
# @Software:PyCharm

import matplotlib.pyplot as plt
import numpy as np
import math
from rw import  *

#Hyperparameters of file loading pathway and workspace
device='cuda:0'
seqpath='SD_//n2SD2//1_1'
model_file='n2_r2_fixed'
model_id=99

#define modelpath and savepath
filepath='Seq2seq//'+model_file+'//'

#load sequence file
Seq=load_obj(seqpath)

#load model file
modelpath='{}//model_{}.pth'.format(filepath,str(model_id))
savepath='Seq2seq_analysis//{}_id={}//'.format(model_file,str(model_id))
model=torch.load(modelpath,map_location=device)
P=model.P

#load Trajectory
Traj=load_obj(savepath+'//activity')

#get encoder right vector coordinates
U_J,S_J,V_J=torch.svd(model.Encoder.U.weight.data@model.Encoder.V.weight.data)
n_clip=2
U_J_clip=U_J[:,:2]

#Input & figname list setup
Batch_Input=[0.*model.In.weight.data.squeeze(dim=1),1.*model.In.weight.data.squeeze(dim=1),-1.*model.In.weight.data.squeeze(dim=1)]
fignamelist=['grad_0','grad_+1','grad_-1']

#sample starting point setup
sample=[
    [0,0],[1,1],[1,-1],[-1,1],[-1,-1],
    [0,100],[100,100],[100,0],[100,-100],[0,-100],[-100,-100],[-100,0],[-100,100],
    [10,25],[-10,-25],[10,10],[-10,-10],[25,85],[-25,-85],[10,65],[-10,-65],
    [0,50],[-50,0],[20,50],[50,20],[-20,-50],[-50,-20]
]

sample1=[
    [0,100],[100,100],[100,0],[100,-100],[0,-100],[-100,-100],[-100,0],[-100,100],
]

T=200
h_space=[-120.,120.,1.]
v_space=[-120.,120.,1.]
#plot vector field
def plot_gradient_(h_space,v_space,model,U_J_clip,Input,sample,T,device='cpu'):
    '''
    :param h_space: horizontal space grid
    :param v_space: vertical space grid
    :param model: Seq2Seq model
    :param U_J_clip: torch.tensor.shape(N,n_clip) coordiantes of plots
    :param Input: background input
    :param sample: sample starting point
    :param T: sample evolving time steps
    :return:
    '''

    #get gradient distribution
    xv, yv = np.meshgrid(np.arange(*h_space), np.arange(*v_space))
    coordinate = torch.cat((tt(xv).unsqueeze(dim=-1), tt(yv).unsqueeze(dim=-1)), dim=-1).to(device)
    U_J_clip.to(device)
    hidden = coordinate @ U_J_clip.T
    grad = model.Encoder.gradient(hidden, Input=Input)
    gradient_U = grad @ U_J_clip
    # imshow plot up-down left-right, but gradient_U matrix down-up left-right, should be filpped through 0th-dim
    #plot gradient background
    m=plt.imshow(np.flip(tn(gradient_U.norm(dim=2)), 0), extent=h_space[:2] + v_space[:2])
    plt.colorbar(m)

    # sample trajectories Evolution
    sample = torch.tensor(sample, dtype=torch.float).to(device)
    hidden_s = sample @ U_J_clip.T
    hidden_t = model.Encoder.Traj(hidden_s, T, Input=Input, device=device)
    hidden_U = hidden_t @ U_J_clip
    # arrow shape and position setup
    scaling = 1
    devide = 12
    #plot sample trajectory
    for i in range(len(hidden_t)):
        plt.plot(tn(hidden_U)[i, :, 0], tn(hidden_U)[i, :, 1], color='w')
        plt.arrow(tn(hidden_U)[i, int(T / devide), 0], tn(hidden_U)[i, int(T / devide), 1],
                  (tn(hidden_U)[i, int(T / devide) + 1, 0] - tn(hidden_U)[i, int(T / devide), 0]) * scaling,
                  (tn(hidden_U)[i, int(T / devide) + 1, 1] - tn(hidden_U)[i, int(T / devide), 1]) * scaling, fc='w',
                  ec='w',
                  head_width=5, head_length=5)

def plot_gradient(ax,h_space,v_space,model,U_J_clip,Input,sample,T,device='cpu'):
    '''
    :param ax: axes
    :param h_space: horizontal space grid
    :param v_space: vertical space grid
    :param model: Seq2Seq model
    :param U_J_clip: torch.tensor.shape(N,n_clip) coordiantes of plots
    :param Input: background input
    :param sample: sample starting point
    :param T: sample evolving time steps
    :return:
    '''

    #get gradient distribution
    xv, yv = np.meshgrid(np.arange(*h_space), np.arange(*v_space))
    coordinate = torch.cat((tt(xv).unsqueeze(dim=-1), tt(yv).unsqueeze(dim=-1)), dim=-1).to(device)
    U_J_clip.to(device)
    hidden = coordinate @ U_J_clip.T
    grad = model.Encoder.gradient(hidden, Input=Input)
    gradient_U = grad @ U_J_clip
    # imshow plot up-down left-right, but gradient_U matrix down-up left-right, should be filpped through 0th-dim
    #plot gradient background
    m=ax.imshow(np.flip(tn(gradient_U.norm(dim=2)), 0), extent=h_space[:2] + v_space[:2])
    plt.colorbar(m)

    # sample trajectories Evolution
    sample = torch.tensor(sample, dtype=torch.float).to(device)
    hidden_s = sample @ U_J_clip.T
    hidden_t = model.Encoder.Traj(hidden_s, T, Input=Input, device=device)
    hidden_U = hidden_t @ U_J_clip
    # arrow shape and position setup
    scaling = 1
    devide = 12
    #plot sample trajectory
    for i in range(len(hidden_t)):
        ax.plot(tn(hidden_U)[i, :, 0], tn(hidden_U)[i, :, 1], color='w')
        ax.arrow(tn(hidden_U)[i, int(T / devide), 0], tn(hidden_U)[i, int(T / devide), 1],
                  (tn(hidden_U)[i, int(T / devide) + 1, 0] - tn(hidden_U)[i, int(T / devide), 0]) * scaling,
                  (tn(hidden_U)[i, int(T / devide) + 1, 1] - tn(hidden_U)[i, int(T / devide), 1]) * scaling, fc='w',
                  ec='w',
                  head_width=5, head_length=5)

#gradient+sample
#grad_0
plt.figure(figsize=(10,10))
plot_gradient_(h_space,v_space,model,U_J_clip,Batch_Input[0],sample,T,device=device)
plt.xlabel(r'$\kappa_1$', fontsize=25)
plt.ylabel(r'$\kappa_2$', fontsize=25)
plt.savefig(savepath+'//{}.png'.format(fignamelist[0]))
plt.close()
#grad_+1
plt.figure(figsize=(10,10))
plot_gradient_(h_space,v_space,model,U_J_clip,Batch_Input[1],sample1,T,device=device)
plt.xlabel(r'$\kappa_1$', fontsize=25)
plt.ylabel(r'$\kappa_2$', fontsize=25)
plt.savefig(savepath+'//{}.png'.format(fignamelist[1]))
plt.close()
#grad_-1
plt.figure(figsize=(10,10))
plot_gradient_(h_space,v_space,model,U_J_clip,Batch_Input[2],sample1,T,device=device)
plt.xlabel(r'$\kappa_1$', fontsize=25)
plt.ylabel(r'$\kappa_2$', fontsize=25)
plt.savefig(savepath+'//{}.png'.format(fignamelist[2]))
plt.close()





# # check dynamic scale
# #grad_0+Traj
# fig=plt.figure(figsize=(10,10))
# ax=fig.add_subplot(111)
# plot_gradient(ax,h_space,v_space,model,U_J_clip,Batch_Input[0],sample,T,device=device)
# ax.set_xlabel(r'$\kappa_1$', fontsize=25)
# ax.set_ylabel(r'$\kappa_2$', fontsize=25)
#
# Traj_pU = tn(Traj['l2']['he'].to(device) @ U_J_clip)
# for i in range(len(Traj_pU)):
#     ax.plot(Traj_pU[i, :, 0], Traj_pU[i, :, 1], label=str(Seq['totalset']['l2'][i]), color='cyan')
#
# fig.savefig(savepath + '//{}.png'.format('grad_0_Traj2'))
# plt.close()



# plt.text(-110, 45, str([1]), fontsize=25)
# plt.text(80, -45, str([2]), fontsize=25)


