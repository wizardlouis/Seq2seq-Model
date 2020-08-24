import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from network import Direction
from rw import *


# Batch_Seq(Batch_Size,seq_length)
def plot_trajectory(Net, Batch_Seq, row, column=5):
    Direct = np.array(Direction)
    h_0 = Net.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _ = Net(h_0, Batch_Seq)
    geometry = geometry.detach().numpy()
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(len(Batch_Seq[0])):
        c[-Net.t_retrieve + Net.t_cue + i * (Net.t_ron + Net.t_roff):-Net.t_retrieve + Net.t_cue + i * (
                Net.t_ron + Net.t_roff) + Net.t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(Batch_Seq)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:, 0], Direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        plt.text(-2, -2, str(Batch_Seq[k]))


def plot_trajectory2(Net, Batch_Seq, row, column=5):
    Direct = np.array(Direction)
    h_0 = Net.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _ = Net(h_0, Batch_Seq)
    seq_length = len(Batch_Seq[0])
    geometry = geometry.detach().numpy()
    geometry = geometry[-Net.t_retrieve - Net.t_delay: -Net.t_retrieve + Net.t_cue + seq_length * Net.t_ron + (
            seq_length - 1) * Net.t_roff]
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(seq_length):
        c[Net.t_delay + Net.t_cue + i * (Net.t_ron + Net.t_roff):Net.t_delay + Net.t_cue + i * (
                Net.t_ron + Net.t_roff) + Net.t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(Batch_Seq)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(Direct[:, 0], Direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        plt.text(-2, -2, str(Batch_Seq[k]), fontsize=20)


def plot_fps(all_fps, tol, components, model, batch_seq, pcaset='fps', is_3d=False):
    delay_points = get_delay_points(model, batch_seq, 30, 15, 5, 10, 5, 50, 150,
                                    5, 1)
    if pcaset == 'fps':
        pca = PCA(n_components=3).fit(all_fps[tol]['fps'])
    if pcaset == 'delay':
        pca = PCA(n_components=3).fit(delay_points.detach().numpy())
    # get all fixed/slow points
    hstars = np.reshape(all_fps[tol]['fps'], (-1, 128))
    # get all delay stage points
    delay_pca = pca.transform(delay_points.detach().numpy())
    print(delay_pca.shape)
    # set a color map for losses
    cm = plt.cm.get_cmap('rainbow')
    losses = np.array(all_fps[tol]['losses'])
    norm = plt.Normalize(losses.min(), losses.max())
    c = norm(losses)
    hstar_pca = pca.transform(hstars)
    if is_3d:
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(hstar_pca[:, 0],
                   hstar_pca[:, 1],
                   hstar_pca[:, 2], c=c, cmap=cm, s=2)
        #         ax.scatter(delay_pca[:, 0], delay_pca[:, 1], delay_pca[:, 2], s=5)
        for i in range(1):
            ax.plot(delay_pca[250 * i:250 * i + 250, 0],
                    delay_pca[250 * i:250 * i + 250, 1],
                    delay_pca[250 * i:250 * i + 250, 2], '^', ms=2)

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        a = 2
        ax.set_xlim(-a, a)
        ax.set_ylim(-a, a)
        ax.set_zlim(-a, a)
        ax.set_title('fixed points in 3d PCA space from ' + pcaset)
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        comp1 = components[0]
        comp2 = components[1]
        ax.scatter(hstar_pca[:, comp1], hstar_pca[:, comp2], c=c, cmap=cm, s=5)
        #         ax.scatter(delay_pca[:, comp1], delay_pca[:, comp2], s=5)
        for i in range(1):
            ax.plot(delay_pca[250 * i:250 * i + 250, 0],
                    delay_pca[250 * i:250 * i + 250, 1], '^', ms=3)
        ax.set_xlabel('PC ' + str(comp1 + 1))
        ax.set_ylabel('PC ' + str(comp2 + 1))
        a = 5
        ax.set_xlim(-a, a)
        ax.set_ylim(-a, a)
        ax.set_title('fixed points in 2d PCA space from ' + pcaset)


def plot_pca_trajectory(all_fps, tol, n_neuro, components, model, batch_seq, row, column=2, pcaset='fps', is_3d=False):
    hidden, c = get_trajectories(model, batch_seq)
    delay_points = get_delay_points(model, batch_seq, 30, 15, 5, 10, 5, 50, 150,
                                    5, 1)
    if pcaset == 'fps':
        pca = PCA(n_components=3).fit(all_fps[tol]['fps'])
    if pcaset == 'delay':
        pca = PCA(n_components=3).fit(delay_points.detach().numpy())
    # get all fixed/slow points
    hstars = np.reshape(all_fps[tol]['fps'], (-1, n_neuro))
    print(hstars.shape)
    hstar_pca = pca.transform(hstars)
    # plot points from delay stage in PC space
    delay_pca = pca.transform(delay_points.detach().numpy())
    cm = plt.cm.get_cmap('autumn')
    if is_3d:
        fig = plt.figure(figsize=(16, 6 * row))
        for k in range(hidden.shape[1]):
            ax = fig.add_subplot(row, column, k + 1, projection='3d')
            ax.scatter(hstar_pca[:, 0], hstar_pca[:, 1], hstar_pca[:, 2])

            #             for i in range(6):
            #                 ax.scatter(delay_pca[300 * i:300 * i + 300, 0],
            #                            delay_pca[300 * i:300 * i + 300, 1],
            #                            delay_pca[300 * i:300 * i + 300, 2],
            #                            s=5)

            traj = hidden[:, k, :]
            tr_pca = pca.transform(traj)
            ax.scatter(tr_pca[:, 0],
                       tr_pca[:, 1],
                       tr_pca[:, 2], c=c, cmap=cm, vmin=0, vmax=1)
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            a = 5
            ax.set_xlim(-a, a)
            ax.set_ylim(-a, a)
            #             ax.set_zlim(-a, a)
            ax.set_title(str(batch_seq[k]))
    else:
        fig = plt.figure(figsize=(16, 6 * row))
        comp1 = components[0]
        comp2 = components[1]
        for k in range(hidden.shape[1]):
            ax = fig.add_subplot(row, column, k + 1)
            # fixed points
            ax.scatter(hstar_pca[:, comp1], hstar_pca[:, comp2])

            #             delay stage
            #             for i in range(6):
            #                 ax.scatter(delay_pca[300 * i:300 * i + 300, comp1],
            #                            delay_pca[300 * i:300 * i + 300, comp2],
            #                            s=5)

            # real tracjectories
            traj = hidden[:, k, :]
            tr_pca = pca.transform(traj)
            ax.scatter(tr_pca[:, comp1],
                       tr_pca[:, comp2], c=c, cmap=cm, vmin=0, vmax=1)
            ax.set_xlabel('PC ' + str(comp1 + 1))
            ax.set_ylabel('PC ' + str(comp2 + 1))
            a = 10
            ax.set_xlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_title(str(batch_seq[k]))
