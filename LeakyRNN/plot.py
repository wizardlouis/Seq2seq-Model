import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from configs import *
from gene_seq import rank_batch2input_batch


def plot_trajectory(model, rank_batch, hps, row, column=5):
    direct = np.array(DIRECTION)
    t_rest = hps['t_rest']
    t_on = hps['t_on']
    t_off = hps['t_off']
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']
    rank_size = hps['rank_size']

    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)
    h_0 = model.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _, _ = model(h_0, input_batch)
    geometry = geometry.detach().numpy()
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(rank_size):
        c[t_delay + t_cue + i * (t_ron + t_roff):t_delay + t_cue + i * (t_ron + t_roff) + t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(rank_batch)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(direct[:, 0], direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        plt.text(-2, -2, str(rank_batch[k]))


def plot_trajectory2(model, rank_batch, hps, row, column=5):
    direct = np.array(DIRECTION)
    t_rest = hps['t_rest']
    t_on = hps['t_on']
    t_off = hps['t_off']
    t_ron = hps['t_ron']
    t_roff = hps['t_roff']
    t_retrieve = hps['t_retrieve']
    t_cue = hps['t_cue']
    rank_size = hps['rank_size']

    input_batch, _, t_delay = rank_batch2input_batch(rank_batch, hps)
    h_0 = model.reset_hidden()
    # geometry (T,Batch_Size,2)
    _, geometry, _, _ = model(h_0, input_batch)
    geometry = geometry.detach().numpy()
    geometry = geometry[-t_delay - t_retrieve: -t_retrieve + t_cue + rank_size * t_ron + (rank_size - 1) * t_roff]
    c = np.zeros(len(geometry))
    for i in range(len(c)):
        c[i] = i / len(c)
    for i in range(rank_size):
        c[t_delay + t_cue + i * (t_ron + t_roff):t_delay + t_cue + i * (
                t_ron + t_roff) + t_ron] = 0
    cm = plt.cm.get_cmap('autumn')
    for k in range(len(rank_batch)):
        plt.subplot(row, column, k + 1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(direct[:, 0], direct[:, 1], color='', edgecolor='g', s=200, marker='o')
        plt.scatter(geometry[:, k, 0], geometry[:, k, 1], vmin=0, vmax=1, c=c, cmap=cm)
        plt.text(-2, -2, str(rank_batch[k]), fontsize=20)


# TODO need to rewrite delay part
def plot_fps(all_fps, tol, components, net, rank_batch, hps, pcaset='fps', is_3d=False):
    delay_points = get_hidden_delay(net, rank_batch, 1, hps)
    if pcaset == 'fps':
        pca = PCA(n_components=3).fit(all_fps[tol]['fps'])
    if pcaset == 'delay':
        pca = PCA(n_components=3).fit(delay_points.detach().numpy())
    print(pca.explained_variance_ratio_)
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
    hidden, c = get_hidden(model, batch_seq)
    delay_points = get_hidden_delay(model, batch_seq, 30, 15, 5, 10, 5, 50, 150,
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


if __name__ == '__main__':
    path = 'test//1'
    seq, loss, model, data_hps = load_path(path)
    trainset = seq['train']
    testset = seq['test']

    data_hps = {
        't_rest': 30,
        't_on': 15,
        't_off': 5,
        't_ron': 10,
        't_roff': 5,
        't_cue': 5,
        't_delay': 50,
        't_add_delay_max': 50,
        't_retrieve': 150,
        'rank_size': 2,
        'delay_fixed': False,
        'add_noise': False,
        'g': 0.1
    }
    plt.figure(figsize=(50, 60))
    plot_trajectory(model, testset, data_hps, 6, 5)
    plt.show()
