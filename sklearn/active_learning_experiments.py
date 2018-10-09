import pandas as pd
from prompt_toolkit.key_binding.bindings.named_commands import yank_last_arg
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import NearestNeighbors
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def get_clusters_dic(labels):
    clusters_dic = dict()
    y_unique = np.unique(labels)
    for c in y_unique:
        clusters_dic[c] = np.argwhere(labels == c).flatten()
        np.random.shuffle(clusters_dic[c])
    return clusters_dic


def get_boundaries_idx(x, labels):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
    nbrs.fit(x)
    dist, ind = nbrs.kneighbors(x)
    boundaries = np.empty((0, 1), int)
    for knn_pair in ind:
        if labels[knn_pair[0]] != labels[knn_pair[1]]:
            boundaries = np.append(boundaries, knn_pair)
    return np.unique(boundaries)


def get_cluster_centers_idx(x, cluster_centers):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nbrs.fit(x)
    _, cluster_centers_idx = nbrs.kneighbors(cluster_centers)
    return cluster_centers_idx.flatten()


def active_learning(x, y, samples_per_iteration=27, iterations=None):
    log = dict()
    x_it, y_it, log[0] = compute_first_iteration()

    n_samples = len(y)
    max_iterations = int(n_samples / samples_per_iteration) + 1 if iterations is None else iterations

    # x_it = np.empty((0, len(x[1])), float)
    # y_it = np.empty((0, 1), str)
    init_len = len(y_it)

    clusters_ids = np.array(list(clusters_dic.keys()))

    it = 1
    while it <= max_iterations:
        step = it * samples_per_iteration
        while len(y_it) - init_len < step and len(y_it) - init_len != n_samples:
            for c_id in (c_id for c_id in clusters_ids if len(clusters_dic[c_id]) != 0):
                val_x = np.take(x, clusters_dic[c_id][:1], axis=0)
                val_y = np.take(y, clusters_dic[c_id][:1], axis=0)
                x_it = np.concatenate((x_it, val_x))  # np.append(x_it, val_x)
                y_it = np.append(y_it, val_y)
                clusters_dic[c_id] = np.delete(clusters_dic[c_id], [0])
                if len(y_it) - init_len == step or len(y_it) - init_len == n_samples:
                    break
            # y_unique, y_unique_count = np.unique(y_it, return_counts=True)
        y_pred = clf.fit(x_it, y_it).predict(x_test)
        log[it] = accuracy_score(y_test, y_pred)
        # cross_val_score(clf, x_it, y_it, scoring='accuracy', cv=5)
        print("Iteration %d, N. Samples %d, Accuracy %.2f" % (it, len(y_it), log[it]))  # , np.std(log[it])))
        it += 1
    return log


def save_files(x, y, df):
    np.savetxt(dir_path + '/data/pca_leaves1_inceptionv3.csv', x, fmt=''.join(['%.18e,'] * 2048 + ['%s']))
    # np.savetxt(dir_path + '/data/x_pca_leaves1_inceptionv3.csv', x_pca, delimiter=',')
    # np.savetxt(dir_path + '/data/y_pca_leaves1_inceptionv3.csv', y, fmt='%s')


def plot():
    plt.plot(list(log_uc.keys()), list(log_uc.values()), 'k', lw=1, mec='b', marker='x', label='Kmeans_UC')
    plt.plot(list(log_rbe.keys()), list(log_rbe.values()), 'k', lw=1, mec='g', mfc='g', marker='^', label='Kmeans_RBE')
    plt.title('Dataset LEAVES-1; Features Size ')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def compute_first_iteration():  # x_train, y_train, x_test, y_test, cluster_centers, clf):
    centers_idx = get_cluster_centers_idx(x_train, cluster_centers)
    x_centers = np.take(x_train, centers_idx, axis=0)
    y_centers = np.take(y_train, centers_idx)
    y_pred = clf.fit(x_centers, y_centers).predict(x_test)
    return x_centers, y_centers, accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    seed = 1  # np.random.random_integers(1, 100, size=1)[0]
    np.random.seed(seed)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_train = pd.read_csv(dir_path + '/data/train_leaves1_inceptionv3.csv')
    df_test = pd.read_csv(dir_path + '/data/test_leaves1_inceptionv3.csv')
    df = pd.read_csv(dir_path + '/data/leaves1_inceptionv3.csv')
    x = df.drop('2048', axis=1).as_matrix()
    y = df['2048'].as_matrix()

    # dtype = np.array([float for i in range(2048)])
    # dtype = np.append(dtype, str)
    # t = np.loadtxt(dir_path + '/data/leaves1_inceptionv3.csv', delimiter=',', dtype=dtype)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    x_train = df_train.drop('2048', axis=1).as_matrix()
    y_train = df_train['2048'].as_matrix()
    x_test = df_test.drop('2048', axis=1).as_matrix()
    y_test = df_test['2048'].as_matrix()

    n_components = 100
    n_clusters = len(np.unique(y_train)) * 2
    n_samples = len(y_train)
    n_samples_per_iteration = 27  # int(n_samples * 0.025)
    n_iterations = int(n_samples / n_samples_per_iteration) + 1

    pca = PCA(n_components=n_components, random_state=seed)
    x_pca = pca.fit_transform(np.concatenate((x_train, x_test)))
    x_train = x_pca[:len(x_train)]
    x_test = x_pca[len(x_train):]

    clf = SVC(random_state=seed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x_train)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    boundaries_idx = get_boundaries_idx(x_train, labels)
    x_rbe = np.take(x_train, boundaries_idx, axis=0)
    y_rbe = np.take(y_train, boundaries_idx)
    labels_rbe = np.take(labels, boundaries_idx)

    clusters_dic = get_clusters_dic(labels_rbe)
    log_rbe = active_learning(x_rbe, y_rbe, n_samples_per_iteration)

    clusters_dic = get_clusters_dic(labels)
    log_uc = active_learning(x_train, y_train, n_samples_per_iteration, iterations=24)

    plot()
