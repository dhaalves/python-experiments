#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
import six.moves.urllib as urllib
import random
import pandas as pd

DB_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale'
DB_SIZE = 768
TARGET_PATH = os.path.dirname(os.path.realpath(__file__))


def get_dataset():
    print('downloading diabetes ...')
    rows = list(urllib.request.urlopen(DB_URL))
    selected = random.sample(rows, DB_SIZE)
    with open(os.path.join(TARGET_PATH, 'diabetes.txt'), 'wb') as f:
        for row in selected:
            f.write(row)
    print('diabetes downloaded successfully !\n')


def run(trn_ds, tst_ds, lbr, model, qs, quota, step=1):
    model.train(trn_ds)
    acc = model.score(tst_ds)

    for i in range(quota):
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)

        print(X.shape)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)
        if (i+1) % step == 0 or (i+1) == quota:
            model.train(trn_ds)
            acc = np.append(acc, model.score(tst_ds))


    return acc


def split_train_test(dataset_filepath, test_size, n_labeled):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    df_train = pd.read_csv(dir_path + '/data/train_leaves1_inceptionv3.csv')
    df_test = pd.read_csv(dir_path + '/data/test_leaves1_inceptionv3.csv')

    x_train = df_train.drop('2048', axis=1).as_matrix()
    y_train = df_train['2048'].as_matrix()
    x_test = df_test.drop('2048', axis=1).as_matrix()
    y_test = df_test['2048'].as_matrix()

    # X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    conca = np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)])
    trn_ds = Dataset(x_train, conca)
    tst_ds = Dataset(x_test, y_test)
    fully_labeled_trn_ds = Dataset(x_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.2  # the percentage of samples in the dataset that will be
    n_labeled = 100  # number of samples that are initially labeled

    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled  # number of samples to query
    step = 40
    print(quota)

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    acc_uncertain = run(trn_ds, tst_ds, lbr, model, qs, quota, step)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    acc_randdom = run(trn_ds2, tst_ds, lbr, model, qs2, quota, step)

    query_num = np.arange(1, int(quota/step) + 3)
    print(query_num.shape)
    print(acc_randdom.shape)
    plt.plot(query_num, acc_uncertain, 'k', mec='b', marker='x', label='Uncertain Sampling', lw=1)
    plt.plot(query_num, acc_randdom, 'k', mec='g', marker='^', mfc='g', label='Random Sampling', lw=1)
    plt.xlabel('Iteration')
    plt.ylabel('Acurracy')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    seed = 1  # np.random.random_integers(1, 100, size=1)[0]
    np.random.seed(seed)

    # get_dataset()
    main()
