import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
# import sklearn
# from sklearn import cross_validation
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
import csv

DATASET_ROOT = 'C:\\Users\\dnago\\Desktop\\folhas'

def create_graph():
    with gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def import_to_csv(images_dir, images, train_test_folder, inception_layer='softmax/logits:0'):
    create_graph()
    with tf.Session() as sess:
        tensor = sess.graph.get_tensor_by_name(inception_layer)
        with open(train_test_folder + '-' + inception_layer.replace('/',':') + '.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(np.arange(2049))
            for classe in sorted(images.keys()):
                for file in sorted(images[classe]):
                    image_path = os.path.join(images_dir, train_test_folder, classe, file)
                    image_data = gfile.FastGFile(image_path, 'rb').read()
                    predictions = sess.run(tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    row = np.append(np.squeeze(predictions), [str(classe)])
                    csvwriter.writerow(np.asarray(row))


def _get_filenames_and_classes(dataset_dir, train_test_folders=True):
    def get_filenames_and_classes(sub_dir):
        dict = {}
        data_dir = os.path.join(dataset_dir, sub_dir)
        for filename in os.listdir(data_dir):
            path = os.path.join(data_dir, filename)
            dict[filename] = []
            for f in os.listdir(path):
                dict[filename].append(f)
        return dict

    if train_test_folders:
        train = get_filenames_and_classes('train')
        test = get_filenames_and_classes('test')
        return train, test
    return get_filenames_and_classes('')


def method_name(dataset_dir):
    dict = {}
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        dict[filename] = []
        for f in os.listdir(path):
            dict[filename].append(f)





if __name__ == '__main__':
    model_dir = '/tmp/imagenet'
    images_dir = '/mnt/sdb1/datasets/leaves/leaves1_splitted'  # Acer-campestre/'
    train, test = _get_filenames_and_classes(images_dir)
    # train = _get_filenames_and_classes(images_dir, False)
    import_to_csv(images_dir, train, 'train', 'pool_3/_reshape:0')
    import_to_csv(images_dir, test, 'test', 'pool_3/_reshape:0')
