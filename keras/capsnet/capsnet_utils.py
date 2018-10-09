import numpy as np
from matplotlib import pyplot as plt
import csv
import math

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def split_train_test(path, perc_split, recreate=False):
    import os
    import shutil
    import numpy as np
    train_root_path = path + '_train'
    test_root_path = path + '_test'
    new_split = os.path.exists(test_root_path) or os.path.exists(train_root_path) or recreate
    if new_split:
        if os.path.exists(train_root_path):
            shutil.rmtree(train_root_path)
        if os.path.exists(test_root_path):
            shutil.rmtree(test_root_path)
        shutil.copytree(path, train_root_path)
        classes = os.listdir(train_root_path)
        for folder in classes:
            train_path = os.path.join(train_root_path, folder)
            test_path = os.path.join(test_root_path, folder)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            files = os.listdir(os.path.join(train_root_path, folder))
            for f in files:
                if np.random.rand(1) < perc_split:
                    shutil.move(train_path + '/' + f, test_path + '/' + f)
                    # plot_log('result/log.csv')
    return train_root_path, test_root_path

if __name__=="__main__":
    split_train_test('/home/daniel/Desktop/datasets/leaves1bkp', 0.2)
