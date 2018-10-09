import math
import random
import os

from os import listdir, path, walk, makedirs
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from os.path import join, basename, exists, splitext


def tif_to_jpg(input_path, output_path, quality=100):
    for root, dirs, files in walk(input_path):
        save_dir = join(output_path, basename(root))
        if not exists(save_dir) and root != input_path:
            makedirs(save_dir)
        for file in files:
            if file.endswith('.tif'):
                image = Image.open(join(root, file))
                image.convert(mode="RGB")
                image.save(splitext(join(save_dir, file))[0] + ".jpg", "JPEG", quality=quality)



def save_class_distribution(folder, save_to=None):
    f = basename(folder) if save_to is None else save_to
    open_mode = 'a' if exists(f + '_dist.csv') else 'w'
    with open(join(f + '_dist.csv'), open_mode) as f:
        zeror = 0
        zeror_sum = 0
        for value in os.listdir(folder):
            count = len([name for name in os.listdir(join(folder, value))])
            zeror_sum += count
            if count > zeror:
                zeror = count
            # if save_to is None:
            #     f.write("%s, %d\n" % (value, count))
            # else:
            f.write("%s, %s, %d\n" % (basename(folder), value, count))
        # if save_to is None:
        #     f.write('%s, %.3f\n' % ('Baseline Acc', zeror/zeror_sum))
        # else:
        f.write('%s, %s, %.3f\n' % (basename(folder), 'Baseline Acc', zeror/zeror_sum))



def augment_images(img_path, ceiling_sample_to=None):
    imgs_count = sum([len(files) for r, d, files in walk(img_path)])
    print('Number of Images: %s' % imgs_count)
    class_count = len(listdir(img_path))
    print('Number of Classes: %s' % class_count)
    if ceiling_sample_to is None:
        ceiling_sample_to = math.ceil(imgs_count / class_count)

    print('Ceiling samples to: %s' % ceiling_sample_to)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    for root, dirs, files in walk(img_path):
        if img_path != root:
            if len(files) < ceiling_sample_to and files:
                total = ceiling_sample_to - len(files)

                for i in range(total):

                    img = load_img(join(root, files[i % len(files)]))  # this is a PIL image
                    x = img_to_array(img)  # this is a Numpy array with shape (3, x, x)
                    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, x, x)

                    for _ in datagen.flow(x, batch_size=1,
                                          save_to_dir=root, save_prefix='aug', save_format='jpeg'):
                        break
        else:
            print('Class Names %s' % dirs)


def train_test_split(img_filenames, perc_spli=0.8):
    random.seed(0)
    random.shuffle(img_filenames)
    train_size = int(round(len(img_filenames) * perc_spli))
    return img_filenames[:train_size], img_filenames[train_size:]

input_path = '/home/daniel/Desktop/soybean2-tif'
output_path = '/mnt/sdb1/dataset/mammoset/exp5-2 _aug'

if __name__ == '__main__':
    # tif_to_jpg(input_path, output_path)
    augment_images(output_path)
