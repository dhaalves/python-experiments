import os
import sys
import glob
import tensorflow as tf
from numpy import unicode

shuffle_data = True  # shuffle the addresses before saving
dataset_path = '/home/daniel/Desktop/datasets/leaves_simple/train/*/*.jpg'


# read addresses and labels from the 'train' folder
train_addrs = glob.glob(dataset_path)
# train_labels = [basename(dirname(addr)) for addr in train_addrs]
# if shuffle_data:
#     c = list(zip(addrs, labels))
#     shuffle(c)
#     addrs, labels = zip(*c)
#
# # Divide the hata into 60% train, 20% validation, and 20% test
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(labels)):]


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, class_name):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': bytes_feature(class_name.encode('utf8')),
    }))

tfrecord_path = '/home/daniel/Desktop/datasets/leaves_simple/train/leaves.tfrecord'

tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_path)

for addr in train_addrs:
    image_data = tf.gfile.FastGFile(addr, 'rb').read()
    class_name = os.path.basename(os.path.dirname(addr))
    example = image_to_tfexample(image_data, class_name)
    print(example)
    tfrecord_writer.write(example.SerializeToString())

tfrecord_writer.close()


with tf.Session() as sess:
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image'], tf.uint8)

    # Cast label data into int32
    label = tf.cast(features['label'], tf.string)

    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    print(images)
    print(labels)