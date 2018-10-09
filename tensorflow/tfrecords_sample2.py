import glob

from PIL import Image
import numpy as np
import os
import skimage.io as io


import tensorflow as tf

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400

tfrecords_filename = '/home/daniel/Desktop/datasets/leaves_simple/train/leaves.tfrecord'

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


def write_and_encode(dataset_dir='/home/daniel/Desktop/datasets/leaves_simple/train/*/*.jpg'):


    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    with tf.Session() as sess:
        for img_path in glob.glob(dataset_dir):

            img = np.array(Image.open(img_path))
            img2 = tf.gfile.FastGFile(img_path, 'rb').read()

            img_t = tf.image.decode_jpeg(img2, channels=3)

            # The reason to store image sizes was demonstrated
            # in the previous example -- we have to know sizes
            # of images to later read raw serialized string,
            # convert to 1d array and convert to respective
            # shape that image used to have.
            height = img_t.eval().shape[0]
            width = img_t.eval().shape[1]

            label = os.path.basename(os.path.dirname(img_path)).encode('utf8')
            filename = os.path.basename(img_path).encode('utf8')

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': int64_feature(height),
                'width': int64_feature(width),
                'image': bytes_feature(img2),
                'label': bytes_feature(label),
                'filename': bytes_feature(filename)}))

            writer.write(example.SerializeToString())

        writer.close()

def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'filename': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.cast(features['label'], tf.string)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])

    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=IMAGE_HEIGHT,
                                                           target_width=IMAGE_WIDTH)

    images = tf.train.shuffle_batch([resized_image],
                                                  batch_size=16,
                                                  capacity=30,
                                                  num_threads=2,
                                                  min_after_dequeue=10)
    return images

if __name__ == '__main__':
    write_and_encode()

    reconstructed_images = []


    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)


    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])

        img_string = (example.features.feature['image']
                      .bytes_list
                      .value[0])

        label = (example.features.feature['label']
                 .bytes_list
                 .value[0])

        filename = (example.features.feature['filename']
                 .bytes_list
                 .value[0])

        img = tf.decode_raw(img_string, tf.uint8)

        img = tf.reshape(img, [height, width, 3])
        with tf.Session() as sess:
            # print(tf.rank(img).eval())
            print(img_string)
            # print(img.shape)
            # print(img.get_shape())
            # print(height, width)
            # # img = tf.reshape(img, [height, width, 3])
            # print(img.eval())
            # print(img.shape)
            # print(img.get_shape())
        # print(filename)
        # print(img)

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))

        # annotation_1d = np.fromstring(label, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # reconstructed_annotation = annotation_1d.reshape((height, width))

        # reconstructed_images.append((reconstructed_img))

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)

    # Even when reading in multiple threads, share the filename
    # queue.
    image = read_and_decode(filename_queue)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # sess.run(image)

        # # Let's read off 3 batches just for example
        # for i in range(3):
        #
        #     i = sess.run([image])
        #     print(img[0, :, :, :].shape)
        #
        #     print('current batch')
        #
        #     # We selected the batch size of two
        #     # So we should get two image pairs in each batch
        #     # Let's make sure it is random
        #
        #     io.imshow(img[0, :, :, :])
        #     io.show()
        #
        #     io.imshow(anno[0, :, :, 0])
        #     io.show()
        #
        #     io.imshow(img[1, :, :, :])
        #     io.show()
        #
        #     io.imshow(anno[1, :, :, 0])
        #     io.show()


        coord.request_stop()
        coord.join(threads)

