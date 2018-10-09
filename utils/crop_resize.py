import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod

cur_dir = os.getcwd()
print("resizing images")
print("current directory:",cur_dir)

def modify_image(image):
    resized = tf.image.resize_images(image, 180, 180, 1)
    resized.set_shape([180,180,3])
    flipped_images = tf.image.flip_up_down(resized)
    return flipped_images

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return image

def inputs():
    filenames = ['img1.jpg', 'img2.jpg' ]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=2)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return reshaped_image

filename_queue = tf.train.string_input_producer(['204.jpg']) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
print(value)
print(key)
my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
back = tf.image.encode_jpeg(my_img)
print(back)
print(value == back)
init_op = tf.initialize_all_variables()
tfrecords_filename = '/home/daniel/ext/experimentos/datasets/folhas/imagens/leaves_test.tfrecord'
filename_queue = tf.train.string_input_producer([tfrecords_filename])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    })
print(features)
image = tf.decode_raw(features['image/encoded'], tf.uint8)
with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # for i in range(1): #length of your filename list
    # image = my_img.eval() #here is your image Tensor :)

    # print(image.shape)

    resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image, 800, 800)
    print(resize_image_with_crop_or_pad)
    #
    # print(my_img.eval().shape)
    resized_image = tf.image.resize_images(
        resize_image_with_crop_or_pad.eval(),
        [300, 300],
        method=ResizeMethod.AREA,
        align_corners=False
    )
    print(resized_image)

    tf.image.convert_image_dtype(resized_image, tf.uint8    )
    # Image.fromarray(np.asarray(resized_image.eval())).show()
    imarray = np.asarray(resized_image.eval())
    img = Image.fromarray(np.asarray(image))
    # img.save('slice56.png')
    plt.imshow(imarray)
    plt.show()

    coord.request_stop()
    coord.join(threads)
