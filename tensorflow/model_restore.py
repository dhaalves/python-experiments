
from sys import path as syspath
import tensorflow as tf
from tensorflow.python.framework import graph_util

from tensorflow.python.tools import inspect_checkpoint, freeze_graph
from tensorflow.core.framework import graph_pb2


models_slim_dir = '/home/daniel/Desktop/models/slim'
syspath.append(models_slim_dir)


from nets import mobilenet_v1

slim = tf.contrib.slim


def freeze_mobilenet(meta_file, img_size=224, factor=1.0, num_classes=1001):

    tf.reset_default_graph()

    inp = tf.placeholder(tf.float32,
                         shape=(None, img_size, img_size, 3),
                         name="input")

    is_training=False
    weight_decay = 0.0
    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        logits, _ = mobilenet_v1.mobilenet_v1(inp,
                                              num_classes=num_classes,
                                              is_training=is_training,
                                              depth_multiplier=factor)

    predictions = tf.contrib.layers.softmax(logits)
    output = tf.identity(predictions, name='output')

    ckpt_file = meta_file.replace('.meta', '')
    output_graph_fn = ckpt_file.replace('.ckpt', '.pb')
    output_node_names = "output"

    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, ckpt_file)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_fn, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("{} ops in the final graph.".format(len(output_graph_def.node)))


if __name__ == '__main__':

    with tf.Session() as sess:

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            print(node)
        freeze_graph.freeze_graph()
        # logits, end_points = mobilenet_v1.mobilenet_v1(tf.placeholder(dtype=tf.float32, shape=(1, 224, 224, 3), name='input'), 1001)
        # saver = tf.train.Saver()
        #
        # print()
        #
        # saver = tf.train.import_meta_graph("mobinet/mobilenet_v1_1.0_224.ckpt.meta")
        # save_path = saver.restore(sess, "mobinet/mobilenet_v1_1.0_224.ckpt")
        # inspect_checkpoint.print_tensors_in_checkpoint_file("mobinet/mobilenet_v1_1.0_224.ckpt", None, True)
        # print("Model saved in file: %s" % save_path)