import tensorflow as tf
import tensorflow.contrib.slim as slim

#######################
# 3d functions
#######################


# convolution
def conv3d(input, output_chn, kernel_size, stride, use_bias=False, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)


def conv_bn_relu(input, output_chn, kernel_size, stride, use_bias, is_training, name):
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride, use_bias, name='conv')
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


# deconvolution
def Deconv3d(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
                                  strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
    return conv


def deconv_bn_relu(input, output_chn, is_training, name):
    with tf.variable_scope(name):
        conv = Deconv3d(input, output_chn, name='deconv')
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

