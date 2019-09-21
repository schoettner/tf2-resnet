""" references for resnet
Check https://arxiv.org/abs/1512.03385 for resnet
Check http://arxiv.org/pdf/1603.05027v2.pdf for bottleneck
"""
import tensorflow as tf


class ResNet:

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.channel_axis = 3

    def _bn_relu_conv2d(self,
                        layer_input,
                        filters: int,
                        kernel_size,
                        strides=(1, 1),
                        padding='same'):
        conv_kernel = tf.random_uniform((kernel_size, kernel_size, filters, filters))
        bn = tf.keras.backend.batch_normalization(x=layer_input, axis=self.channel_axis)
        relu = tf.keras.backend.relu(x=bn)
        conv = tf.keras.backend.conv2d(x=relu,
                                       kernel=conv_kernel,
                                       strides=strides,
                                       padding=padding)
        return conv

    def _conv2d_bn_relu(self,
                        layer_input,
                        filters: int,
                        kernel_size,
                        strides=(1, 1),
                        padding='same'):
        conv_kernel = tf.random_uniform((kernel_size, kernel_size, filters, filters))
        conv = tf.keras.backend.conv2d(x=layer_input,
                                       kernel=conv_kernel,
                                       strides=strides,
                                       padding=padding)
        bn = tf.keras.backend.batch_normalization(x=conv, axis=self.channel_axis)
        relu = tf.keras.backend.relu(x=bn)
        return relu

    def _residual_block(self, block_input, filters, adjust_strides: bool = False):
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(layer_input=block_input,
                                     filters=filters,
                                     kernel_size=(3, 3),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(layer_input=conv1,
                                     filters=filters,
                                     kernel_size=(3, 3))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            conv_kernel = tf.random_uniform((1, 1, filters, filters * 2))
            shortcut_layer = tf.keras.backend.conv2d(x=block_input,
                                                     kernel=conv_kernel,
                                                     padding='valid')
        return tf.keras.layers.add([shortcut_layer, conv2])

    def _bottleneck_block(self, block_input, filters, adjust_strides: bool = False):

        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(layer_input=block_input,
                                     filters=filters,
                                     kernel_size=(1, 1),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(layer_input=conv1,
                                     filters=filters,
                                     kernel_size=(3, 3))
        conv3 = self._bn_relu_conv2d(layer_input=conv2,
                                     filters=filters,
                                     kernel_size=(1, 1))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            conv_kernel = tf.random_uniform((1, 1, filters, filters * 2))
            shortcut_layer = tf.keras.backend.conv2d(x=block_input,
                                                     kernel=conv_kernel,
                                                     padding='valid')
        return tf.keras.layers.add([shortcut_layer, conv3])

    def _head(self, model_inputs):
        conv1 = self._conv2d_bn_relu(layer_input=model_inputs,
                                     filters=64,
                                     kernel_size=(7, 7),
                                     strides=(2, 2))
        pool1 = tf.keras.backend.pool2d(x=conv1,
                                        pool_size=(3, 3),
                                        strides=(2, 2),
                                        padding='same',
                                        pool_mode='max')
        return pool1

    def _classification(self, classification_input):
        bn_classification = tf.keras.backend.batch_normalization(x=classification_input, axis=self.channel_axis)
        relu_classification = tf.keras.backend.relu(x=bn_classification)
        pool_classification = tf.keras.backend.pool2d(x=relu_classification,
                                                      pool_size=(2, 2),
                                                      strides=(1, 1),
                                                      padding='valid',
                                                      pool_mode='avg')
        flatt_classification = tf.keras.backend.flatten(x=pool_classification)
        dense_classification = tf.layers.dense(inputs=flatt_classification,
                                               units=self.num_classes)
        logits = tf.keras.backend.softmax(x=dense_classification)
        return logits
