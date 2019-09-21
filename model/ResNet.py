""" references for resnet
Check https://arxiv.org/abs/1512.03385 for resnet
Check http://arxiv.org/pdf/1603.05027v2.pdf for bottleneck
"""
import tensorflow as tf


class ResNet:

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.bn_axis = 3

    def _bn_relu_conv2d(self,
                        filters: int,
                        kernel_size,
                        strides=(1, 1),
                        padding='same') -> list:
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)
        relu = tf.keras.layers.ReLU()
        conv = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      kernel_regularizer=tf.keras.regularizers.l2,
                                      strides=strides,
                                      padding=padding)
        return [bn, relu, conv]

    def _conv2d_bn_relu(self,
                        filters: int,
                        kernel_size,
                        strides=(1, 1),
                        padding='same') -> list:
        conv = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      kernel_regularizer=tf.keras.regularizers.l2,
                                      strides=strides,
                                      padding=padding)
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)
        relu = tf.keras.layers.ReLU()
        return [conv, bn, relu]

    def _residual_block(self, block_input, filters, adjust_strides: bool = False):
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(filters=filters,
                                     kernel_size=(3, 3),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(filters=filters,
                                     kernel_size=(3, 3))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        if adjust_strides:
            shortcut_layer = tf.layers.conv2d(filters=filters,
                                              kernel_size=(1, 1),
                                              kernel_regularizer=tf.keras.regularizers.l2,
                                              padding='valid')
            add = tf.keras.layers.Add()([shortcut_layer, conv2])
            return [conv1, conv2, shortcut_layer, add]

        add = tf.keras.layers.Add()([block_input, conv2])
        return [conv1, conv2, add]

    def _bottleneck_block(self, block_input, filters, adjust_strides: bool = False):
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(filters=filters,
                                     kernel_size=(1, 1),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(filters=filters,
                                     kernel_size=(3, 3))
        conv3 = self._bn_relu_conv2d(filters=filters,
                                     kernel_size=(1, 1))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        if adjust_strides:
            shortcut_layer = tf.keras.layers.Conv2D(inputs=block_input,
                                                    filters=filters,
                                                    kernel_size=(1, 1),
                                                    kernel_regularizer=tf.keras.regularizers.l2,
                                                    padding='valid')
            add = tf.keras.layers.Add()([shortcut_layer, conv2])
            return [conv1, conv2, conv3, shortcut_layer, add]

        add = tf.keras.layers.Add()([block_input, conv2])
        return [conv1, conv2, conv3, add]

    def _head(self):
        conv1 = self._conv2d_bn_relu(filters=64,
                                     kernel_size=(7, 7),
                                     strides=(2, 2))
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')
        return [conv1, pool1]

    def _classification(self):
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)
        relu = tf.keras.layers.ReLU()
        pool = tf.keras.layers.AvgPool2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='valid')
        flatt = tf.keras.layers.Flatten()
        dense = tf.keras.layers.Dense(units=self.num_classes,
                                      name='logits',
                                      activation='softmax')
        return [bn, relu, pool, flatt, dense]
