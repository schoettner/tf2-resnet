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
                        inputs: tf.keras.layers.Layer,
                        filters: int,
                        kernel_size: (),
                        strides=(1, 1),
                        padding='same') -> tf.keras.layers.Layer:
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(inputs)
        relu = tf.keras.layers.ReLU()(bn)
        conv = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      # kernel_regularizer=tf.keras.regularizers.l2,
                                      strides=strides,
                                      padding=padding)(relu)
        return conv

    def _conv2d_bn_relu(self,
                        inputs: tf.keras.layers.Layer,
                        filters: int,
                        kernel_size: (),
                        strides=(1, 1),
                        padding='same') -> tf.keras.layers.Layer:
        conv = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size,
                                      # kernel_regularizer=tf.keras.regularizers.l2,
                                      strides=strides,
                                      padding=padding)(inputs)
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(conv)
        relu = tf.keras.layers.ReLU()(bn)
        return relu

    def _residual_block(self, block_input: tf.keras.layers.Layer, filters: int,
                        adjust_strides: bool = False) -> tf.keras.layers.Layer:
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(inputs=block_input,
                                     filters=filters,
                                     kernel_size=(3, 3),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(inputs=conv1,
                                     filters=filters,
                                     kernel_size=(3, 3))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            shortcut_layer = tf.keras.layers.Conv2D(filters=filters,
                                                    kernel_size=(1, 1),
                                                    # kernel_regularizer=tf.keras.regularizers.l2,
                                                    strides=stride,
                                                    padding='valid')(block_input)
        return tf.keras.layers.Add()([shortcut_layer, conv2])

    def _bottleneck_block(self, block_input: tf.keras.layers.Layer, filters: int,
                          adjust_strides: bool = False) -> tf.keras.layers.Layer:
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(inputs=block_input,
                                     filters=filters,
                                     kernel_size=(1, 1),
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(inputs=conv1,
                                     filters=filters,
                                     kernel_size=(3, 3))
        conv3 = self._bn_relu_conv2d(inputs=conv2,
                                     filters=filters,
                                     kernel_size=(1, 1))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            shortcut_layer = tf.keras.layers.Conv2D(filters=filters,
                                                    kernel_size=(1, 1),
                                                    # kernel_regularizer=tf.keras.regularizers.l2,
                                                    strides=stride,
                                                    padding='valid')(block_input)
        return tf.keras.layers.Add()([shortcut_layer, conv3])

    def _head(self, model_input: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        conv1 = self._conv2d_bn_relu(inputs=model_input,
                                     filters=64,
                                     kernel_size=(7, 7),
                                     strides=(2, 2))
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')(conv1)
        return pool1

    def _classification(self, feature_network: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        bn = tf.keras.layers.BatchNormalization(axis=self.bn_axis)(feature_network)
        relu = tf.keras.layers.ReLU()(bn)
        pool = tf.keras.layers.AvgPool2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='valid')(relu)
        flatt = tf.keras.layers.Flatten()(pool)
        dense = tf.keras.layers.Dense(units=self.num_classes,
                                      name='logits',
                                      # kernel_regularizer=tf.keras.regularizers.l2,
                                      activation='softmax')(flatt)
        return dense
