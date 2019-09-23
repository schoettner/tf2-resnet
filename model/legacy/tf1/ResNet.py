import tensorflow as tf

from model.legacy.tf1.Model import Model


class ResNet(Model):
    """provide build blocks required fall all resnet implementations
    base structure is taken from the original paper and the keras.applications implementation"""

    def __init__(self, num_classes: int):
        """requires rgb picture with channel last (NHWC & channel_last)
        Args:
          images: 4-D Tensor of images with Shape [batch_size, image_width, image_height, 3]
          num_classes: amount of units on output layer
        Return:
          A wrapper For building model
        """
        self.num_classes = num_classes

    def _bn_relu_conv2d(self,
                        layer_input,
                        filters: int,
                        kernel_size,
                        is_training: bool,
                        strides=(1, 1),
                        padding='same'):
        """Build a pre-activation block with
        Check http://arxiv.org/pdf/1603.05027v2.pdf for details
        """
        bn = self._batch_norm(layer_input, is_training)
        relu = self._relu(bn)
        conv = tf.layers.conv2d(inputs=relu,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding)
        return conv

    def _conv2d_bn_relu(self,
                        layer_input,
                        filters: int,
                        kernel_size,
                        is_training: bool,
                        strides=(1, 1),
                        padding='same'):
        conv = tf.layers.conv2d(inputs=layer_input,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding)
        bn = self._batch_norm(conv, is_training)
        relu = self._relu(bn)
        return relu

    def _residual_block(self, block_input, filters, is_training: bool, adjust_strides: bool = False):
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(layer_input=block_input,
                                     filters=filters,
                                     kernel_size=(3, 3),
                                     is_training=is_training,
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(layer_input=conv1,
                                     filters=filters,
                                     is_training=is_training,
                                     kernel_size=(3, 3))

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            shortcut_layer = tf.layers.conv2d(inputs=block_input,
                                              filters=filters,
                                              kernel_size=(1, 1),
                                              strides=stride,
                                              padding='valid')
        return tf.add(x=shortcut_layer, y=conv2)

    def _bottleneck_block(self, block_input, filters, is_training: bool, adjust_strides: bool = False):
        """other order than regular residual block
        Check http://arxiv.org/pdf/1603.05027v2.pdf for details about bottleneck
        """
        if adjust_strides:
            stride = (2, 2)
        else:
            stride = (1, 1)

        # conv blocks
        conv1 = self._bn_relu_conv2d(layer_input=block_input,
                                     filters=filters,
                                     kernel_size=(1, 1),
                                     is_training=is_training,
                                     strides=stride)
        conv2 = self._bn_relu_conv2d(layer_input=conv1,
                                     filters=filters,
                                     kernel_size=(3, 3),
                                     is_training=is_training)
        conv3 = self._bn_relu_conv2d(layer_input=conv2,
                                     filters=filters,
                                     kernel_size=(1, 1),
                                     is_training=is_training)

        # if stride was changed, you need a conv2d layer in the shortcut path to adjust the shape
        shortcut_layer = block_input
        if adjust_strides:
            shortcut_layer = tf.layers.conv2d(inputs=block_input,
                                              filters=filters,
                                              kernel_size=(1, 1),
                                              strides=stride,
                                              padding='valid')
        return tf.add(x=shortcut_layer, y=conv3)

    def _head(self, model_inputs, is_training: bool):
        """Build the input section of ResNet"""
        conv1 = self._conv2d_bn_relu(layer_input=model_inputs,
                                     filters=64,
                                     kernel_size=(7, 7),
                                     is_training=is_training,
                                     strides=(2, 2))
        pool1 = tf.nn.max_pool_v2(input=conv1,
                                  ksize=(3, 3),
                                  strides=(2, 2),
                                  padding='same')
        return pool1

    def _classification(self, classification_input, is_training: bool):
        """Build the classification end of the ResNet model.
        """
        bn_classification = self._batch_norm(classification_input, is_training)  # norm over depth
        relu_classification = self._relu(bn_classification)
        pool_classification = tf.nn.avg_pool_v2(input=relu_classification,
                                                ksize=(2, 2),
                                                strides=(1, 1),
                                                padding='valid')
        flatt_classification = tf.layers.flatten(pool_classification)
        dense_classification = tf.layers.dense(inputs=flatt_classification,
                                               units=self.num_classes,
                                               name='logits',
                                               activation='softmax')
        return dense_classification
