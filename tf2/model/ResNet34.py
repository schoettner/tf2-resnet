import tensorflow as tf
from tf2.model.ResNet import ResNet


class ResNet34(ResNet):

    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def build(self, model_inputs, is_training: bool, reuse: bool):
        with tf.variable_scope('resnet34', reuse=reuse):
            with tf.variable_scope('head'):
                head_end = self._head(model_inputs, is_training)
            with tf.variable_scope('64_filter_block'):
                block64_1 = self._residual_block(head_end, 64, is_training)
                block64_2 = self._residual_block(block64_1, 64, is_training)
                block64_3 = self._residual_block(block64_2, 64, is_training)
            with tf.variable_scope('128_filter_block'):
                block128_1 = self._residual_block(block64_3, 128, is_training, adjust_strides=True)
                block128_2 = self._residual_block(block128_1, 128, is_training)
                block128_3 = self._residual_block(block128_2, 128, is_training)
                block128_4 = self._residual_block(block128_3, 128, is_training)
            with tf.variable_scope('256_filter_block'):
                block256_1 = self._residual_block(block128_4, 256, is_training, adjust_strides=True)
                block256_2 = self._residual_block(block256_1, 256, is_training)
                block256_3 = self._residual_block(block256_2, 256, is_training)
                block256_4 = self._residual_block(block256_3, 256, is_training)
                block256_5 = self._residual_block(block256_4, 256, is_training)
                block256_6 = self._residual_block(block256_5, 256, is_training)
            with tf.variable_scope('512_filter_block'):
                block512_1 = self._residual_block(block256_6, 512, is_training, adjust_strides=True)
                block512_2 = self._residual_block(block512_1, 512, is_training)
                block512_3 = self._residual_block(block512_2, 512, is_training)
            with tf.variable_scope('classification'):
                output = self._classification(block512_3, is_training)
        return output
