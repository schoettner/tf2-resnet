import tensorflow as tf
from tensorflow_core.python.keras.engine.input_layer import Input

from model.ResNet import ResNet


class ResNet34(ResNet):

    def __init__(self, num_classes: int, input_shape: ()):
        super().__init__(num_classes)
        self.input_shape = input_shape

    def build(self):
        model_inputs = tf.keras.layers.Input(shape=self.input_shape)
        head_end = self._head(model_inputs)

        block64_1 = self._residual_block(head_end[-1], 64)
        block64_2 = self._residual_block(block64_1[-1], 64)
        block64_3 = self._residual_block(block64_2[-1], 64)

        block128_1 = self._residual_block(block64_3[-1], 128, adjust_strides=True)
        block128_2 = self._residual_block(block128_1[-1], 128)
        block128_3 = self._residual_block(block128_2[-1], 128)
        block128_4 = self._residual_block(block128_3[-1], 128)

        block256_1 = self._residual_block(block128_4[-1], 256, adjust_strides=True)
        block256_2 = self._residual_block(block256_1[-1], 256)
        block256_3 = self._residual_block(block256_2[-1], 256)
        block256_4 = self._residual_block(block256_3[-1], 256)
        block256_5 = self._residual_block(block256_4[-1], 256)
        block256_6 = self._residual_block(block256_5[-1], 256)

        block512_1 = self._residual_block(block256_6[-1], 512, adjust_strides=True)
        block512_2 = self._residual_block(block512_1[-1], 512)
        block512_3 = self._residual_block(block512_2[-1], 512)

        output = self._classification()