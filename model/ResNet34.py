from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_core.python.ops.variable_scope import variable_scope

from model.ResNet import ResNet


class ResNet34(ResNet):

    def __init__(self, images, num_classes: int, is_training: bool = True):
        super().__init__(is_training, images, num_classes)

    def build(self):
        head_end = self._head()

        with variable_scope('ResNet_64'):
            block64_1 = self._residual_block(head_end, 64)
            block64_2 = self._residual_block(block64_1, 64)
            block64_3 = self._residual_block(block64_2, 64)
        with variable_scope('ResNet_128'):
            block128_1 = self._residual_block(block64_3, 128, adjust_strides=True)
            block128_2 = self._residual_block(block128_1, 128)
            block128_3 = self._residual_block(block128_2, 128)
            block128_4 = self._residual_block(block128_3, 128)
        with variable_scope('ResNet_256'):
            block256_1 = self._residual_block(block128_4, 256, adjust_strides=True)
            block256_2 = self._residual_block(block256_1, 256)
            block256_3 = self._residual_block(block256_2, 256)
            block256_4 = self._residual_block(block256_3, 256)
            block256_5 = self._residual_block(block256_4, 256)
            block256_6 = self._residual_block(block256_5, 256)
        with variable_scope('ResNet_512'):
            block512_1 = self._residual_block(block256_6, 512, adjust_strides=True)
            block512_2 = self._residual_block(block512_1, 512)
            block512_3 = self._residual_block(block512_2, 512)

        output = self._classification(block512_3)
        return output
