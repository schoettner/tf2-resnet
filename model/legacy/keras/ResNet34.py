import tensorflow as tf

from model.legacy.keras.ResNet import ResNet


class ResNet34(ResNet):
    def __init__(self, num_classes: int):
        super(ResNet34, self).__init__(num_classes)

    def build(self, model_inputs):
        head_end = self._head(model_inputs)

        block64_1 = self._residual_block(head_end, 64)
        block64_2 = self._residual_block(block64_1, 64)
        block64_3 = self._residual_block(block64_2, 64)

        block128_1 = self._residual_block(block64_3, 128, adjust_strides=True)
        block128_2 = self._residual_block(block128_1, 128)
        block128_3 = self._residual_block(block128_2, 128)
        block128_4 = self._residual_block(block128_3, 128)

        block256_1 = self._residual_block(block128_4, 256, adjust_strides=True)
        block256_2 = self._residual_block(block256_1, 256)
        block256_3 = self._residual_block(block256_2, 256)
        block256_4 = self._residual_block(block256_3, 256)
        block256_5 = self._residual_block(block256_4, 256)
        block256_6 = self._residual_block(block256_5, 256)

        block512_1 = self._residual_block(block256_6, 512, adjust_strides=True)
        block512_2 = self._residual_block(block512_1, 512)
        block512_3 = self._residual_block(block512_2, 512)

        output = self._classification(block512_3)

        return tf.keras.models.Model(inputs=model_inputs, outputs=output)
