from tensorflow_core.python.layers.normalization import batch_normalization
from tensorflow_core.python.ops.gen_nn_ops import Relu

class Model(object):

    def __init__(self, is_training):
        self.is_training = is_training

    def build(self):
        raise NotImplementedError('Not implemented for this model')

    def compile(self):
        raise NotImplementedError('Not implemented for this model')

    def train(self):
        raise NotImplementedError('Not implemented for this model')

    def export(self):
        raise NotImplementedError('Not implemented for this model')

    def save(self):
        raise NotImplementedError('Not implemented for this model')

    def load(self):
        raise NotImplementedError('Not implemented for this model')

    def deploy(self):
        raise NotImplementedError('Not implemented for this model')

    def predict(self):
        raise NotImplementedError('Not implemented for this model')

    def _batch_norm(self, inputs, name: str = 'batch_norm'):
        return batch_normalization(
            inputs=inputs,
            name=name,
            axis=3,
            training=self.is_training,
            fused=True)

    def _relu(self, inputs):
        return Relu(inputs)
