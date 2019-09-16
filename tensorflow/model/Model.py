"""https://www.tensorflow.org/beta/guide/effective_tf2
"""
import logging
import os

from tensorflow_core.python.client.session import Session
from tensorflow_core.python.framework.dtypes import float32
from tensorflow_core.python.framework.ops import get_collection, GraphKeys
from tensorflow_core.python.layers.normalization import batch_normalization
from tensorflow_core.python.ops import metrics_impl
from tensorflow_core.python.ops.control_flow_ops import group
from tensorflow_core.python.ops.gen_math_ops import equal
from tensorflow_core.python.ops.gen_nn_ops import Relu
from tensorflow_core.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow_core.python.ops.math_ops import reduce_mean, cast
from tensorflow_core.python.ops.variable_scope import variable_scope
from tensorflow_core.python.ops.variables import variables_initializer
from tensorflow_core.python.summary import summary
from tensorflow_core.python.training.adam import AdamOptimizer
from tensorflow_core.python.training.saver import Saver

from tensorflow.model.Trainable import Trainable


class Model(Trainable):

    def build(self, model_inputs, is_training: bool, reuse: bool):
        raise NotImplementedError('Not implemented for abstract model')

    def compile(self, session: Session,
                train_inputs: dict, train_steps: int,
                eval_inputs: dict, eval_steps: int,
                test_inputs: dict, test_steps: int, ):
        self.session = session
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.test_steps = test_steps

        # initiate the iterators
        session.run([train_inputs['iterator_init_op'], eval_inputs['iterator_init_op'], test_inputs['iterator_init_op']])
        x_train, y_train = train_inputs['iterator'].get_next()
        x_eval, y_eval = eval_inputs['iterator'].get_next()
        x_test, y_test = test_inputs['iterator'].get_next()

        self.train_spec = self._model_fn(inputs={'images': x_train, 'labels': y_train}, is_training=True)
        self.eval_spec = self._model_fn(inputs={'images': x_eval, 'labels': y_eval}, is_training=False)
        self.test_spec = self._model_fn(inputs={'images': x_test, 'labels': y_test}, is_training=False)

    def _model_fn(self, inputs: dict, is_training: bool):
        images = inputs['images']
        labels = inputs['labels']
        # share weights between train/eval/test. with reuse, eval/test can access the weights of training
        reuse = not is_training
        logits = self.build(images, is_training, reuse)

        loss = softmax_cross_entropy(logits=logits, onehot_labels=labels)
        if is_training:
            with variable_scope('optimizer'):
                optimizer = AdamOptimizer()
                # global step no longer required in tf2
                # make sure to also get update ops for batch normalization
                update_ops = get_collection(GraphKeys.UPDATE_OPS)
                train_op = optimizer.minimize(loss)
                train_op = group([train_op, update_ops])

        with variable_scope('metrics'):
            metrics = {
                'accuracy': metrics_impl.accuracy(labels=labels, predictions=logits),
                'loss'    : metrics_impl.mean(loss)
            }
        update_metrics_op = group(*[op for _, op in metrics.values()])
        metric_variables = get_collection(GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_init_op = variables_initializer(metric_variables)

        # additional summary
        with variable_scope('summary'):
            acc = reduce_mean(cast(equal(labels, logits), float32))

        summary.scalar('accuracy', acc)
        summary.scalar('loss', reduce_mean(loss))
        summary.image('train_image', images)

        model_spec = inputs
        model_spec['loss'] = loss
        model_spec['accuracy'] = acc
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['summary_op'] = summary.merge_all()
        model_spec['update_metrics'] = update_metrics_op
        if is_training:
            model_spec['train_op'] = train_op
        return model_spec

    def train(self, epochs: int):
        if not self.session:
            raise RuntimeError('Model was no session. Please compile the model before training!')
        last_saver = Saver(max_to_keep=3)  # keep last 3 epochs
        best_saver = Saver(max_to_keep=1)  # keep best epoch

        best_eval_accuracy = 0.0
        last_save_path = os.path.join('output/checkpoints', 'last_weights', 'after-epoch')
        best_save_path = os.path.join('output/checkpoints', 'best_weights', 'after-epoch')
        for epoch in range(0, epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))

            # train
            self._train_sess(sess=self.session, train_spec=self.train_spec, training_steps=self.train_steps)
            last_saver.save(self.session, last_save_path, global_step=epoch + 1)

            # evaluate
            eval_accuracy = self._eval_sess(sess=self.session, eval_spec=self.eval_spec, eval_steps=self.eval_steps)['accuracy']
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_save_path = best_saver.save(self.session, best_save_path, global_step=epoch + 1)
                logging.info("Found new best accuracy, saving in {}".format(best_save_path))
        logging.info('Training for all epochs complete')

    def test(self):
        if not self.session:
            raise RuntimeError('Model was no session. Please compile the model before evaluation with test-set!')
        raise NotImplementedError('todo')

    @staticmethod
    def _relu(inputs):
        return Relu(inputs)

    @staticmethod
    def _batch_norm(inputs,
                    is_training: bool,
                    name: str = 'batch_norm'):
        return batch_normalization(
            inputs=inputs,
            name=name,
            axis=3,
            training=is_training,
            fused=True)
