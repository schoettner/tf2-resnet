import logging

from tensorflow_core.python.client.session import Session
from tensorflow_core.python.ops.math_ops import reduce_mean
from tqdm import trange


class Trainable(object):

    @staticmethod
    def _train_sess(sess: Session, train_spec: dict, training_steps: int = 10):
        loss = train_spec['loss']
        train_op = train_spec['train_op']
        update_metrics = train_spec['update_metrics']
        metrics = train_spec['metrics']

        iterator_init = train_spec['iterator_init_op']
        metric_init = train_spec['metrics_init_op']
        sess.run([iterator_init, metric_init])

        training_steps = trange(training_steps)
        for _ in training_steps:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
            reduced_loss = sess.run(reduce_mean(loss_val))
            training_steps.set_postfix(loss='losses: {:05.3f}'.format(reduced_loss))

        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("Train metrics: " + metrics_string)

    @staticmethod
    def _eval_sess(sess: Session, eval_spec: dict, eval_steps: int = 10):
        update_metrics = eval_spec['update_metrics']
        eval_metrics = eval_spec['metrics']

        iterator_init = eval_spec['iterator_init_op']
        metric_init = eval_spec['metrics_init_op']
        sess.run([iterator_init, metric_init])

        eval_steps = trange(eval_steps)
        eval_steps.set_postfix('{evaluating model...}')
        for _ in eval_steps:
            sess.run(update_metrics)

        metrics_values = {k: v[0] for k, v in eval_metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("Eval metrics: " + metrics_string)
        return metrics_val
