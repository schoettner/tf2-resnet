from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
import logging
import os
import tensorflow as tf


def train(args: Namespace,
          model: tf.keras.Model,
          train_dataset: tf.data.Dataset,
          eval_dataset: tf.data.Dataset,
          callbacks: [tf.keras.callbacks.Callback] = None):
    if args.test:
        model.fit(train_dataset,
                  epochs=args.epochs,
                  steps_per_epoch=3,
                  validation_steps=3,
                  verbose=2,
                  validation_data=eval_dataset,
                  callbacks=callbacks,
                  )
    else:
        model.fit(train_dataset,
                  epochs=args.epochs,
                  validation_data=eval_dataset,
                  callbacks=callbacks,
                  )


def evaluate(args: Namespace,
             model: tf.keras.Model,
             test_dataset: tf.data.Dataset):
    if not args.test:
        metrics = model.evaluate(test_dataset)
        logging.info(metrics)


def export_model(args: Namespace,
                 model: tf.keras.Model):
    logging.info("exporting model and weights...")
    weight_path = os.path.join(args.job_dir, args.id, 'training/weights/weights')
    model_path = os.path.join(args.job_dir, args.id, 'training/model')
    model.save_weights(filepath=weight_path, save_format='tf')
    model.save(filepath=model_path, save_format='tf')
    logging.info("export complete!")
