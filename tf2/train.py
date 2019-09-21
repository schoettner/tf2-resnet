import argparse
import logging
import sys

import tensorflow as tf

from dataset.dataset import create_dataset, load_data_array, split_dataset
from model.legacy.tf1.ResNet34 import ResNet34

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_size', type=int, default=224)
FLAGS = parser.parse_args()


def main():
    # tf.logging.set_verbosity('DEBUG')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # disable eager for the moment
    # tf.disable_eager_execution()

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    input_size = FLAGS.input_size

    df, labels = load_data_array('data/dataset.csv')
    num_classes = len(labels)
    train_set = split_dataset(df, 'TRAIN')
    eval_set = split_dataset(df, 'VALIDATION')
    test_set = split_dataset(df, 'TEST')

    train_inputs, train_steps = create_dataset(data=train_set, num_classes=num_classes,
                                               input_size=input_size, batch_size=batch_size,
                                               epochs=epochs, is_training=True)
    eval_inputs, eval_steps = create_dataset(data=eval_set, num_classes=num_classes,
                                             input_size=input_size, batch_size=batch_size,
                                             epochs=epochs, is_training=False)
    test_inputs, test_steps = create_dataset(data=test_set, num_classes=num_classes,
                                             input_size=input_size, batch_size=batch_size,
                                             epochs=epochs, is_training=False)
    model = ResNet34(num_classes=num_classes)
    with tf.InteractiveSession() as sess:
        model.compile(session=sess,
                      train_inputs=train_inputs, train_steps=train_steps,
                      eval_inputs=eval_inputs, eval_steps=eval_steps,
                      test_inputs=test_inputs, test_steps=test_steps,
                      )
        model.train(epochs=epochs)
        model.test()


if __name__ == "__main__":
    main()
