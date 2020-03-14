from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import uuid
from argparse import Namespace

from dataset.food_dataset_provider import FoodDatasetProvider
from model.model_provider import load_model, get_loss_function, get_optimizer, get_callbacks
from trainer.train import train, export_model, evaluate


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()

    # fix params
    parser.add_argument('--job-dir', help='google bucket where all logs and exports are', type=str, required=True)
    parser.add_argument('--id', help='identifier for machine if distributed', type=str, default=str(uuid.uuid1()))
    parser.add_argument('--epochs', help='amount of epochs to train', type=int, default=1)
    parser.add_argument('--frozen_layers', help='first n layers to freeze (not trained)', type=int, default=0)
    parser.add_argument('--channels', help='color channels [1, 3 or 4]', type=int, default=3)
    parser.add_argument('--monitor_loss', help='loss to monitor for model evaluation [loss, acc, val_loss, val_acc]', type=str, default='val_loss')
    parser.add_argument('--initial_weights', help='initial weights for the model', type=str, default=None)
    parser.add_argument('--label_file', help='label file of the dataset to define model output size', type=str, default=None)

    # hyper params
    parser.add_argument('--batch_size', help='batch_size for training', type=int, default=32)
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=1.e-4)
    parser.add_argument('--lr_decay', help='epoch to start learning rate reduction', type=int, default=10)
    parser.add_argument('--early_stopping', help='epoch to stop training if monitor value gets worse', type=int, default=10)
    parser.add_argument('--input_width', help='width of model input', type=int, default=448)
    parser.add_argument('--input_height', help='height of model input', type=int, default=448)
    parser.add_argument('--loss', help='loss function to use [crossentropy, mse, hinge, cosh, poisson]', type=str, default='crossentropy')
    parser.add_argument('--optimizer', help='optimizer to use [adam, sgd, adagrad, adamax, rms]', type=str, default='adam')
    parser.add_argument('--base_model', help='base_model to use [resnet50, resnet101, mobilenet]', type=str, default='resnet50')

    # test config
    parser.add_argument('--test', help='test model by only using 3 steps per epoch', dest='test', action='store_true')
    parser.set_defaults(test=False)
    return parser.parse_args()


def train_model(args: Namespace):
    logger = logging.getLogger('TrainingLogger')
    logger.info('Training for {} epochs...'.format(args.epochs))

    model, train_dataset, eval_dataset, test_dataset = get_local_model(args, logger)

    logger.info('Start training...')
    callbacks = get_callbacks(args)
    train(args=args,
          model=model,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          callbacks=callbacks)

    logger.info('Export model...')
    export_model(args, model)

    logger.info('Evaluate model...')
    evaluate(args, model, test_dataset)


def get_local_model(args: Namespace, logger):
    logger.info('Download dataset...')
    provider = FoodDatasetProvider(args)
    train_dataset = provider.load_train_dataset()
    eval_dataset = provider.load_eval_dataset()
    test_dataset = provider.load_test_dataset()

    logger.info('Load ML model...')
    model = load_model(args, output_size=101)
    model.summary()

    logger.info('Get training functions...')
    optimizer = get_optimizer(args)
    loss_fn = get_loss_function(args)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    return model, train_dataset, eval_dataset, test_dataset


if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)
