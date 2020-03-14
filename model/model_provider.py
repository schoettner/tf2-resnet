from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from argparse import Namespace

import tensorflow as tf


def load_model(args: Namespace, output_size: int) -> tf.keras.Model:
    channels = args.channels
    input_width = args.input_width
    input_height = args.input_height
    input_shape = [input_width, input_height, channels]
    num_classes = output_size
    init_weights = args.initial_weights
    frozen_layers = args.frozen_layers
    base_model = args.base_model
    input_tensor = tf.keras.layers.InputLayer(name='image',
                                              dtype=tf.float32,
                                              input_shape=input_shape).input
    if base_model == 'resnet50':
        model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,
                                                           weights=init_weights,
                                                           input_tensor=input_tensor,
                                                           pooling=None,
                                                           classes=num_classes)
    elif base_model == 'resnet101':
        model = tf.keras.applications.resnet_v2.ResNet101V2(include_top=True,
                                                            weights=init_weights,
                                                            input_tensor=input_tensor,
                                                            pooling=None,
                                                            classes=num_classes)
    elif base_model == 'mobilenet':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True,
                                                               weights=init_weights,
                                                               input_tensor=input_tensor,
                                                               pooling=None,
                                                               classes=num_classes)
    else:
        raise ValueError('Base model {} not supported'.format(base_model))
    if init_weights:
        print('load models')

    for i in range(frozen_layers):
        model.layers[i].trainable = False
    return model


def get_optimizer(args: Namespace) -> tf.keras.optimizers.Optimizer:
    optimizer = args.optimizer
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(args.learning_rate)
    elif optimizer == 'sgd':
        return tf.keras.optimizers.SGD(args.learning_rate)
    elif optimizer == 'rms':
        return tf.keras.optimizers.RMSprop(args.learning_rate)
    elif optimizer == 'adagrad':
        return tf.keras.optimizers.Adagrad(args.learning_rate)
    elif optimizer == 'adamax':
        return tf.keras.optimizers.Adamax(args.learning_rate)
    else:
        raise ValueError('Optimizer {} not supported'.format(optimizer))


def get_loss_function(args: Namespace) -> tf.keras.losses.Loss:
    loss = args.loss
    if loss == 'crossentropy':
        return tf.keras.losses.categorical_crossentropy
    elif loss == 'mse':
        return tf.keras.losses.mean_squared_error
    elif loss == 'hinge':
        return tf.keras.losses.categorical_hinge
    elif loss == 'cosh':
        return tf.keras.losses.logcosh
    elif loss == 'poisson':
        return tf.keras.losses.poisson
    else:
        raise ValueError('Loss function {} not supported'.format(loss))


def get_callbacks(args: Namespace) -> [tf.keras.callbacks.Callback]:
    # possible monitor metrics are: val_loss, train_loss, train_acc, val_acc
    loss_to_monitor = args.monitor_loss
    tensorboard_path = os.path.join(args.job_dir, args.id, 'training/tensorboard/')
    checkpoint_path = os.path.join(args.job_dir, args.id, 'training/checkpoint/best')
    # lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=loss_to_monitor, min_lr=1e-6)

    tb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True)
    stopping = tf.keras.callbacks.EarlyStopping(monitor=loss_to_monitor, patience=args.early_stopping, restore_best_weights=True)

    def scheduler(epoch):
        if epoch < args.lr_decay:
            return args.learning_rate
        else:
            return args.learning_rate * tf.math.exp(0.1 * (10 - epoch))

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
    return [tb, lr_schedule, checkpoint, stopping]

