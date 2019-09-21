from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import tensorflow as tf


from dataset.dataset import load_data_array, split_dataset, create_dataset
from model.ModelBuilder import build_resnet50

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_size', type=int, default=224)
FLAGS = parser.parse_args()


def main():
    print(tf.__version__)

    epochs = FLAGS.epochs
    input_size = FLAGS.input_size
    batch_size = FLAGS.batch_size

    print('Create Datasets...')
    df, labels = load_data_array('data/dataset.csv')
    num_classes = len(labels)
    train_set = split_dataset(df, 'TRAIN')
    eval_set = split_dataset(df, 'VALIDATION')
    test_set = split_dataset(df, 'TEST')
    train_ds, train_steps = create_dataset(data=train_set, num_classes=num_classes,
                              input_size=input_size, batch_size=batch_size,
                              epochs=epochs, is_training=True)
    eval_ds, eval_steps = create_dataset(data=eval_set, num_classes=num_classes,
                             input_size=input_size, batch_size=batch_size,
                             epochs=epochs, is_training=False)
    test_ds, test_steps = create_dataset(data=test_set, num_classes=num_classes,
                             input_size=input_size, batch_size=batch_size,
                             epochs=epochs, is_training=False)

    print('Compile Model for Training...')
    model: tf.keras.models.Model = build_resnet50(classes=num_classes, input_shape=(input_size, input_size, 3))
    # model = build_small(classes=num_classes, input_shape=(input_size, input_size, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])
    model.summary()

    loss_to_monitor = 'val_loss'
    callbacks = [
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(patience=2, monitor=loss_to_monitor),
        tf.keras.callbacks.ModelCheckpoint(filepath='output/checkpoint',
                        monitor=loss_to_monitor,
                        verbose=0,
                        save_best_only=True,
                        mode='auto',
                        save_freq=1)
    ]

    print('Starting Training...')
    model.fit(train_ds,
              epochs=epochs,
              steps_per_epoch=train_steps,
              callbacks=callbacks,
              validation_data=eval_ds,
              validation_steps=eval_steps,
              validation_freq=1,
              )

    print('Training completed. Saving Model...')
    save_dir = 'output'
    model_dir = 'model'
    scores = model.evaluate(test_ds, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_dir)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == "__main__":
    main()
