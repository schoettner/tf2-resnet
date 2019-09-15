"""https://keras.io/examples/cifar10_cnn/
"""
import logging
import sys
import os

from tensorflow_core.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow_core.python.keras.datasets import cifar10
from tensorflow_core.python.keras.losses import categorical_crossentropy
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.utils.np_utils import to_categorical
from tensorflow_core.python.platform.tf_logging import set_verbosity
from tensorflow_core.python.training import tensorboard_logging

from keras.model.ModelBuilder import build_small


def main():
    set_verbosity(tensorboard_logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    num_classes = 10
    epochs = 2
    batch_size = 64

    # create dataset
    print('create dataset')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('Compile Model for Training')
    model = build_small(classes=num_classes)
    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy,
                  metrics=['acc'])
    model.summary()

    loss_to_monitor = 'val_loss'
    callbacks = [
        TensorBoard(),
        EarlyStopping(patience=2, monitor=loss_to_monitor),
        ModelCheckpoint(filepath='output',
                        monitor=loss_to_monitor,
                        verbose=0,
                        save_best_only=True,
                        mode='auto',
                        period=1)
    ]

    print('Starting Training')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(x_test, y_test),
              shuffle=True)

    print('Training completed. Saving Model...')
    save_dir = 'output'
    model_name = 'keras_example'
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == "__main__":
    main()
