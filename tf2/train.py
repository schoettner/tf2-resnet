import argparse
import tensorflow as tf

from dataset.dataset import create_dataset, load_data_array, split_dataset
from model.ModelBuilder import build_resnet34
from model.legacy.tf1.ResNet34 import ResNet34

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_size', type=int, default=224)
FLAGS = parser.parse_args()

model = build_resnet34(classes=10, input_shape=(224, 224, 3))
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
eval_loss = tf.keras.metrics.Mean(name='eval_loss')
eval_accuracy = tf.keras.metrics.CategoricalAccuracy(name='eval_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


def main():
    print(tf.__version__)

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    input_size = FLAGS.input_size

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

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Eval Loss: {}, Eval Accuracy: {}'
    for epoch in range(epochs):
        # train
        for image_batch, label_batch in train_ds:
            train_step(image_batch, label_batch)
        # eval
        for image_batch, label_batch in eval_ds:
            eval_step(image_batch, label_batch)
        # print results
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              eval_loss.result(),
                              eval_accuracy.result()*100))
        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        eval_loss.reset_states()
        eval_accuracy.reset_states()
    # test
    for image_batch, label_batch in test_ds:
        test_step(image_batch, labels)
    print('Test Loss: {}, Test Accuracy: {}').format(test_loss.result(), test_accuracy.result()*100)


@tf.function
def train_step(image_batch, label_batch):
    with tf.GradientTape() as tape:
        predictions = model(image_batch)
        loss = loss_fn(label_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(label_batch, predictions)


@tf.function
def eval_step(image_batch, label_batch):
    predictions = model(image_batch)
    e_loss = loss_fn(label_batch, predictions)
    eval_loss(e_loss)
    eval_accuracy(label_batch, predictions)


@tf.function
def test_step(image_batch, label_batch):
    predictions = model(image_batch)
    t_loss = loss_fn(label_batch, predictions)
    test_loss(t_loss)
    test_accuracy(label_batch, predictions)


if __name__ == "__main__":
    main()
