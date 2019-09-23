import tensorflow as tf


class ModelTrainer:

    def __init__(self):
        self.model: tf.keras.models.Model = None
        self.loss_fn = None
        self.optimizer = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def eval_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_fn(labels, predictions)

        self.eval_loss(t_loss)
        self.eval_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_fn(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def rest_metrics(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.eval_loss.reset_states()
        self.eval_accuracy.reset_states()
