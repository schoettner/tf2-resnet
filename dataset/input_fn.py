# from tensorflow_core.python.data.ops.dataset_ops import Dataset
import tensorflow as tf
from tensorflow_core.python.ops.gen_image_ops import decode_jpeg


def one_hot_encode(image, label, num_classes: int):
    return image, tf.one_hot(label, num_classes)


def load_img(filename: str, label, size: int):
    image_string = tf.io.read_file(filename)
    image_decoded = decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize(image, [size, size])
    return resized_image, label


def augment_img(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # make sure the augmentation did not break normalization
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def input_fn(filenames: [],
             labels: [],
             shuffle_size: int,
             num_classes: int,
             input_size: int,
             batch_size: int,
             epochs: int,
             is_training: bool):
    load_fn = lambda f, l: load_img(f, l, input_size)
    augment_fn = lambda f, l: augment_img(f, l)
    one_hot_fn = lambda f, l: one_hot_encode(f, l, num_classes)

    if is_training:
        dataset = (
            tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(shuffle_size)
                .repeat(epochs)
                .map(load_fn, num_parallel_calls=4)
                .map(augment_fn, num_parallel_calls=4)
                .map(one_hot_fn, num_parallel_calls=4)
                .batch(batch_size)
                .prefetch(2)
        )

    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(load_fn, num_parallel_calls=4)
                   .map(one_hot_fn, num_parallel_calls=4)
                   .batch(batch_size)
                   .prefetch(1)
                   )
    return dataset
