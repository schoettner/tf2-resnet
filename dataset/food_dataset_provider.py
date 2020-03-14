from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace

import tensorflow as tf
import tensorflow_datasets as tfds


class FoodDatasetProvider:
    """
    test dataset with 101 classes of foods
    """

    def __init__(self, args: Namespace):
        self.dataset = 'food101'
        if args:
            self.batch_size = args.batch_size
            self.img_width = args.input_width
            self.img_height = args.input_height
        else:
            self.batch_size = 32
            self.img_width = 448
            self.img_height = 448

        self.converter = lambda data: convert_dataset(data, self.img_height, self.img_width)
        self.augmenter = lambda image, label: augment_img(image, label)

    def load_train_dataset(self) -> tf.data.Dataset:
        # https://github.com/tensorflow/datasets/blob/master/docs/splits.md
        split = 'train[:95%]'
        builder = tfds.builder(self.dataset)
        builder.download_and_prepare()

        ds = builder.as_dataset(split=split)
        ds = ds.map(self.converter, num_parallel_calls=4)
        ds = ds.map(self.augmenter, num_parallel_calls=4)
        ds = ds.batch(batch_size=self.batch_size)
        return ds

    def load_eval_dataset(self) -> tf.data.Dataset:
        split = 'train[-5%:]'
        builder = tfds.builder(self.dataset)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=split)
        ds = ds.map(self.converter, num_parallel_calls=4)
        ds = ds.batch(batch_size=self.batch_size)
        return ds

    def load_test_dataset(self) -> tf.data.Dataset:
        split = 'train[-5%:]'
        builder = tfds.builder(self.dataset)
        builder.download_and_prepare()
        ds = builder.as_dataset(split=split)
        ds = ds.map(self.converter, num_parallel_calls=4)
        ds = ds.batch(batch_size=self.batch_size)
        return ds


#######################################################################################
############################ SUPPORT FUNCTIONS ########################################
#######################################################################################

def convert_dataset(data, img_height: int, img_width: int):
    image = data['image']
    label = data['label']
    new_label = tf.one_hot(label, depth=101)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, img_width, img_height)

    # convert to [-1,1]
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2)
    return img, new_label


def augment_img(image, label):
    # https://arxiv.org/abs/1805.09501
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # do any augmentation you like. however, keep in mind the [-1,1] range (or remove it)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label


def measure_loading_time(ds: tf.data.Dataset, batches_to_load: int = 20, print_element: bool = False):
    import time

    batch_count = 0
    start = time.time()
    print('start loading data')
    for element in ds:
        batch_count += 1
        if print_element:
            print(element)
        if batch_count % 10 == 0:
            print('10 batches done, {} total after {} seconds'.format(batch_count, time.time() - start))
        if batch_count >= batches_to_load:
            break
    end = time.time()
    print('{} seconds for {} batches'.format(end - start, batch_count))


if __name__ == '__main__':
    # measure your loading time
    ds = FoodDatasetProvider(None).load_test_dataset()
    measure_loading_time(ds)
