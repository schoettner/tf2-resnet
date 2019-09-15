
import logging
import sys

from dataset.builder import create_datasets
from tensorflow.model.ResNet34 import ResNet34


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    labels = None
    num_classes = len(labels)
    (x_train, y_train), (x_eval, y_eval), (x_test, y_test) = create_datasets()
    model = ResNet34(num_classes=num_classes)
