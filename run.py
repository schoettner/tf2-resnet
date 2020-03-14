from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trainer.task import parse_arguments, train_model

if __name__ == '__main__':
    args = parse_arguments()
    train_model(args)

