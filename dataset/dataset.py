import logging

import pandas as pd

from dataset.input_fn import input_fn


def load_data_array(csv_file: str = 'data/dataset.csv'):
    df = pd.read_csv(csv_file, names=['SET', 'FILE', 'LABEL'])
    # get the labels as string
    labels = df['LABEL'].unique()
    logging.debug('labels: {}'.format(labels))
    # convert the label from string to categorical numerics
    df['LABEL'] = df['LABEL'].astype('category').cat.codes
    return df, list(labels)


def split_dataset(df: pd.DataFrame, dataset: str):
    assert dataset in ['TRAIN', 'VALIDATION', 'TEST'], 'Not a valid dataset'
    sub_df = df[df['SET'] == dataset]
    logging.debug("Dataframe header: \n{}".format(sub_df.head()))
    return sub_df


def create_dataset(data: pd.DataFrame,
                   num_classes: int,
                   input_size: int,
                   batch_size: int,
                   epochs: int,
                   is_training: bool):
    filenames = data['FILE'].to_list()
    labels = data['LABEL'].to_list()
    num_entries = len(filenames)
    assert num_entries == len(labels), 'Length of labels is not equal to image list'
    steps = (num_entries - 1) // batch_size

    inputs = input_fn(filenames=filenames,
                      labels=labels,
                      shuffle_size=num_entries,
                      num_classes=num_classes,
                      input_size=input_size,
                      batch_size=batch_size,
                      epochs=epochs,
                      is_training=is_training)

    return inputs, steps
