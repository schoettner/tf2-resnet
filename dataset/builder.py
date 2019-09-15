
def create_datasets(dataset_file: str = 'dataset.csv'):
    """create iterators for all three datasets
    based on the tutorial i wrote on medium
    """
    x_train = None
    y_train = None
    x_eval = None
    y_eval = None
    x_test = None
    y_test = None
    return (x_train, y_train), (x_eval, y_eval), (x_test, y_test)