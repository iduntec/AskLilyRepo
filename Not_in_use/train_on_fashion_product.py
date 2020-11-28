import tensorflow as tf

COLUMNS = 'id'
LABEL_COL = 'subCategory'
DEFAULTS = ['moshe']


def parse_csv(value):
    columns = tf.io.decode_csv(value, DEFAULTS)
    features = dict(zip(COLUMNS, columns))
    label = features.pop(LABEL_COL)
    label = tf.math.greater_equal(label, cutoff)
    return features, label


def get_input_fn(csv_path, mode=tf.estimator.ModeKeys.TRAIN, batch_size=32, cutoff=5):
    # read, parse, shuffle and batch dataset
    dataset = tf.data.TextLineDataset(csv_path).skip(1)  # skip header
    if mode == tf.estimator.ModeKeys.TRAIN:
        # shuffle and repeat
        dataset = dataset.shuffle(16 * batch_size).repeat()

    dataset = dataset.map(parse_csv, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    return dataset

    return input_fn


if __name__ == "__main__":

    csv_path_in = "C:\\Users\Idan\Desktop\AskLily files\kaggle Fashion Product Images Dataset\small_dataset\shorten.csv"
    mode = tf.estimator.ModeKeys.TRAIN
    batch_size = 8
    cutoff = 1
    data = get_input_fn(csv_path_in, mode, batch_size, cutoff)
