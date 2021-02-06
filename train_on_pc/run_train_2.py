from __future__ import print_function

import pathlib
import numpy as np
import tensorflow as tf
import os


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0 / 127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y


# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).
if __name__ == '__main__':
    # Parameters
    # Dataset Parameters - CHANGE HERE
    MODE = 'folder'  # or 'file', if you choose a plain text file (see above).
    DATASET_PATH = "/home/idan/AskLilyData/Validated/Sleeve_length/"  # the dataset file or root folder path.
    data_dir = pathlib.Path(DATASET_PATH)
    image_count = len(list(data_dir.glob('*/*')))
    class_names = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    # Image Parameters
    N_CLASSES = len(class_names)  # CHANGE HERE, total number of classes
    IMG_HEIGHT = 224  # CHANGE HERE, the image height to be resized to
    IMG_WIDTH = 224  # CHANGE HERE, the image width to be resized to
    CHANNELS = 3  # The 3 color channels, change to 1 if grayscale

    print(image_count)

    learning_rate = 0.001
    num_steps = 10000
    batch_size = 128
    display_step = 100

    # Network Parameters
    dropout = 0.75  # Dropout, probability to keep units

    # Build the data input
    X, Y = read_images(data_dir, MODE, batch_size)

    print('hi')
