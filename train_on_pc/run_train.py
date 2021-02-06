from __future__ import absolute_import, division, print_function, unicode_literals

import PIL
import keras
from keras import layers

from train_on_pc.models.conv_3_4 import get_3_4_conv_model
from train_on_pc.models.pretrained_vgg16 import get_pretrained_VGG16_model
from train_on_pc.models.augmented_input_3_layers_model import get_augmented_input_3_layers_model
from train_on_pc.models.flowers_model import get_flowers_model

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
STEPS_BETWEEN_LR_DROPS = 50


def scheduler(epoch, current_lr, steps_between_lr_drops=STEPS_BETWEEN_LR_DROPS):
    if epoch % steps_between_lr_drops != 0:
        return current_lr
    else:
        return current_lr * 0.5


def plot_train_progress(history_in, epochs_in):
    acc = history_in.history['accuracy']
    val_acc = history_in.history['val_accuracy']

    loss = history_in.history['loss']
    val_loss = history_in.history['val_loss']

    epochs_range = range(epochs_in)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history.history['lr'], label='Learning Rate')
    plt.legend(loc='upper right')
    plt.title('Training Learning Rate')
    plt.show()


if __name__ == '__main__':

    directory = "/home/idan/AskLilyData/Validated/skirt_length/Part3_mixed_Google_and_dataset1/"

    data_dir = pathlib.Path(directory)
    image_count = len(list(data_dir.glob('*/*')))
    print(image_count)

    # roses = list(data_dir.glob('Off_shoulders_sleeve_top/*'))
    # pil_img = (PIL.Image.open(str(roses[0])))
    # plt.imshow(np.asarray(pil_img))
    # plt.show()

    batch_size = 16
    img_height = 224
    img_width = 224
    input_size = (img_height, img_width, 3)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        validation_split=0.1,
        subset="training",
        color_mode='rgb',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        validation_split=0.1,
        subset="validation",
        color_mode='rgb',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    import matplotlib.pyplot as plt

    class_names = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()

    # sample_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.1,
    #     labels='inferred',
    #     label_mode=None,  # gives no labels (sample)
    #     color_mode='rgb',
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    # sample_ds = sample_ds.cache().prefetch(buffer_size=AUTOTUNE)

    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # choose model
    compiled_model = get_augmented_input_3_layers_model(len(class_names), input_size)
    compiled_model = get_3_4_conv_model(len(class_names),input_size)
    compiled_model = get_pretrained_VGG16_model(len(class_names), input_size)
    compiled_model = get_flowers_model(len(class_names), input_size)

    epochs = 100
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = compiled_model.fit(normalized_train_ds, validation_data=normalized_val_ds, epochs=epochs,
                                 callbacks=[callback], verbose=1)
    plot_train_progress(history, epochs)




    # predicted = compiled_model.predict(
    #     sample_ds, batch_size=None, verbose=1, steps=None, callbacks=None, max_queue_size=10,
    #     workers=1, use_multiprocessing=False
    # )
    #
    # for image_pred_vector in predicted:
    #     pred_class = np.argmax(image_pred_vector)
    #     print(pred_class)

    print('hi')
    # matrix = tf.metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
