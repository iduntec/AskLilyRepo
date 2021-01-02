from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, MaxPooling2D, SpatialDropout2D, UpSampling2D, concatenate, Flatten)

AUTOTUNE = tf.data.experimental.AUTOTUNE
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pathlib


def get_3_layers_model(net_model_name):
    inputs = Input((224, 224, 3))

    conv_1 = Conv2D(round(96), kernel_size=(7, 7), strides=(3, 3), activation="relu", name='conv_1')(inputs)
    batch_norm_1 = BatchNormalization(name='batch_norm_1')(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(batch_norm_1)

    drop_2=Dropout(0.5)(maxpool_1)
    conv_2 = Conv2D(round(192), kernel_size=(5, 5), strides=(1, 1), activation="relu", name='conv_2')(drop_2)
    batch_norm_2 = BatchNormalization(name='batch_norm_2')(conv_2)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(batch_norm_2)

    drop_3 = Dropout(0.5)(maxpool_2)
    conv_3 = Conv2D(round(256), kernel_size=(3, 3), strides=(1, 1), activation="relu", name='conv_3')(drop_3)
    batch_norm_3 = BatchNormalization(name='batch_norm_3')(conv_3)

    global_average_pooling2d_1 = GlobalAveragePooling2D(name='global_average_pooling2d_1')(batch_norm_3)
    out = Activation('softmax', name='activation_output')(global_average_pooling2d_1)

    out_model = Model(inputs=inputs, outputs=out, name=net_model_name)
    out_model.summary()
    return out_model


if __name__ == '__main__':

    directory = "/home/idan/AskLilyData/SkirtLength/"
    data_dir = pathlib.Path(directory)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    batch_size = 36
    img_height = 224
    img_width = 224
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    model = get_3_layers_model('my_model')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    epochs = 300
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
