import keras
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, MaxPooling2D, SpatialDropout2D, UpSampling2D, concatenate, Flatten)
from tensorflow.python.keras.optimizer_v1 import SGD


def get_augmented_input_3_layers_model(number_of_classes, input_shape=(224, 224, 3)):
    # data_augmentation = keras.Sequential(
    #     [
    #         preprocessing.RandomFlip("horizontal"),
    #         preprocessing.RandomRotation(0.1),
    #         preprocessing.RandomZoom(0.1),
    #     ]
    # )

    inputs = Input(input_shape)
    # augment_layer = data_augmentation(inputs)

    conv_1 = Conv2D(round(96), kernel_size=(14, 14), strides=(3, 3), activation="relu", name='conv_1')(inputs)
    batch_norm_1 = BatchNormalization(name='batch_norm_1')(conv_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(batch_norm_1)

    # drop_2 = Dropout(0.3)(maxpool_1)
    conv_2 = Conv2D(round(64), kernel_size=(7, 7), strides=(1, 1), activation="relu", name='conv_2')(maxpool_1)
    batch_norm_2 = BatchNormalization(name='batch_norm_2')(conv_2)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(batch_norm_2)

    # drop_3 = Dropout(0.3)(maxpool_2)
    conv_3 = Conv2D(round(128), kernel_size=(3, 3), strides=(1, 1), activation="relu", name='conv_3')(maxpool_2)
    batch_norm_3 = BatchNormalization(name='batch_norm_3')(conv_3)

    flat = Flatten()(batch_norm_3)
    out_dense = Dense(128)(flat)
    out = Dense(number_of_classes)(out_dense)

    out_model = Model(inputs=inputs, outputs=out)

    # epochs =50
    # learning_rate = 0.1
    # decay_rate = learning_rate / epochs
    # momentum = 0.8
    # sgd = tf.keras.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # out_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    out_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    out_model.summary()

    return out_model


if __name__ == '__main__':
    output_classes = 6
    model = get_augmented_input_3_layers_model(output_classes)
