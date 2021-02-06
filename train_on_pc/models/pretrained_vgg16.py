from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, MaxPooling2D, SpatialDropout2D, UpSampling2D, concatenate, Flatten)
from tensorflow.keras.models import Model
import tensorflow as tf


def get_pretrained_VGG16_model(number_of_classes, inputs=(224, 224, 3)):
    model = VGG16(include_top=False, weights='imagenet', input_shape=inputs)

    # Freeze some of layers:
    for layer in model.layers:
        if layer is model.layers[-2]:  # <--- un-freeze 2 layers
            break
        layer.trainable = False

    # Added layers:
    # drop = Dropout(0.1)()
    global_average_pooling2d_1 = GlobalAveragePooling2D(name='global_average_pooling2d_1')(model.layers[-1].output)
    # flat = Flatten(model.layers[-1].output)
    final_activation = Activation('softmax', name='activation_output')(global_average_pooling2d_1)
    out = Dense(number_of_classes)(final_activation)

    model = Model(inputs=model.inputs, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
