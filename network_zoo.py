from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
img_size = (3, 32, 32)


def kerasnet(nb_classes, lastKernel=False, reg=0.01):

    trainable = not lastKernel

    model = Sequential()
    #Layer 1
    model.add(Convolution2D(32, kernel_size=(3, 3), padding='valid',
                            input_shape=img_size, trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    # Layer 2
    model.add(Convolution2D(32, kernel_size=(3, 3), trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Convolution2D(64, kernel_size=(3, 3), padding='valid',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    #Layer 4
    model.add(Convolution2D(64, kernel_size=(3, 3), trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Layer 5
    model.add(Dense(512, trainable=trainable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Last layers
    reg = None
    if lastKernel:
        reg = regularizers.l2(reg),
    model.add(Dense(nb_classes, kernel_regularizer=reg))
    model.add(Activation('softmax'))

    return model


def shallownet(nb_classes, lastKernel=False, reg=.01):
    trainable = not lastKernel
    global img_size
    model = Sequential()
    model.add(Convolution2D(64, kernel_size=(5, 5), padding='same',
                            input_shape=img_size, trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Convolution2D(64, kernel_size=(5, 5), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                           trainable=trainable))

    model.add(Flatten())
    model.add(Dense(384, trainable=trainable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.5))
    model.add(Dense(192, trainable=trainable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.5))

    # Last layer
    reg = None
    if lastKernel:
        reg = regularizers.l2(reg),
    model.add(Dense(nb_classes, activation='softmax', kernel_regularizer=reg))

    return model


def deepnet(nb_classes, lastKernel=False, reg=.01):
    trainable = not lastKernel
    global img_size
    model = Sequential()

    # Layer 1
    model.add(Convolution2D(64, kernel_size=(3, 3), padding='same',
                            input_shape=img_size, trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.3))

    # Layer 2
    model.add(Convolution2D(64, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Layer 3
    model.add(Convolution2D(128, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 4
    model.add(Convolution2D(128, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Layer 5
    model.add(Convolution2D(256, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 6
    model.add(Convolution2D(256, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 7
    model.add(Convolution2D(256, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Layer 8
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 9
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 10
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Layer 11
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 12
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.4))

    # Layer 13
    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same',
                            trainable=trainable))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    if trainable:
        model.add(Dropout(0.5))

    # Layer 14
    model.add(Dense(512, trainable=trainable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if trainable:
        model.add(Dropout(0.5))

    # Last Layer
    reg = None
    if lastKernel:
        reg = regularizers.l2(reg),
    model.add(Dense(nb_classes, activation='softmax', kernel_regularizer=reg))
    return model

