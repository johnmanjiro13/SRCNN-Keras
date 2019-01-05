import numpy as np
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import dataset

batch_size = 128
epochs = 200


def model():
    # input size = 33, output size = 21
    # f=9,1,5
    w = keras.initializers.RandomNormal(stddev=0.001)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, 9, activation='relu', input_shape=(33, 33, 1),
                     use_bias=True, kernel_initializer=w))
    SRCNN.add(Conv2D(64, 1, activation='relu',
                     use_bias=True, kernel_initializer=w))
    SRCNN.add(Conv2D(1, 5, activation='relu',
                     use_bias=True, kernel_initializer=w))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss=losses.mean_squared_error, metrics=['mean_squared_error'])
    return SRCNN

def train():
    srcnn = model()
    low, label = dataset.read_dataset("./train.h5")
    val_low, val_label = dataset.read_dataset("./test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss',
                                 verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    srcnn.fit(low, label, validation_data=(val_low, val_label),
              callbacks=callbacks_list,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=0)

if __name__ == "__main__":
    train()
