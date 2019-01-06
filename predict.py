import numpy as np
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

def predict_model():
    w = keras.initializers.RandomNormal(stddev=0.001)
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, 9, activation='relu', input_shape=(None, None, 1),
                     use_bias=True, kernel_initializer=w))
    SRCNN.add(Conv2D(64, 1, activation='relu',
                     use_bias=True, kernel_initializer=w))
    SRCNN.add(Conv2D(1, 5, activation='relu',
                     use_bias=True, kernel_initializer=w))
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss=losses.mean_squared_error, metrics=['mean_squared_error'])
    return SRCNN

def predict():
    srcnn = predict_model()
    srcnn.load_weights("SRCNN_check.h5")
    IMG_NAME = "data/Test/butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "pre.jpg"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    scale = 3
    h, w, ch = img.shape
    lshape = (h//scale, w//scale)
    low = cv2.resize(img[:, :, 0], lshape, interpolation=cv2.INTER_CUBIC)
    low = cv2.resize(low, (h, w), interpolation=cv2.INTER_CUBIC)
    img[:, :, 0] = low
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(INPUT_NAME, img)

    in_data = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    in_data[0, :, :, 0] = low.astype('float32') / 255.
    pre = srcnn.predict(in_data, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:]< 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

if __name__ == "__main__":
    predict()
