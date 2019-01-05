import os
import cv2
import h5py
import numpy as np
from skimage.util import view_as_windows

DATA_PATH = "data/Train/"
TEST_PATH = "data/Test/"
patch_size = 33
label_size = 21
scale = 3
stride = 14

def dataset(_path):
    lows, labels = create_dataset(_path)
    
    # change size of label image
    pad = (patch_size - label_size)//2
    labels = [img[pad:-pad, pad:-pad, :] for img in labels]
    return lows, labels

def create_dataset(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()
    print(nums, "images")

    lows = []
    labels = []
    images = [cv2.imread(_path + name) for name in names]

    # create low image
    for img in images:
        # convert RGB to YCbCr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = img[:, :, 0]

        # create low resolution image
        h, w = img.shape
        lshape = (h//scale, w//scale)
        small = cv2.resize(img, lshape, interpolation=cv2.INTER_CUBIC)
        low = cv2.resize(small, img.shape, interpolation=cv2.INTER_CUBIC)

        img = img[:, :, np.newaxis]
        low = low[:, :, np.newaxis]
        img = img.astype('float32') / 255
        low = low.astype('float32') / 255
        labels.append(img)
        lows.append(low)

        # generate patches
        low_patches = to_patches(lows)
        label_patches = to_patches(labels)
    return low_patches, label_patches

def to_patches(images):
    patches_list = []
    for img in images:
        _, _, ch = img.shape
        patches = view_as_windows(img.copy(), (patch_size, patch_size, ch), stride)
        nrow, ncol, nz, w, h, ch = patches.shape
        patches = patches.reshape((nrow*ncol*nz, w, h, ch))
        patches_list.extend(patches)
    return patches_list

def write_hdf5(lows, labels, file_name):
    low = np.array(lows).astype('float32')
    label = np.array(labels).astype('float32')

    with h5py.File(file_name, 'w') as h:
        h.create_dataset('lows', data=low, shape=low.shape)
        h.create_dataset('labels', data=label, shape=label.shape)

def read_dataset(file):
    with h5py.File(file, 'r') as h:
        low = np.array(h.get('lows'))
        label = np.array(h.get('labels'))
        train_low = np.transpose(low, (0, 2, 3, 1))
        train_label = np.transpose(low, (0, 2, 3, 1))
        return train_low, train_label

if __name__ == "__main__":
    low, label = dataset(DATA_PATH)
    write_hdf5(low, label, "train.h5")
    low, label = dataset(TEST_PATH)
    write_hdf5(low, label, "test.h5")
