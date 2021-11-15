import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype('uint8'), x_test.astype('uint8')
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=456)
    return x_train, x_test, y_train, y_test


def check_bbs(x,y,w,h):
    ## x
    if x+w >= 64:
        x2 = 63
    else:
        x2 = x+w

    ## y
    if y+h >= 64:
        y2 = 63
    else:
        y2 = y+h
    return x,x2,y,y2

def generator_image_and_mask(image):
    patchs = []
    mask = np.zeros(image.shape)
    for _ in range(3):
        x, y, w, h = random.randint(5,60),random.randint(5,60),25,25
        x1,x2,y1,y2 = check_bbs(x,y,w,h)
        patch_image = image[x1:x2,y1:y2,:]
        patch_image = np.resize(patch_image,(64,64,1))
        mask[x1:x2,y1:y2,:] = 1
        patchs.append(patch_image)
    return np.array(patchs),mask

def transform(image):
    img = cv2.resize(image, (64,64))
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    return img

def prepare_batch_data(images,image_label,choose_shuffle=True):
    '''
    ### if batch size = 64
    :param images: images is mnist data, shape = (64,64,64,1)
    :param image_label: image label, shape = (64,)
    :return: patch image(64*3), mask(64*1), GT_image(64*1), shuffle_GT(64*1)
    '''

    patch_image,GT_mask,GT_image,GT_shuffle = [],[],[],[]
    for idx in range(images.shape[0]):
        image = images[idx,:,:]
        image = transform(image)
        GT_image.append(image)
        patchs,mask = generator_image_and_mask(image)
        patch_image.append(patchs), GT_mask.append(mask)

    if choose_shuffle:
        for idx in range(images.shape[0]):
            target_label = image_label[idx]
            while True:
                choose_idx = random.randint(0,images.shape[0]-1)
                choose_label = image_label[choose_idx]
                if target_label != choose_label:
                    image = images[choose_idx, :, :]
                    image = transform(image)
                    GT_shuffle.append(image)
                    break

    ### patch_image,GT_mask,GT_image,shuffle_GT
    ### (batch_size,3,64,64,1), (batch_size,64,64,1), (batch_size,64,64,1), (batch_size,64,64,1)
    return patch_image,GT_mask,GT_image,GT_shuffle