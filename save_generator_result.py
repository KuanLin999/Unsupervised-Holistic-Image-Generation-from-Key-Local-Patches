import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import model
import losses
import prepare_data
import datetime
import time
import os
import cv2

class training_Double_Unet():
    def __init__(self):
        self.G = model.generator()
        self.G.build(input_shape=(1 ,3, 64, 64, 1))
        self.G.summary()

        self.D = model.discriminator()
        self.D.build((1,64,64,1))
        self.D.summary()

        self.G_optimizer = Adam(learning_rate=1e-4,decay=1e-7)
        self.D_optimizer = Adam(learning_rate=1e-4,decay=1e-7)

        self.checkpoint = tf.train.Checkpoint(g_optimizer=self.G_optimizer, d_optimizer=self.D_optimizer,
                                              generator=self.G, discriminator=self.D)

        root = '/model_checkpoint/training_checkpoints_20211110-123055'
        self.checkpoint.restore(tf.train.latest_checkpoint(root))

    @tf.function
    def get_result(self,patchs):
        pred_mask, pred_image = self.G(patchs)
        return pred_mask, pred_image

    def save_fig(self,save_root,save_name,data):
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        s = save_root + save_name
        cv2.imwrite(s, data)

    def save_100_images(self):
        x_train, x_test, y_train, y_test = prepare_data.load_mnist()
        save_file_root = '/Result'
        for idx in range(100):
            print(idx)
            image, label = x_train[idx], y_train[idx]
            image, label = np.array([image]),np.array([label])
            patch_image, GT_mask, GT_image, GT_shuffle = prepare_data.prepare_batch_data(image,label,False)
            patch_image, GT_mask, GT_image = np.array(patch_image), np.array(GT_mask), np.array(GT_image)
            pred_mask, pred_image = self.get_result(patch_image)
            pred_mask, pred_image = pred_mask.numpy(), pred_image.numpy()
            # print(GT_mask.shape,GT_image.shape,pred_mask.shape,pred_image.shape)

            save_class = 'GT_mask'
            save_root = save_file_root + '/' + save_class +'/'
            save_name = str(idx) + '.png'
            self.save_fig(save_root, save_name, GT_mask[0] * 255)

            save_class = 'GT_image'
            save_root = save_file_root + '/' + save_class + '/'
            save_name = str(idx) + '.png'
            self.save_fig(save_root, save_name, GT_image[0] * 255)

            save_class = 'pred_mask'
            save_root = save_file_root + '/' + save_class + '/'
            save_name = str(idx) + '.png'
            self.save_fig(save_root, save_name, pred_mask[0] * 255)

            save_class = 'pred_image'
            save_root = save_file_root + '/' + save_class + '/'
            save_name = str(idx) + '.png'
            self.save_fig(save_root, save_name, pred_image[0] * 255)

save_result = training_Double_Unet()
save_result.save_100_images()