import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import model
import losses
import prepare_data
import datetime
import time

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

        self.batch_size = 128

    @tf.function
    def G_train_step(self,patchs,GT_mask,GT_image):
        patchs = tf.cast(patchs, tf.float32)
        GT_mask = tf.cast(GT_mask, tf.float32)
        GT_image = tf.cast(GT_image, tf.float32)

        with tf.GradientTape() as tape:
            pred_mask,pred_image = self.G(patchs,training = True)
            mask_loss = losses.mask_loss(GT_mask,pred_mask)
            image_loss = losses.image_loss(GT_image,GT_mask,pred_mask,pred_image)
            pred_label = self.D(pred_image,trainin=False)
            real_label = tf.ones(pred_label.shape)
            GAN_G_losses = losses.real_fake_loss(real_label,pred_label)
            total_loss = mask_loss + image_loss + GAN_G_losses
        grad = tape.gradient(total_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad, self.G.trainable_variables))

    @tf.function
    def D_train_step(self,patchs,GT_mask,GT_image,GT_shuffle):
        patchs = tf.cast(patchs, tf.float32)
        GT_mask = tf.cast(GT_mask, tf.float32)
        GT_image = tf.cast(GT_image, tf.float32)
        GT_shuffle = tf.cast(GT_shuffle, tf.float32)
        real_label = tf.ones((GT_image.shape[0],2,2,1))
        fake_label = tf.zeros((GT_image.shape[0],2,2,1))

        with tf.GradientTape() as tape:
            pred_mask,pred_image = self.G(patchs,training = False)
            systhesis1 = tf.multiply(GT_mask, GT_image) + tf.multiply(1 - GT_mask, GT_shuffle)
            systhesis2 = tf.multiply(GT_mask, GT_shuffle) + tf.multiply(1 - GT_mask, GT_image)
            systhesis3 = tf.multiply(GT_mask, GT_image) + tf.multiply(1 - GT_mask, pred_image)
            systhesis4 = tf.multiply(GT_mask, pred_image) + tf.multiply(1 - GT_mask, GT_image)

            images = tf.concat([GT_image,pred_image,systhesis1,systhesis2,systhesis3,systhesis4],axis=0)
            pred_label = self.D(images,trainin=True)
            GT_label = tf.concat([real_label,fake_label,fake_label,fake_label,fake_label,fake_label],axis=0)
            GAN_D_losses = losses.real_fake_loss(GT_label,pred_label)
            total_loss = GAN_D_losses
        grad = tape.gradient(total_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grad, self.D.trainable_variables))

    @tf.function
    def test_step(self,patchs,GT_mask,GT_image,GT_shuffle):
        patchs = tf.cast(patchs, tf.float32)
        GT_mask = tf.cast(GT_mask, tf.float32)
        GT_image = tf.cast(GT_image, tf.float32)
        GT_shuffle = tf.cast(GT_shuffle, tf.float32)
        real_label = tf.ones((GT_image.shape[0],2,2,1))
        fake_label = tf.zeros((GT_image.shape[0],2,2,1))

        ### G part
        pred_mask, pred_image = self.G(patchs, training=False)
        mask_loss = losses.mask_loss(GT_mask, pred_mask)
        image_loss = losses.image_loss(GT_image, GT_mask, pred_mask, pred_image)
        pred_label = self.D(pred_image, trainin=False)
        GAN_G_losses = losses.real_fake_loss(real_label, pred_label)
        G_total_loss = mask_loss + image_loss + GAN_G_losses

        ### D part
        pred_mask, pred_image = self.G(patchs, training=False)
        systhesis1 = tf.multiply(GT_mask, GT_image) + tf.multiply(1 - GT_mask, GT_shuffle)
        systhesis2 = tf.multiply(GT_mask, GT_shuffle) + tf.multiply(1 - GT_mask, GT_image)
        systhesis3 = tf.multiply(GT_mask, GT_image) + tf.multiply(1 - GT_mask, pred_image)
        systhesis4 = tf.multiply(GT_mask, pred_image) + tf.multiply(1 - GT_mask, GT_image)
        images = tf.concat([GT_image, pred_image, systhesis1, systhesis2, systhesis3, systhesis4], axis=0)
        pred_label = self.D(images, trainin=False)
        GT_label = tf.concat([real_label, fake_label, fake_label, fake_label, fake_label, fake_label],axis=0)
        GAN_D_losses = losses.real_fake_loss(GT_label, pred_label)

        return mask_loss,image_loss,GAN_G_losses,G_total_loss,GAN_D_losses

    def plot_training_stage(self,train_writer,val_writer,train_data,val_data,loss_name,e,train_description=None,val_description=None):
        with train_writer.as_default():
            tf.summary.scalar(loss_name, np.mean(train_data), step=e + 1,description=train_description)
        with val_writer.as_default():
            tf.summary.scalar(loss_name, np.mean(val_data), step=e + 1,description=val_description)

    def update(self,G_update_times,D_update_times):
        images, _, labels, _ = prepare_data.load_mnist()
        for _ in range(D_update_times):
            for batch in range(int(len(images)/self.batch_size)):
                range_min = batch * self.batch_size
                range_max = (batch + 1) * self.batch_size
                if batch * self.batch_size > len(images):
                    range_max = len(images)
                index = list(range(range_min, range_max))
                train_image, train_label = images[index], labels[index]
                patch_image, GT_mask, GT_image, GT_shuffle = prepare_data.prepare_batch_data(train_image,train_label)
                self.D_train_step(patch_image, GT_mask, GT_image, GT_shuffle)

        for _ in range(G_update_times):
            for batch in range(int(len(images) / self.batch_size)):
                range_min = batch * self.batch_size
                range_max = (batch + 1) * self.batch_size
                if batch * self.batch_size > len(images):
                    range_max = len(images)
                index = list(range(range_min, range_max))
                train_image, train_label = images[index], labels[index]
                patch_image, GT_mask, GT_image, GT_shuffle = prepare_data.prepare_batch_data(train_image, train_label)
                self.G_train_step(patch_image, GT_mask, GT_image)

    def caluate_loss_per_epoch(self,train_or_test):
        if train_or_test == 'train':
            images, _, labels, _ = prepare_data.load_mnist()
        else:
            _, images, _, labels = prepare_data.load_mnist()

        loss_list = [[] for _ in range(5)]
        for batch in range(int(len(images) / self.batch_size)):
            range_min = batch * self.batch_size
            range_max = (batch + 1) * self.batch_size
            if batch * self.batch_size > len(images):
                range_max = len(images)
            index = list(range(range_min, range_max))
            train_image, train_label = images[index], labels[index]
            # print(train_image.shape,train_label.shape)
            patch_image, GT_mask, GT_image, GT_shuffle = prepare_data.prepare_batch_data(train_image, train_label)
            mask_loss,image_loss,GAN_G_losses,G_total_loss,GAN_D_losses = self.test_step(patch_image, GT_mask, GT_image, GT_shuffle)
            idx = 0
            for loss in [mask_loss,image_loss,GAN_G_losses,G_total_loss,GAN_D_losses]:
                loss = loss.numpy()
                loss_list[idx].append(loss)
                idx += 1
        return loss_list

    def main(self):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = 'model_checkpoint/tb_logs_'
        train_log_dir = path + current_time + '/train'
        val_log_dir = path + current_time + '/val'
        train_writer = tf.summary.create_file_writer(train_log_dir)
        val_writer = tf.summary.create_file_writer(val_log_dir)
        checkpoint_dir = f'model_checkpoint/training_checkpoints_{current_time}/model'
        min_test_G_total_loss = 1000000000000000

        loss_name = ['mask loss(epoch)','image loss(epoch)','GAN G loss(epoch)','total G loss(epoch)','GAN D loss(epoch)']
        for epoch in range(101):
            new_epoch_start_time = time.time()
            if epoch != 0:
                self.update(G_update_times=3,D_update_times=2)

            train_losses = self.caluate_loss_per_epoch('train')
            test_losses = self.caluate_loss_per_epoch('test')

            end_now_epoch_time = time.time()
            print('==============', 'epoch:', epoch, '==============')
            print('spant time:', end_now_epoch_time - new_epoch_start_time)
            for idx in range(len(loss_name)):
                name = loss_name[idx]
                train_loss, test_loss = train_losses[idx],test_losses[idx]
                train_loss, test_loss = np.mean(train_loss),np.mean(test_loss)
                self.plot_training_stage(train_writer,val_writer,train_losses,test_loss,name,epoch)
                print('train %s:'%name,train_loss,'test %s:'%name,test_loss)

            if np.mean(test_losses[-2]) < min_test_G_total_loss:
                min_test_G_total_loss = np.mean(test_losses[-2])
                if epoch != 0:
                    self.checkpoint.save(checkpoint_dir)
                    print('save model')


training_model = training_Double_Unet()
training_model.main()