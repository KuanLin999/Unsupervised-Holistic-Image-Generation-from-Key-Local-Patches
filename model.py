from tensorflow.keras.layers import Conv2D,BatchNormalization,LeakyReLU,Conv2DTranspose,Concatenate,Flatten,Dense,Reshape,Dropout
from tensorflow import keras
import tensorflow as tf

class generator(keras.Model):
    def __init__(self):
        super(generator, self).__init__()
        ## encoder part
        self.encoder_layer_1 = Conv2D(32,kernel_size=5, strides=2, padding='same',activation='sigmoid',trainable=True)
        self.encoder_layer_2 = Conv2D(64,kernel_size=3, strides=2, padding='same',activation='sigmoid',trainable=True)
        self.encoder_layer_3 = Conv2D(128,kernel_size=3, strides=2, padding='same',activation='sigmoid',trainable=True)
        self.encoder_layer_4 = Conv2D(256,kernel_size=3, strides=2, padding='same',activation='sigmoid',trainable=True)
        self.encoder_layer_5 = Conv2D(512,kernel_size=3, strides=2, padding='same',activation='sigmoid',trainable=True)

        ## transform
        self.transfor_layer = Conv2D(512,kernel_size=3, strides=1, padding='same',activation='sigmoid',trainable=True)


        ## mask decoder
        self.mask_decoder_layer_5 = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.mask_decoder_layer_4 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.mask_decoder_layer_3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.mask_decoder_layer_2 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.mask_decoder_layer_1 = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.mask_concat5 = Concatenate()
        self.mask_concat4 = Concatenate()
        self.mask_concat3 = Concatenate()
        self.mask_concat2 = Concatenate()

        ## image decoder
        self.image_decoder_layer_5 = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.image_decoder_layer_4 = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.image_decoder_layer_3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.image_decoder_layer_2 = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.image_decoder_layer_1 = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid',trainable=True)
        self.image_concat5 = Concatenate()
        self.image_concat4 = Concatenate()
        self.image_concat3 = Concatenate()
        self.image_concat2 = Concatenate()

    def encoder(self,x):
        out = self.encoder_layer_1(x)
        out1 = self.encoder_layer_2(out)
        out2 = self.encoder_layer_3(out1)
        out3 = self.encoder_layer_4(out2)
        out4 = self.encoder_layer_5(out3)
        return out, out1, out2, out3, out4

    def mask_decoder(self,layer2,layer3,layer4,layer5,last_layer):
        # print(layer2.shape,layer3.shape,layer4.shape,layer5.shape,last_layer.shape)
        # (batch_size, 32, 32, 32) (batch_size, 16, 16, 64) (batch_size, 8, 8, 128) (batch_size, 4, 4, 256) (batch_size, 2, 2, 512)

        out = self.mask_decoder_layer_5(last_layer)
        layer5_out = self.mask_concat5([out, layer5])

        out = self.mask_decoder_layer_4(layer5_out)
        layer4_out = self.mask_concat4([out, layer4])

        out = self.mask_decoder_layer_3(layer4_out)
        layer3_out = self.mask_concat3([out, layer3])

        out = self.mask_decoder_layer_2(layer3_out)
        layer2_out = self.mask_concat2([out, layer2])

        out = self.mask_decoder_layer_1(layer2_out)

        return layer2_out,layer3_out,layer4_out,layer5_out,out

    def image_decoder(self,mask_layer2,mask_layer3,mask_layer4,mask_layer5,mask_last_layer):
        # print(mask_layer2.shape, mask_layer3.shape, mask_layer4.shape, mask_layer5.shape, mask_last_layer.shape)
        # (batch_size, 32, 32, 64) (batch_size, 16, 16, 128) (batch_size, 8, 8, 256) (batch_size, 4, 4, 512) (batch_size, 2, 2, 768)

        out = self.image_decoder_layer_5(mask_last_layer)
        layer5_out = self.image_concat5([out, mask_layer5])

        out = self.image_decoder_layer_4(layer5_out)
        layer4_out = self.image_concat4([out, mask_layer4])

        out = self.image_decoder_layer_3(layer4_out)
        layer3_out = self.image_concat3([out, mask_layer3])

        out = self.image_decoder_layer_2(layer3_out)
        layer2_out = self.image_concat2([out, mask_layer2])

        out = self.image_decoder_layer_1(layer2_out)
        return out


    def call(self, inputs,training=None, **kwargs):
        input1,input2,input3 = inputs[:,0,:,:,:],inputs[:,1,:,:,:],inputs[:,2,:,:,:]

        ### encoder simase ###
        input1_layer2, input1_layer3, input1_layer4, input1_layer5, input1_last_enc = self.encoder(input1)
        input2_layer2, input2_layer3, input2_layer4, input2_layer5, input2_last_enc = self.encoder(input2)
        input3_layer2, input3_layer3, input3_layer4, input3_layer5, input3_last_enc = self.encoder(input3)

        ### add feature map ###
        layer2 = input1_layer2 + input2_layer2 + input3_layer2
        layer3 = input1_layer3 + input2_layer3 + input3_layer3
        layer4 = input1_layer4 + input2_layer4 + input3_layer4
        layer5 = input1_layer5 + input2_layer5 + input3_layer5
        last_layer = input1_last_enc + input2_last_enc + input3_last_enc
        last_layer = self.transfor_layer(last_layer)

        ### mask decoder ###
        layer2_out,layer3_out,layer4_out,layer5_out,pred_mask = self.mask_decoder(layer2,layer3,layer4,layer5,last_layer)


        ### image decoder ###
        noise = tf.random.normal((last_layer.shape[0],last_layer.shape[1],last_layer.shape[2],256),mean=0,stddev=0.05)
        last_layer_add_noise = tf.concat([last_layer,noise],axis=-1)
        pred_image = self.image_decoder(layer2_out, layer3_out, layer4_out, layer5_out, last_layer_add_noise)

        return pred_mask,pred_image

class discriminator(keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.ac = LeakyReLU(0.2)
        self.BN = BatchNormalization()
        self.encoder_layer_1 = Conv2D(64, kernel_size=5, strides=2, padding='same',trainable=True)
        self.BN1 = BatchNormalization()
        self.ac1 = LeakyReLU(0.2)
        self.encoder_layer_2 = Conv2D(128, kernel_size=5, strides=2, padding='same',trainable=True)
        self.BN2 = BatchNormalization()
        self.ac2 = LeakyReLU(0.2)
        self.encoder_layer_3 = Conv2D(256, kernel_size=5, strides=2, padding='same',trainable=True)
        self.BN3 = BatchNormalization()
        self.ac3 = LeakyReLU(0.2)
        self.encoder_layer_4 = Conv2D(512, kernel_size=5, strides=2, padding='same',trainable=True)
        self.BN4 = BatchNormalization()
        self.ac4 = LeakyReLU(0.2)
        self.encoder_layer_5 = Conv2D(512, kernel_size=5, strides=2, padding='same',trainable=True)
        self.encoder_output_for_GAN = Conv2D(1, kernel_size=1, strides=1, padding='same',activation='sigmoid',dtype=tf.float32,trainable=True)

    def model(self,img):
        out1 = self.encoder_layer_1(img)
        out1 = self.BN(out1)
        out1 = self.ac(out1)
        out2 = self.encoder_layer_2(out1)
        out2 = self.BN1(out2)
        out2 = self.ac1(out2)
        out3 = self.encoder_layer_3(out2)
        out3 = self.BN2(out3)
        out3 = self.ac2(out3)
        out4 = self.encoder_layer_4(out3)
        out4 = self.BN3(out4)
        out4 = self.ac3(out4)
        out5 = self.encoder_layer_5(out4)
        out5 = self.BN4(out5)
        out5 = self.ac4(out5)
        real_fake = self.encoder_output_for_GAN(out5)
        return real_fake

    def call(self, inputs, training=None, **kwargs):
        predict_label_for_real_fake = self.model(inputs)
        return predict_label_for_real_fake