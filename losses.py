import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

# Genreator loss #
def mask_loss(GT_mask,pred_mask):
    return tf.reduce_mean(tf.abs(GT_mask - pred_mask))

def image_loss(input_image,GT_mask,pred_image,pred_mask):
    input_image, pred_image = tf.cast(input_image,tf.float32), tf.cast(pred_image,tf.float32)
    GT_mask,pred_mask = tf.cast(GT_mask,tf.float32),tf.cast(pred_mask,tf.float32)

    GT_part = tf.multiply(GT_mask,input_image)
    pred_part = tf.multiply(pred_mask,pred_image)
    return tf.reduce_mean(tf.abs(GT_part - pred_part))

# Discriminator loss #
def real_fake_loss(GT_label,pred_label):
    return tf.reduce_mean(tf.abs(GT_label - pred_label))
