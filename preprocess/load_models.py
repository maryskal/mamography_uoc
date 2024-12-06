
import tensorflow as tf
from vit_keras import vit
import tensorflow_addons as tfa
import constants 


def create_CNN(frozen_backbone_prop=0.8):
    backbone = tf.keras.applications.EfficientNetB3(weights="imagenet",
                                                    include_top=False,
                                                    input_shape=(constants.IMG_SIZE,constants.IMG_SIZE,3))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(3,3,padding="same", input_shape=(constants.IMG_SIZE,constants.IMG_SIZE,1),
                                     activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    model.add(tf.keras.layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_out_1"))
    model.add(tf.keras.layers.Dense(768, activation="elu"))
    model.add(tf.keras.layers.Dense(128, activation="elu"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_out_2"))
    model.add(tf.keras.layers.Dense(32, activation="elu"))
    model.add(tf.keras.layers.Dense(3, activation="softmax", name="fc_out"))

    fine_tune_at = int(len(backbone.layers)*frozen_backbone_prop)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    return model



def create_ViT(frozen_backbone_prop=0.8):
    backbone = vit.vit_b16(
        image_size = (constants.IMG_SIZE,constants.IMG_SIZE),
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 5)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(3,3,padding="same", input_shape=(constants.IMG_SIZE,constants.IMG_SIZE,1),
                                     activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tfa.activations.gelu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation = tfa.activations.gelu))
    model.add(tf.keras.layers.Dense(32, activation = tfa.activations.gelu))
    model.add(tf.keras.layers.Dense(3, 'softmax'))

    fine_tune_at = int(len(backbone.layers)*frozen_backbone_prop)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    return model