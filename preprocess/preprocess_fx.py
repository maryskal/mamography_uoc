import os
import cv2
import tensorflow as tf
import numpy as np

IMG_SIZE = 256

def normalize(image):
    image = tf.cast(image, tf.float32)
    mean = tf.math.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / std
    return image


def apply_clahe(image,tfrecord=True):
    if tfrecord:
      image = image.numpy().squeeze()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image.astype(np.uint8))
    image_clahe = np.expand_dims(image_clahe, axis=-1)
    image_clahe = tf.convert_to_tensor(image_clahe, dtype=tf.float32)
    return image_clahe


def tf_apply_clahe(image, img_size=IMG_SIZE):
    image = tf.py_function(
        func=apply_clahe,
        inp=[image],
        Tout=tf.float32
    )
    image.set_shape([img_size, img_size, 1])
    return image


def preprocess_image(image, tfrecord=True, img_size=IMG_SIZE):
    if tfrecord:
      image = tf_apply_clahe(image)
    else:
      image = apply_clahe(image,False)
    image = tf.image.resize(image, [img_size, img_size])
    image = normalize(image)
    return image


def preprocess_label(label):
    grouped_label = tf.where(label == 0, 0, 
                         tf.where(label <= 2, 1, 2))
    processed_label = tf.one_hot(grouped_label, 3, dtype=tf.float32)
    return processed_label