import tensorflow as tf

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100

FEATURES = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}