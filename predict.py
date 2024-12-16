import os
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import preprocess.load_models as lm
import preprocess.dataset_fx as dffx


def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  
        'label': tf.io.VarLenFeature(tf.int64),        
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.decode_png(parsed_features['image'], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    label = tf.sparse.to_dense(parsed_features['label'])
    
    return image, label


def load_tfrecord(file_path, batch_size=32):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='model_1.h5',
                        help="model to predict with")
    args = parser.parse_args()

    model_name = args.model
    model = lm.create_ViT(0.4)
    model.load_weights(os.path.join('./results', model_name))

    tfrecord_path = './datasets/test_df.tfrecord'
    dataset = load_tfrecord(tfrecord_path)

    y_true = []
    for image, label in dataset:
        y_true.append(label.numpy())
        print(image.shape)
        print(label)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = model.predict(dataset)

    y_df = pd.DataFrame({'y_real':y_true, 'y_pred':y_pred})

    y_df.to_csv('./results/y_df.csv')