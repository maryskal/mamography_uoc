import os
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import preprocess.load_models as lm
import preprocess.load_data as load


feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def parse_example(example_proto):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(parsed_example['image'], channels=1)     
    label = tf.sparse.to_dense(parsed_example['label'])
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='model_2.h5',
                        help="model to predict with")
    args = parser.parse_args()

    model_name = args.model
    model = lm.create_ViT(0.4)
    model.load_weights(os.path.join('./results', model_name))

    #raw_dataset = tf.data.TFRecordDataset("test_df.tfrecord")
    #parsed_dataset = raw_dataset.map(parse_example)

    #dataset = (parsed_dataset.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE))

    train_df, val_df, dataset= load.load_df()

    y_true = []
    y_pred = []

    for image_batch, label_batch in dataset:
        y_true.append(label_batch)
        predictions = model.predict(image_batch)
        y_pred.append(predictions)

    y_df = pd.DataFrame({'y_real': y_true, 'y_pred': y_pred})

    y_df.to_csv('./results/y_df.csv', index=False)