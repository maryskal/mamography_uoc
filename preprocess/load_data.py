import os
import numpy as np
import preprocess.dataset_fx as ds


def explore_dataset(dataset_to_explore, boost=np.array([1,1.3,2])):
    class_counts = np.zeros(3)
    total_samples = 0

    for _, labels in dataset_to_explore:
        batch_labels = labels.numpy()
        class_counts += np.sum(batch_labels, axis=0)
        total_samples += len(batch_labels)

    epsilon = 1e-7
    class_frequencies = class_counts / total_samples
    class_weights = 1 / (class_frequencies + epsilon)
    class_weights[class_counts == 0] = 0.0
    class_weights = class_weights * boost

    if np.sum(class_weights) > 0:
        class_weights = class_weights * len(class_counts) / np.sum(class_weights)
        min_weight = np.min(class_weights[class_weights > 0])
        class_weights = class_weights / min_weight

    class_weights_dict = dict(enumerate(class_weights))

    print("\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print("\nClass Distribution:")
    label_names = ['Normal', 'Benign', 'Malignant']

    print("\nDetailed Class Analysis:")
    print(f"{'Class':<15} {'Count':>8} {'Frequency':>12} {'Weight':>10}")
    print("-" * 45)

    for i, (count, freq, weight) in enumerate(zip(class_counts,
                                                class_frequencies,
                                                class_weights)):
      print(f"{label_names[i]:<15} {int(count):8d} {freq:12.4f} {weight:10.4f}")

    print("\nWeight Statistics:")
    print(f"Mean weight: {np.mean(class_weights):.4f}")
    print(f"Max/Min ratio: {np.max(class_weights)/np.min(class_weights[class_weights>0]):.4f}")

    return class_weights_dict


def load_df():
    train_files = [f'./datasets/training10_{i}/training10_{i}.tfrecords' for i in range(5)]

    # Train
    train_df = ds.create_concat_df(train_files)

    # Validation
    test_data = np.load("./datasets/test10_data/test10_data.npy")
    test_label = np.load("./datasets/test10_labels.npy")
    cv_data = np.load("./datasets/cv10_data/cv10_data.npy")
    cv_label = np.load("./datasets/cv10_labels.npy")

    validation_info = {'data':[test_data, cv_data], 'label':[test_label, cv_label]}
    validation_info = ds.combine_split(validation_info)
    val_df = ds.create_from_array(validation_info['data'][0], validation_info['label'][0])
    test_df = ds.create_from_array(validation_info['data'][1], validation_info['label'][1])

    return train_df, val_df, test_df


def write_to_tfrecord(dataset, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for example in dataset:
            image_data = tf.cast(example[0] * 255, tf.uint8)
            png_encoded = tf.io.encode_png(image_data)

            labels = example[1].numpy()
            labels_argmax = np.argmax(labels, axis=1)

            features = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_encoded.numpy().tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels_argmax.tolist()))
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())