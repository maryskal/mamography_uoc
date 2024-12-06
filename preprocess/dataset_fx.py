import tensorflow as tf
import numpy as np
import preprocess.preprocess_fx as prepr
import constants


def pack_dataset(dataset, batch_size=constants.BATCH_SIZE):
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("Estructura final del dataset:", dataset.element_spec)
    return dataset


def create_from_tfrecord(tfrecord_files):
    
    def decode_process(proto, feature_description=constants.FEATURES):
      parsed = tf.io.parse_single_example(proto, feature_description)
      decoded_img = tf.io.decode_raw(parsed['image'], tf.uint8)
      reshaped_img = tf.reshape(decoded_img, [299, 299, 1])
      processed_img = prepr.preprocess_image(reshaped_img)
      processed_label = prepr.preprocess_label(parsed['label'])
      return processed_img, processed_label

    dataset = tf.data.TFRecordDataset(tfrecord_files)
    processed_dataset = dataset.map(decode_process)
    packed_dataset = pack_dataset(processed_dataset)
    return packed_dataset


def create_concat_df(df_url_list):
  all_datasets = [create_from_tfrecord(data) for data in df_url_list]
  combined_dataset = all_datasets[0]
  for dataset in all_datasets[1:]:
      combined_dataset = dataset.concatenate(combined_dataset)
  return combined_dataset


def create_from_array(images, labels):
  dataset = tf.data.Dataset.from_tensor_slices({
      'image': images,
      'label': labels
  })

  def process(dataset):
    processed_img = prepr.preprocess_image(dataset['image'])
    processed_label = prepr.preprocess_label(dataset['label'])
    return processed_img, processed_label

  processed_dataset = dataset.map(process)
  packed_dataset = pack_dataset(processed_dataset)
  return packed_dataset


def combine_split(validation_info):
  for key in validation_info.keys():
    validation_info[key] = np.concatenate(validation_info[key])

  index = np.random.permutation(len(validation_info['data']))
  for key in validation_info.keys():
    validation_info[key] =  validation_info[key][index]

  val_split = int(len(validation_info['data']) * 0.5)

  for key in validation_info.keys():
    validation_info[key] = [validation_info[key][:val_split], validation_info[key][val_split:]]

  return validation_info