import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import constants
import preprocess.load_models as md
import preprocess.load_data as load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='CNN',
                        help="CNN or ViT")
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=1e-4,
                        help="learning rate to use")
    parser.add_argument('-fp',
                        '--frozen_proportion',
                        type=float,
                        default=0.8,
                        help="backbone frozen proportion")
    parser.add_argument('-lo',
                        '--loops',
                        type=int,
                        default=1,
                        help="times to train")
    args = parser.parse_args()

    lr = args.learning_rate
    frozen_proportion = args.frozen_proportion
    model_name = args.model
    loops = args.loops

    # DATA PREPARATION
    train_df, val_df, test_df= load.load_df()
    load.write_to_tfrecord(test_df, './datasets/test_df.tfrecord')
    class_weights = load.explore_dataset(train_df)

    # MODEL PREPARATION
    if model_name == 'CNN': 
        model = md.create_CNN(frozen_proportion)
    else:
        model = md.create_ViT(frozen_proportion)
    
    callb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = 3)

    for i in range(loops):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy', 'Precision', 'AUC'],
                weighted_metrics=['accuracy', 'Precision', 'AUC'])
        
        print(f'----> Training {model_name} <-----')
        history = model.fit(train_df,
                        validation_data = val_df,
                        batch_size = constants.BATCH_SIZE,
                        class_weight=class_weights,
                        callbacks = callb,
                        epochs = constants.EPOCHS,
                        shuffle = True)
                        
        with open(f'./results/history_{i}.json', 'w') as f:
            json.dump(history.history, f)

        model.save(f'./results/model_{i}.h5')

    


