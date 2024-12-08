import tensorflow as tf
import constants
import numpy as np
import pandas as pd
from mango import Tuner, scheduler
from scipy.stats import uniform
import json
import preprocess.load_models as md
import preprocess.load_data as load


param_space = dict(model_name=['CNN', 'ViT'],
                    frozen_prop = uniform(0,1),
                    lr= uniform(1e-5, 1e-3))

conf_dict = dict(num_iteration=50)


def train(model_name, frozen_prop, lr, train_df, val_df, class_weights):
    if model_name == 'CNN':
      model = md.create_CNN(frozen_prop)
    else:
      model = md.create_ViT(frozen_prop)

    # Compilado
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy', 'Precision', 'AUC'],
              weighted_metrics=['accuracy', 'Precision', 'AUC'])

    # CALLBACK
    callb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = 3)
    
    # TRAIN
    history = model.fit(train_df,
                    validation_data = val_df,
                    batch_size = constants.BATCH_SIZE,
                    class_weight=class_weights,
                    callbacks = callb,
                    epochs = constants.EPOCHS,
                    shuffle = True,
                    verbose=1)
                    
    new_row = {
        'model_name': model_name,
        'frozen_proportion': frozen_prop,
        'learning_rate': lr,
        'val_auc': history.history['val_auc'],
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss'],
        'val_precision': history.history['val_precision']
        }
        
    results_df = pd.read_csv('./hyperparameter_results.csv')
    results_df.loc[len(results_df)] = new_row
    results_df.to_csv('./hyperparameter_results.csv', index=False)

    return max(history.history['val_auc'])


if __name__ == "__main__":

    train_df, val_df, _ = load.load_df()
    train_df = train_df.take(100)
    class_weights = load.explore_dataset(train_df)
    
    # Create dataframe
    columns = ['model_name', 'frozen_proportion', 'learning_rate', 
               'val_auc', 'val_accuracy', 'val_loss', 'val_precision']
    results_df = pd.DataFrame(columns=columns)
    results_df.to_csv('./hyperparameter_results.csv')
    
    @scheduler.serial
    def objective(**params):
        print('--------NEW COMBINATION--------')
        print(params)
        results = []
        for x in range(2):
            results.append(train(**params, 
                                 train_df=train_df, 
                                 val_df=val_df, 
                                 class_weights=class_weights))
            print('results {}: {}'.format(x, results[x]))
        print('FINAL RESULTS {}'.format(np.mean(results)))
        return np.mean(results)

    tuner = Tuner(param_space, objective, conf_dict)
    results = tuner.maximize()

    for k, v in results.items():
        if type(v) is np.ndarray:
            results[k] = list(v)
    
    print('best parameters:', results['best_params'])
    print('best f1score:', results['best_objective'])

    with open('./hyperparameter_results.json', 'w') as j:
            json.dump(results, j)
    