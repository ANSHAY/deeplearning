#!/usr/bin/python
# Filename: model_gen.py
"""
Defines a fetch_model function that loads a saved model for
given parameters. It also defines and trains a model if a
saved model is not found and saves the model hence trained.
"""

## import modules
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import plot_results as pltres
import config

## define callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('acc') > 0.99):
            self.model.stop_training = True
            print ("\nStopping training as accuracy is above 99%")
callback = myCallback()

## define model
def fetch_model(train_gen, val_gen):
    model_path = "models\\model" +\
                 "_R"+str(config.Nrows) +\
                 "_C"+str(config.Nrows) +\
                 "_fs"+str(config.FILTER_SIZE[0]) +\
                 "_ep"+str(config.NUM_EPOCHS) +\
                 ".h5"
    try:
        print("\nLoading saved model")
        model = tf.keras.models.load_model(model_path)
        print("\nmodel loaded")
    except:
        print("\nModel not found. Training new model...")
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, config.FILTER_SIZE, activation='relu', input_shape=(config.Nrows,config.Ncols,3)),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(32, config.FILTER_SIZE, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(32, config.FILTER_SIZE, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(64, config.FILTER_SIZE, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(64, config.FILTER_SIZE, activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(512, activation='relu'),
                                     tf.keras.layers.Dense(1, activation='sigmoid')])
        ## compile model
        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.summary()
        ## fit model to data - training
        history = model.fit_generator(train_gen,
                                      epochs=config.NUM_EPOCHS,
                                      validation_data=val_gen,
                                      verbose=1,
                                      callbacks=[callback])
        print("\nNew model trained")
        ## save model to file
        print("\nSaving model for later use...")
        model.save(model_path)
        print("\nModel Successfully saved")
        ## plot results
        print("\nPlotting results...")
        pltres.plot_results(history)
        print("\n........................")
    return model
