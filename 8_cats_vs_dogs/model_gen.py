#!/usr/bin/python
# Filename: model_gen.py

## import modules
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import plot_results as pltres

## define callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('acc') > 0.99):
            self.model.stop_training = True
            print ("\nStopping training as accuracy is above 99%")
callback = myCallback()

## define model
def fetch_model(train_gen, val_gen, metadata):
    model_path = "models\\model" +\
                 "_R"+str(metadata['Nrows']) +\
                 "_C"+str(metadata['Ncols']) +\
                 "_fs"+str(metadata['FILTER_SIZE'][0]) +\
                 "_ep"+str(metadata['NUM_EPOCHS']) +\
                 ".h5"
    try:
        print("\nLoading saved model")
        model = tf.keras.models.load_model(model_path)
        print("\nmodel loaded")
    except:
        print("\nModel not found. Training new model...")
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, metadata['FILTER_SIZE'], activation='relu', input_shape=(metadata['Nrows'],metadata['Ncols'],3)),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(32, metadata['FILTER_SIZE'], activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Conv2D(64, metadata['FILTER_SIZE'], activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(256, activation='relu'),
                                     tf.keras.layers.Dense(1, activation='sigmoid')])
        ## compile model
        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.summary()
        ## fit model to data - training
        history = model.fit_generator(train_gen,
                                      epochs=metadata['NUM_EPOCHS'],
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
