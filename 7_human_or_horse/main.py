import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

## set target image size
rows = 300
cols = 300

data_gen_handler = ImageDataGenerator(rescale=1/255.0)
## load train data
data_path_train = 'D://Datasets//human_horse//train_data'
data_gen_train = data_gen_handler.flow_from_directory(
                    data_path_train,
                    batch_size=16,
                    target_size=(rows,cols),
                    class_mode='binary'
)

## load test data
data_path_val = 'D://Datasets//human_horse//val_data'
data_gen_val = data_gen_handler.flow_from_directory(
                    data_path_val,
                    batch_size=16,
                    target_size=(rows,cols),
                    class_mode='binary'
)

## define callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('acc')>0.99):
            self.model.stop_training = True
            print ("\nStopping training since accuracy reached above 99%")
callback = myCallback()

## define model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, (3,3), input_shape=(rows,cols,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

## see model summary
model.summary()

## train model
model.fit_generator(data_gen_train, verbose=1, epochs=5, callbacks=[callback], validation_data=data_gen_val)

## test model
result = model.evaluate_generator(data_gen_val, verbose=1)

## predict on test data
#predictions = model.predict_generator(data_gen_test)
print (result)
