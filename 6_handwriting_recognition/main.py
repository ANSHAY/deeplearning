import tensorflow as tf

## load data for handwriting digits
mnist = tf.keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

## normalize dataset
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data.reshape(60000, 28, 28, 1)
test_data = test_data.reshape(10000, 28, 28, 1)

## define callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.99):
            print ("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
callback = myCallback()

## define model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                             tf.keras.layers.MaxPooling2D(2, 2),
                             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(2, 2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

## train the model
model.fit(train_data, train_labels, epochs=10, callbacks=[callback])

## predict on test dataset
predictions = model.predict(test_data)

## evaluate model
loss, acc = model.evaluate(test_data, test_labels)

## print loss and accuracy
print("\nLoss on test data:" + str(loss))
print("\nAccuracy on test data:" + str(acc))
