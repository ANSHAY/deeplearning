import tensorflow as tf
import matplotlib.pyplot as plt

## load data
fmnist = tf.keras.datasets.fashion_mnist
(train_data, train_label), (test_data, test_label) = fmnist.load_data()

## show a training image
plt.imshow(train_data[0])
plt.show()

## normalizing data
train_data = train_data / 255.0
test_data = test_data / 255.0

## define a callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("loss") < 0.4):
            print ("\nStopping training on reaching 0.4 loss")
            self.model.stop_training = True
callback = myCallback()

## define network model
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(512, activation = tf.nn.relu),
                             tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_data, train_label, epochs = 5, callbacks = [callback])

## evaluate the model
predictions = model.predict(test_data)
test_loss, test_acc = model.evaluate(test_data, test_label)

print("Test accuracy: "+str(test_acc))
