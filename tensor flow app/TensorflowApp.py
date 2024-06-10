#imort some module
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Printing stuff
import matplotlib.pyplot as plt

# Load a pre-defined dataset (70 of 28x28)
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data
# print(train_labels[0])
# print(train_images[0])
# plt.imshow(train_images[1], cmap='gray', vmin=0, vmax=255)
# plt.show()

# Define our neural net structure
model = keras.Sequential([

    # input is a 28x28 image ("Flatten" flattens the 28x28 into a single 784x1 input layer)
    keras.layers.Flatten(input_shape=(28,28)),

    # hidden layer is 128 deep, relu returns the value, or 0 (works good enough, much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # hidden layer is 128 deep, relu returns the value, or 0 (works good enough, much faster)
    keras.layers.Dense(units=128, activation=tf.nn.relu),

    # output is 0-10 (depending on what piece of clothing it is). return maximum
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])


# Compile our model
model.compile(optimizer=tf._optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train our model, using our training data
model.fit(train_images, train_labels, epochs=5)

# Test our model, using our testing data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(test_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()

print(test_labels[1])

# Make predictions
predictions = model.predict(test_images)

# print()
print(predictions[1])


# print out prediction
print(list(predictions[1]).index(max(predictions[1])))
