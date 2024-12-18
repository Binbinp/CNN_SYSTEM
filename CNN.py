import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the images (scaling)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert the training and test sets into TensorFlow Dataset objects
train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Batch the datasets
train = train.batch(32)
test = test.batch(32)

# Define the CNN model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))  # CIFAR-10 has 10 classes

# Compile the model using sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
hist = model.fit(train, epochs=20)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Plot the training accuracy and loss
acc = hist.history['accuracy']
loss = hist.history['loss']

epochs_range = range(20)

plt.figure(figsize=(12, 6))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')

plt.show()
