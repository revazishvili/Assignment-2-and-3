import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Download bird image, resize it, and save it as 'bird.png'
bird_img = cv2.imread('bird.png')  # Load the bird image
bird_resized = cv2.resize(bird_img, (32, 32))  # Resize the image to 32x32

# Convert bird image to RGB
bird_resized_rgb = cv2.cvtColor(bird_resized, cv2.COLOR_BGR2RGB)

# Save the resized RGB image
cv2.imwrite('bird.png', bird_resized_rgb)

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Add bird image to the training dataset
bird_label = np.array([2])  # Bird label is 2 according to CIFAR-10 class_names
bird_label = np.expand_dims(bird_label, axis=1)  # Expand dimensions to match train_labels
train_images = np.concatenate((train_images, [bird_resized_rgb]), axis=0)
train_labels = np.concatenate((train_labels, bird_label), axis=0)

# Verify the data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)

# Print the model class prediction on the new file
predictions = model.predict(np.expand_dims(bird_resized_rgb, axis=0))
predicted_class = np.argmax(predictions)
print("Predicted class for bird image:", class_names[predicted_class])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
