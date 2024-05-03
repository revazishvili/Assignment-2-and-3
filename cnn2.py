import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, preprocessing
from PIL import Image
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the bird image and resize it to 32x32
bird_image = Image.open('bird.png').convert('RGB').resize((32, 32))

# Convert the image to a numpy array
bird_image_array = np.array(bird_image) / 255.0

# Reshape the bird image array to match the shape of the CIFAR-10 images
bird_image_array = np.expand_dims(bird_image_array, axis=0)

# Repeat the bird image to match the number of training samples
bird_image_array = np.repeat(bird_image_array, len(train_images), axis=0)

# Create a new dataset with the bird image added
new_train_images = np.concatenate((train_images, bird_image_array), axis=0)
new_train_labels = np.concatenate((train_labels, [[2]] * len(train_labels)), axis=0)  # 2 is the label for 'bird'

# Data augmentation
datagen = preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

datagen.fit(new_train_images)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(new_train_images, new_train_labels, batch_size=32),
                    epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

# Predict the class of the bird image
bird_prediction = model.predict(bird_image_array)
predicted_class = np.argmax(bird_prediction)
print(f"Predicted class for the bird image: {class_names[predicted_class]}")

# Write the test accuracy to a file
with open('accuracy2.txt', 'w') as f:
    f.write(f'Test accuracy: {test_acc:.4f}\n')
    f.write(f'Training accuracy: {history.history["accuracy"][-1]:.4f}\n')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
