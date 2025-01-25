import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter 
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import plot_model

# Defining batch specifications
batch_size = 100
img_height = 250
img_width = 250

# Loading training set
training_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

# Loading validation dataset
validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data/val',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

# Get class names
class_names = training_data.class_names

# Create the model
model = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    training_data,
    validation_data=validation_data,
    epochs=10
)

# Save the model
model.save('accident_classification_model.h5')

# Plot training results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['accuracy'], label='training accuracy')
plt.grid(True)
plt.legend()
plt.title('Training Metrics')

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='validation loss')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.grid(True)
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.savefig('training_results.png')

# Test predictions
plt.figure(figsize=(30, 30))
for images, labels in validation_data.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'Pred: {predlabel[i]} actual:{class_names[labels[i]]}')
        plt.axis('off')
        plt.grid(True)

plt.savefig('prediction_results.png') 