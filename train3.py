import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32  # Set your desired batch size

# Define the dataset path
dataset_path = "/Users/jn/Desktop/hand-write-main/dataset 2"

# Define the number of epochs
num_epochs = 80

# Define the input shape of your images
image_height = 200
image_width = 200
num_channels = 3
input_shape = (image_height, image_width, num_channels)

# Define your model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    # Add more layers as needed
    # ...
    layers.Flatten(),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create an instance of the ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255)

# Load the dataset using the ImageDataGenerator
train_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Check if any images are found
if train_data.samples == 0:
    raise ValueError("No images found in the dataset directory.")

# Calculate precision and recall during training
class_names = train_data.class_indices.keys()
num_classes = len(class_names)
precision_values = []
recall_values = []

for epoch in range(num_epochs):
    history = model.fit(
        train_data,
        epochs=1,
        verbose=0
    )
    y_true = []
    y_pred = []
    for batch in train_data:
        batch_images, batch_labels = batch
        batch_pred = model.predict(batch_images)
        y_true.extend(np.argmax(batch_labels, axis=1))
        y_pred.extend(np.argmax(batch_pred, axis=1))
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    precision_values.append(precision.result().numpy())
    recall_values.append(recall.result().numpy())

# Plot the precision and recall curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), precision_values, label='Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), recall_values, label='Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

# Print the final precision and recall values
final_precision = precision_values[-1]
final_recall = recall_values[-1]
plt.text(num_epochs - 1, final_precision, f'Precision: {final_precision:.4f}', ha='right', va='center')
plt.text(num_epochs - 1, final_recall, f'Recall: {final_recall:.4f}', ha='right', va='center')

plt.tight_layout()
plt.show()
