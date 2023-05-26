import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 32  

dataset_path = "/Users/jn/Desktop/hand-write-main/dataset 2"

#define epochs
num_epochs = 50

#shape of images
image_height = 200
image_width = 200
num_channels = 3
input_shape = (image_height, image_width, num_channels)

#model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#create an instance
data_generator = ImageDataGenerator(rescale=1./255)

#load the dataset using the ImageDataGenerator
train_data = data_generator.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

#img found
if train_data.samples == 0:
    raise ValueError("No images found in the dataset directory.")

#train the model
history = model.fit(
    train_data,
    epochs=num_epochs
)

#plot the loss and accuracy 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
