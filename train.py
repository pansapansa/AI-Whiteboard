import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32  # Set your desired batch size

# Define the dataset path
dataset_path = "/Users/jn/Desktop/hand-write-main/dataset 2"

# Define the number of epochs
num_epochs = 100

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

# Train the model
model.fit(
    train_data,
    epochs=num_epochs
)
