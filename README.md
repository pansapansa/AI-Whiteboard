# AI-Whiteboard
**Group Members**<br>
64110107 Phurinat Pinyomit<br>
64011699 Wasin Udomdej<br>
64011678 Tharathep Tanthacharoenkit<br>
64011512 Pansa Tuangsubsin<br>
64011373 Chatchaya Miyamoto<br>

# Task 1: Data Acquisition
We made the decision to create the AI Whiteboard. Within this project, we have categorized gestures into four distinct types: drawing, moving, and saving. We captured approximately 100 photographs for each gesture type. Furthermore, we converted all the file names to .jpg images.

# Task 2: Data Preparation
We label each image by dividing it into their corresponding sub-folder. Then we used an ImageDataGenerator to rescale the image which will be loaded with the function 'flow_from_directory().' The dataset is divided into 80% training and 20% validation. To create a bigger dataset, we could pass in parameters into the ImageDataGenerator to augment the dataset in various ways.

# Task 3: Model Training and Deployment
We utilized Visual Studio Code to train the dataset. During the training process, we observed that when the image size was set to 128x128 pixels and the number of epochs was set to 100, the training loss exhibited a promising trend, starting with a low value. Additionally, the training accuracy reached a value of one, indicating a high level of accuracy in the training process.

# Task 4: Model Inference
Begin by setting up your Jetson Nano by connecting it to a power source, monitor, keyboard, mouse, and camera. Install the necessary operating system and software including libraries for computer vision, image processing, and machine learning.
