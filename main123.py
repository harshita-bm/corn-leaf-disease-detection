

import numpy as np
import os
import re
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D , Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

sn.set(font_scale=1.4)

# Define the class names and store them in the labels
class_names = [
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___healthy',
    'Corn___Northern_Leaf_Blight',
    'Corn__Common_rust'
   
]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

IMAGE_SIZE = (224, 224)

def pre_process(img_path):
    """
    Preprocess the image by reading it, resizing it, and converting it to RGB.
    """
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    return image

def load_data():
    """
    Load the images and labels from the folders.
    """
    dataset = 'A:/final test/train'
    images = []
    labels = []
    print("Loading {}".format(dataset))

    for folder in os.listdir(dataset):
        label = class_names_label[folder]

        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            img_path = os.path.join(os.path.join(dataset, folder), file)
            image = pre_process(img_path)

            images.append(image)
            labels.append(label)

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    return images, labels

# Load the data
images, labels = load_data()

# Split the data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=25)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print("Number of training examples: {}".format(n_train))
print("Number of testing examples: {}".format(n_test))
print("Each image is of size: {}".format(IMAGE_SIZE))

# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0

def display_examples(class_names, images, labels):
    """
    Display some examples of the images in the dataset.
    """
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

def display_random_image(class_names, images, labels):
    """
    Display a random image from the dataset.
    """
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()


display_examples(class_names, train_images, train_labels)

#my edits


# Define constants for image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Define the model for resizing and rescaling
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

# Define the model for data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])
# Load your image data as a dataset
# For example:
train_dataset = keras.preprocessing.image_dataset_from_directory(
    "A:/final test/train",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
)

# Get a batch of images
for images, _ in train_dataset.take(1):
    # Apply resizing and rescaling
    images = resize_and_rescale(images)

    # Apply data augmentation and get intermediate image outputs
    augmented_images = data_augmentation(images)
    intermediate_outputs = data_augmentation.predict(images)

    # Print the shape of the augmented images and intermediate outputs
    print("Augmented images shape:", augmented_images.shape)
    print("Intermediate outputs shape:", intermediate_outputs.shape)

    # Save the augmented images and intermediate outputs to disk
    for i, image in enumerate(augmented_images):
        keras.preprocessing.image.save_img(f"augmented_image_{i}.jpg", image)
    for i, output in enumerate(intermediate_outputs):
        keras.preprocessing.image.save_img(f"intermediate_output_{i}.jpg", output)
# Create the model
'''input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(nb_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])'''

'''input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False

x = layers.MaxPooling2D(pool_size=(2, 2))(base_model.output)
x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(nb_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Print a summary of the model architecture
model.summary()

# Convert labels to one-hot encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Train the model
history = model.fit(train_images, one_hot_train_labels, epochs=5, batch_size=128,
                    validation_data=(test_images, one_hot_test_labels))'''
                    
import time

# Set the batch size and image size
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Load the pre-trained ResNet50V2 model
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.ResNet50V2(
    input_shape=input_shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False

import time
import pandas as pd

# Define input shape and base model
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.ResNet50V2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
base_model.trainable = False

# Add new layers to the base model
x = layers.MaxPooling2D(pool_size=(2, 2))(base_model.output)
x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(nb_classes, activation='softmax')(x)

# Create the model and compile it
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Convert labels to one-hot encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Train the model and measure training time
start_time = time.time()
history = model.fit(train_images, one_hot_train_labels, epochs=5, batch_size=128, validation_data=(test_images, one_hot_test_labels))
end_time = time.time()
training_time = end_time - start_time

# Measure prediction speed
start_time = time.time()
model.predict(test_images[:100])
end_time = time.time()
prediction_speed = (end_time - start_time) / 100

# Print the results in table form
results = pd.DataFrame({'Training accuracy': history.history['accuracy'], 'Validation accuracy': history.history['val_accuracy']})
results.index.name = 'Epoch'
print(results)
print('Training time: {:.2f} seconds'.format(training_time))
print('Prediction speed: {:.2f} seconds per image'.format(prediction_speed))


'''#eeee


# Load a sample image
img_path = 'A:/data set/blight/image (397).JPG'
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Create a model to get intermediate outputs
intermediate_layer_models = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.MaxPooling2D) or isinstance(layer, tf.keras.layers.Dense):
        intermediate_layer_models.append(tf.keras.Model(inputs=model.input, outputs=layer.output))

# Get intermediate outputs
outputs = []
for intermediate_layer_model in intermediate_layer_models:
    intermediate_output = intermediate_layer_model.predict(x)
    outputs.append(intermediate_output)

# Display the output as images
for i, output in enumerate(outputs):
    for j in range(output.shape[-1]):
        plt.imshow(output[0, :, :, j])
        plt.title(f"Layer {i+1}, Channel {j+1}")
        plt.show()'''

version_numbers = [int(re.search(r'\d+', filename).group()) for filename in os.listdir("A:/copy project/model")]
version_numbers.append(0)
model_version = max(version_numbers, default=0) + 1
model.save("A:/copy project/model/" + str(model_version))
# Save the model
model.save("train3_model.h5")


# Plot the accuracy and loss graphs
'''plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1.5])
plt.legend(loc='upper right')'''
plt.subplot(221)
plt.plot(history.history['accuracy'], 'bo--', label="accuracy")
plt.plot(history.history['val_accuracy'], 'ro--', label="val_accuracy")
plt.title("train_acc vs val_acc")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend()

    # Plot loss function
plt.subplot(222)
plt.plot(history.history['loss'], 'bo--', label="loss")
plt.plot(history.history['val_loss'], 'ro--', label="val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")

plt.legend()
plt.show()

# Evaluate the model
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print("Test accuracy:", test_acc)

# Predict the labels of the test images
test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)

# Plot the confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 10))
sn.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

