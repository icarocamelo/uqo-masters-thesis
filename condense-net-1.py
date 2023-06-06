import os
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from condensenet import condensenet

# Define the file names and URLs of the dataset
dataset_name = 'SUNRGBD'
dataset_url = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
train_images_name = 'sun_rgbd_train_images.npy'
train_labels_name = 'sun_rgbd_train_labels.npy'
test_images_name = 'sun_rgbd_test_images.npy'
test_labels_name = 'sun_rgbd_test_labels.npy'

# Define the file paths of the dataset
dataset_dir = os.path.join(os.getcwd(), dataset_name)
train_images_path = os.path.join(dataset_dir, train_images_name)
train_labels_path = os.path.join(dataset_dir, train_labels_name)
test_images_path = os.path.join(dataset_dir, test_images_name)
test_labels_path = os.path.join(dataset_dir, test_labels_name)

# Download and extract the dataset if it is not present
if not os.path.exists(dataset_dir):
    print('Downloading the SUN RGB-D dataset...')
    urllib.request.urlretrieve(dataset_url, os.path.join(os.getcwd(), 'SUNRGBD.zip'))
    with zipfile.ZipFile(os.path.join(os.getcwd(), 'SUNRGBD.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

# Load the SUN RGB-D dataset
train_images = np.load(train_images_path)
train_labels = np.load(train_labels_path)
test_images = np.load(test_images_path)
test_labels = np.load(test_labels_path)

# Load the CondenseNet model
model = condensenet(input_shape=(224, 224, 3), num_classes=1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_split=0.1, callbacks=[checkpoint])

# Evaluate the model
model.load_weights('model.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
