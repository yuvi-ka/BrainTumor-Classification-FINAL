import cv2
import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Set encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Specify the image directory
image_directory = r'C:\Users\hp\OneDrive\Desktop\BrainTumor Classification DL\BrainTumor Classification DL\datasets'

# Check if the specified directory exists
if not os.path.exists(image_directory):
    print(f"Error: The directory '{image_directory}' does not exist.")
    exit()

# Check if 'no' and 'yes' folders exist inside the dataset folder
no_path = os.path.join(image_directory, 'no')
yes_path = os.path.join(image_directory, 'yes')

if not os.path.exists(no_path):
    print(f"Error: The 'no' folder does not exist inside '{image_directory}'.")
    exit()

if not os.path.exists(yes_path):
    print(f"Error: The 'yes' folder does not exist inside '{image_directory}'.")
    exit()

# List files in 'no' and 'yes' folders
no_tumor_images = os.listdir(no_path)
yes_tumor_images = os.listdir(yes_path)

dataset = []
label = []

INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image_path = os.path.join(no_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)
        else:
            print(f"Warning: Unable to read image '{image_path}'")

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image_path = os.path.join(yes_path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)
        else:
            print(f"Warning: Unable to read image '{image_path}'")

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

try:
    # Model Building
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, to_categorical(y_train, num_classes=2),
                        batch_size=16,
                        verbose=1,  # Set verbose to 1
                        epochs=10,
                        validation_data=(x_test, to_categorical(y_test, num_classes=2)),
                        shuffle=False)

    # Save the model
    model.save('BrainTumor10EpochsCategorical.h5')

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
