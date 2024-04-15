import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_directory = 'dataset/'
images = []
labels = []

# Dictionary to hold label encoding
label_dict = {}

# Iterate over each person's folder
for person_name in os.listdir(data_directory):
    person_folder = os.path.join(data_directory, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Assign a unique number to each person
    if person_name not in label_dict:
        label_dict[person_name] = len(label_dict)

    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)
        image = Image.open(image_path)
        image = image.resize((64, 64))  # Resize to uniform size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        images.append(image_array)
        labels.append(label_dict[person_name])

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Convert labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
