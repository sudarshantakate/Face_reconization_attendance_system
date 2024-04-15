import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

data_directory = 'dataset/'
images = []
labels = []
label_dict = {}  # Define label_dict at the top-level scope

# Iterate over each person's folder
for person_name in os.listdir(data_directory):
    person_folder = os.path.join(data_directory, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Assign a unique number to each person
    if person_name not in label_dict:
        label_dict[person_name] = len(label_dict)

    # Collect all images for this person
    person_images = []
    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)
        image = Image.open(image_path)
        image = image.resize((64, 64))  # Resize to uniform size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        person_images.append(image_array)

    # Split the images for this person into training and testing sets
    X_train_person, X_test_person = train_test_split(person_images, test_size=0.2, random_state=42)

    # Add the images and labels to the overall dataset
    images.extend(X_train_person)
    labels.extend([label_dict[person_name]] * len(X_train_person))

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Convert labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define and train the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_dict), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('facial_recognition_model.h5')
