# Introduction 

"IoT-based Smart Attendance System using AI

Revolutionize traditional attendance tracking with automated, accurate, and real-time monitoring. This system leverages AI and IoT to minimize errors, optimize resources, and enhance security. Features include facial recognition, biometric authentication, and data analysis for informed decision-making."

# Project Overview

The project is divided into the following main components:

Dataset Collection: Capturing ASL gestures using a webcam and saving them as images. Model Training: Training a CNN model on the collected dataset to recognize ASL gestures. Real-time Prediction: Using the trained model to predict ASL gestures in real-time.

# Data Collection 

To collect the dataset, we use OpenCV to capture face images from a webcam and save them as images. Each individual's face is associated with a unique identifier (e.g. student ID), and images are saved in directories corresponding to each individual.

# Model Training

The collected dataset is used to train a CNN model to recognize ASL gestures. The model is trained on the images saved during the data collection phase.

