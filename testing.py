import cv2
import numpy as np
import tensorflow as tf
from openpyxl import Workbook
import time

# Load the trained model
model = tf.keras.models.load_model('facial_recognition_model.h5')

# Define the label mapping
label_dict = {0: 'Akanksha Pophalkar', 1: 'Sudarshan Takate', 2: 'Sumeet Shelke'}  # Update as per your labels

# Function to preprocess frames
def preprocess_frame(frame):
    # Resize the frame to match the input size of the model (64x64)
    resized_frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match the model's input shape
    return np.expand_dims(normalized_frame, axis=0)

# Function to initialize video capture
def initialize_video_capture(url, max_attempts=10):
    attempts = 0
    cap = None
    while attempts < max_attempts:
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                return cap
            else:
                attempts += 1
                time.sleep(1)  # Wait for a moment before retrying
        except Exception as e:
            print("Error:", e)
            attempts += 1
            time.sleep(1)  # Wait for a moment before retrying
    return None

# Open a connection to the webcam
# Open a connection to the webcam
cap = cv2.VideoCapture(0)


# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an Excel workbook and select the active worksheet
wb = Workbook()
ws = wb.active
ws.append(["Student Name", "Time of Entry", "Date"])  # Header row

# Initialize a set to keep track of detected faces
detected_faces = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Preprocess the captured frame
        preprocessed_frame = preprocess_frame(frame)

        # Predict the face
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions)
        predicted_label = label_dict[predicted_class]
        confidence = np.max(predictions) * 100  # Confidence percentage of the prediction

        # Get the current time and date
        current_time = time.strftime("%H:%M:%S")
        current_date = time.strftime("%Y-%m-%d")

        # Check if the face has already been detected
        if predicted_label not in detected_faces:
            # Add the detected face to the set
            detected_faces.add(predicted_label)

            # Write the results to the Excel sheet
            ws.append([predicted_label, current_time, current_date])

        # Display the resulting frame with the predicted face and confidence
        cv2.putText(frame, '{}: {:.2f}%'.format(predicted_label, confidence), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw rectangles around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the Excel workbook
wb.save('detected_persons.xlsx')

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
