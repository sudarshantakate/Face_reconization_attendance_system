import cv2
import numpy as np
import tensorflow as tf
from openpyxl import Workbook
import time
import requests
from io import BytesIO

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

# URL for the ESP32-CAM's video stream
esp32_stream_url = "http://192.168.54.109/cam-hi.jpg"

# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an Excel workbook and select the active worksheet
wb = Workbook()
ws = wb.active
ws.append(["Student Name", "Time of Entry", "Date"])  # Header rowq

# Initialize a set to keep track of detected faces
detected_faces = set()

while True:
    try:
        # Request frame from ESP32-CAM
        response = requests.get(esp32_stream_url)
        frame_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(frame_array, -1)

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

    except Exception as e:
        print("Error:", e)
        continue

# Save the Excel workbook
wb.save('detected_persons.xlsx')

# When everything is done, close all windows
cv2.destroyAllWindows()
