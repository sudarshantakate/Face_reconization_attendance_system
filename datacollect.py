import cv2
import os

# Initialize the laptop's default webcam
cap = cv2.VideoCapture(0)

# Directory where you want to save face images
data_directory = 'dataset/'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Define face detection using Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Person identifier and maximum images to capture
person_name = input("Enter the person's name: ")
max_images = 100
count = 0

# Path for storing captured faces
person_path = os.path.join(data_directory, person_name)
if not os.path.exists(person_path):
    os.makedirs(person_path)

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces and save them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_filename = f'{person_name}_{count}.jpg'
        cv2.imwrite(os.path.join(person_path, face_filename), face)
        count += 1
        if count >= max_images:
            break

    # Display the frame
    cv2.imshow('Collecting Faces', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
