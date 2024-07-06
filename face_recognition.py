import numpy as np
import cv2 as cv
import os
import random


face_cascade = cv.CascadeClassifier('haar_face.xml')




face_recognizer = cv.face.LBPHFaceRecognizer_create()


# Initialize variables
PATH = 'Faces'
test_images_path = 'Test'

people = []
labels = []
label_dict = {}

# Function to train the recognizer
def train_recognizer():
    global people, labels
    if len(people) > 0:
        face_recognizer.train(people, np.array(labels))

# Load existing faces and labels from the training folder
current_label = 0
for person_name in os.listdir(PATH):
    person_folder = os.path.join(PATH, person_name)
    if os.path.isdir(person_folder):
        label_dict[current_label] = person_name
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(person_folder, filename)
                face_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                people.append(face_image)
                labels.append(current_label)
        current_label += 1

# Train the recognizer with the loaded data
train_recognizer()

# Get a list of test images
test_images = [f for f in os.listdir(test_images_path) if f.endswith(".jpg")]

while True:
    # Select a random test image
    test_image_name = random.choice(test_images)
    test_image_path = os.path.join(test_images_path, test_image_name)
    frame = cv.imread(test_image_path)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)

        if confidence > 82:  # Adjust confidence threshold as needed
            # Face not recognized, add to the database
            cv.imshow('Face Recognition', frame)
            face_id = len(labels)
            face_image_path = os.path.join(PATH, f"New_Face_{face_id}.jpg")
            cv.imwrite(face_image_path, face)
            people.append(face)
            labels.append(face_id)
            label_dict[face_id] = input("write name")
            train_recognizer()
            label_text = label_dict[face_id]
        else:
            label_text = f"{label_dict[label]}, Confidence: {confidence}"
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(frame, label_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            print(label_text)
            cv.imshow('Face Recognition', frame)


    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()