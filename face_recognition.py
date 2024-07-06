import numpy as np
import cv2 as cv
import os
import random


face_cascade = cv.CascadeClassifier('haar_face.xml')
face_id = 0
people = []
label = []

def train_recognizer():
    global people, label
    if len(people) > 0:
        face_recognizer.train(people, np.array(label))

# TO BE FILLED
PATH = 'Faces'
test_images_path = ''

for filename in os.listdir(PATH):
    if filename.endswith(".jpg"):
        image_path = os.path.join(PATH, filename)
        face_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        people.append(face_image)
        label.append(int(filename.split("_")[0]))

train_recognizer()

face_recognizer = cv.face.LBPHFaceRecognizer_create()

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

        if confidence < 30:  # Adjust confidence threshold as needed
            # Face not recognized, add to the database
            face_id += 1
            face_image_path = os.path.join(PATH, f"{face_id}_{confidence}.jpg")
            cv.imwrite(face_image_path, face)
            people.append(face)
            label.append(face_id)
            label_text = f"New Face {face_id}"
        else:
            label_text = f"Face {label}, Confidence: {confidence}"

        train_recognizer()

        # Draw a rectangle around the face and label it
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(frame, label_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv.imshow('Face Recognition', frame)
    cv.waitKey(0)
