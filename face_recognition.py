import numpy as np
import cv2 as cv
import os


haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = []
label = []

def train_recognizer():
    global people, labels
    if len(people) > 0:
        face_recognizer.train(people, np.array(labels))

# TO BE FILLED
PATH = ''

for filename in os.listdir(PATH):
    if filename.endswith(".jpg"):
        image_path = os.path.join(PATH, filename)
        face_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        people.append(face_image)
        label.append(int(filename.split("_")[0]))

train_recognizer()

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'F:\codes\opencv\Faces\val\elton_john\1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)