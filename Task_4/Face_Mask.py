from keras.models import model_from_json
import cv2
import numpy as np
import tensorflow as tf

face = ["Mask", "No Mask"]
# Loading the Model
with open("model.json", 'r') as f:
    loaded_model = f.read()
    f.close()

model = model_from_json(loaded_model)
model.load_weights("model.h5")

faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    faces = faceClassifier.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        faceImg = frame[y:y + w, x:x + w]
        imgGray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
        imgResized = cv2.resize(imgGray, (256, 256))
        imgReshaped = imgResized.reshape(256, 256, 1)
        imgReshaped = imgReshaped / 255.0
        finalImg = np.array([imgReshaped])
        pred = int(model.predict(finalImg)[0].round())
        print(face[pred])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.rectangle(frame, (x-2, y - 40), (x + w+2, y),(0, 255, 0), cv2.FILLED)
        cv2.putText(frame, face[pred], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Croped Image", faceImg)
    cv2.imshow("Image", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break



