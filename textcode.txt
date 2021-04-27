from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import sys

model = load_model('Model_2_weights_best.h5')
CLASSES = open('labels.txt').read().strip().split("\n")

cap = cv2.VideoCapture('video2.mp4')

while True:
    frames = []
    while len(frames) < 30:
        ret, frame = cap.read()

        if ret == False:
            sys.exit(0)

        temp_frame = cv2.resize(frame, (128, 128))
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
        temp_frame = img_to_array(temp_frame)
        frames.append(temp_frame)

    temp_frame = np.stack(frames)
    temp_frame = np.expand_dims(temp_frame, axis=0)

    frame = cv2.resize(frame, (500, 500))
    class_label = CLASSES[np.argmax(model.predict(temp_frame), axis=-1)[0]]
    cv2.putText(frame, class_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Activity Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        sys.exit(0)
