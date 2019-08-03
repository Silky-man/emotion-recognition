# -*- coding:utf-8 -*-
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras.models import load_model
from utils import preprocess_input

# parameters for loading data and images
detector=MTCNN()
emotion_model_path = 'emotion.hdf5'
 emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                 4:'sad',5:'surprise',6:'neutral'}
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    fa = detector.detect_faces(bgr_image)
    #print(fa)
    #if len(fa) > 0:
    for i in range(len(fa)):    
        faces = list(fa[i].values())[0]
        x1, y1, width, height = faces
        x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
        if x1>0 and y1>0:
            gray_face = bgr_image[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(gray_face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            # emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            cv2.putText(bgr_image, emotion_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(bgr_image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
