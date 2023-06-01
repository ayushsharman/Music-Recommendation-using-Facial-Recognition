import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.models import model_from_json
import requests
import imutils
import time as t

df = pd.read_csv("C:/Users/Ayush Sharma/Dropbox/Music Recommendation Using Facial Emotion Recognition/Music Database - Sheet1.csv")

# Loading the model and weights into the notebook
json_file = open("C:/Users/Ayush Sharma/Dropbox/Music Recommendation Using Facial Emotion Recognition/emotion_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("C:/Users/Ayush Sharma/Dropbox/Music Recommendation Using Facial Emotion Recognition/emotion_model.h5")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
print("Loaded model from disk")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_classifier = cv2.CascadeClassifier("C:/Users/Ayush Sharma/Dropbox/Music Recommendation Using Facial Emotion Recognition/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # for webcam
while True:
    camera, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            print(label)

            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if label == 'Neutral':
                random_song_index = random.randint(0, 10)
                print(df.loc[random_song_index, [label]])

            elif label == "Surprise":
                random_song_index = random.randint(0, 10)
                print(df.iloc[random_song_index, [2]])

            elif label == "Happy":
                random_song_index = random.randint(0, 10)
                print(df.loc[random_song_index, [label]])

            elif label == "Sad":
                random_song_index = random.randint(0, 10)
                print(df.loc[random_song_index, [label]])

            elif label == "Angry":
                random_song_index = random.randint(0, 10)
                print(df.loc[random_song_index, [label]])

            elif label == "Disgust":
                random_song_index = random.randint(0, 10)
                print(df.loc[random_song_index, [label]])

            elif label == "Fear":
                random_song_index = random.randint(0, 9)
                print(df.iloc[random_song_index, [6]])

            else:
                print("Detecting")
    
    cv2.imshow('Emotion Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
