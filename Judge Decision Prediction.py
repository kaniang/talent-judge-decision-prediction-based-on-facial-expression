from tkinter import Tk
from tkinter.filedialog import askopenfilename

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

face_classifier = cv2.CascadeClassifier(r'haarcasecade\haarcascade_frontalface_alt.xml')
classifier = load_model(r'model\alexnetmodbest.h5')

emotion_labels = ['Angry', 'Disgust', 'Happy', 'Neutral', 'Surprise']
listEkpresi = list()

Tk().withdraw()
filename = askopenfilename()
cap = cv2.VideoCapture(filename)

while cap.isOpened():
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]  # This particular code will return the cropped face from the image.
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            listEkpresi.append(label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

angryJml = listEkpresi.count('Angry')
disgustJml = listEkpresi.count('Disgust')
happyJml = listEkpresi.count('Happy')
neutralJml = listEkpresi.count('Neutral')
surpriseJml = listEkpresi.count('Surprise')

print('\nJumlah dari Setiap Ekpresi yang ditampilkan oleh juri :')
print('\nAngry : %d \nDisgust : %d\nHappy : %d\nNeutral : %d\nSurprise : %d' %
      (angryJml, disgustJml, happyJml, neutralJml, surpriseJml))


#######
#fuzzy#
#######

angry = ctrl.Antecedent(np.arange(0, 100, 1), 'angry')
disgust = ctrl.Antecedent(np.arange(0, 100, 1), 'disgust')
happy = ctrl.Antecedent(np.arange(0, 100, 1), 'happy')
neutral = ctrl.Antecedent(np.arange(0, 100, 1), 'neutral')
surprise = ctrl.Antecedent(np.arange(0, 100, 1), 'surprise')
kualifikasi = ctrl.Consequent(np.arange(0, 5, 1), 'kualifikasi')

# Set the membership of angry
angry['low'] = fuzz.trapmf(angry.universe, [0, 0, 20, 40])
angry['high'] = fuzz.trapmf(angry.universe, [20, 40, 100, 100])

# Set the membership of disgust
disgust['low'] = fuzz.trapmf(disgust.universe, [0, 0, 10, 20])
disgust['high'] = fuzz.trapmf(disgust.universe, [10, 20, 100, 100])

# Set the membership of happy
happy['low'] = fuzz.trapmf(happy.universe, [0, 0, 40, 60])
happy['high'] = fuzz.trapmf(happy.universe, [40, 60, 100, 100])

# Set the membership of neurtal
neutral['low'] = fuzz.trapmf(neutral.universe, [0, 0, 60, 70])
neutral['high'] = fuzz.trapmf(neutral.universe, [60, 70, 100, 100])

# Set the membership of surprise
surprise['low'] = fuzz.trapmf(surprise.universe, [0, 0, 5, 10])
surprise['high'] = fuzz.trapmf(surprise.universe, [5, 10, 100, 100])

# Set the membership function of the output result
kualifikasi['no'] = fuzz.trapmf(kualifikasi.universe, [0, 0, 1, 3])
kualifikasi['yes'] = fuzz.trapmf(kualifikasi.universe, [2, 3, 5, 5])

# rule

rule1 = ctrl.Rule(angry['low'] & disgust['low'] & happy['low']
                  & neutral['low'] & surprise['low'], kualifikasi['no'])

rule2 = ctrl.Rule(angry['low'] & disgust['low'] & happy['low']
                  & neutral['low'] & surprise['high'], kualifikasi['yes'])

rule3 = ctrl.Rule(angry['low'] & disgust['low'] & happy['low']
                  & neutral['high'] & surprise['low'], kualifikasi['no'])

rule4 = ctrl.Rule(angry['low'] & disgust['low'] & happy['low']
                  & neutral['high'] & surprise['high'], kualifikasi['yes'])

rule5 = ctrl.Rule(angry['low'] & disgust['low'] & happy['high']
                  & neutral['low'] & surprise['low'], kualifikasi['yes'])

rule6 = ctrl.Rule(angry['low'] & disgust['low'] & happy['high']
                  & neutral['low'] & surprise['high'], kualifikasi['yes'])

rule7 = ctrl.Rule(angry['low'] & disgust['low'] & happy['high']
                  & neutral['high'] & surprise['low'], kualifikasi['yes'])

rule8 = ctrl.Rule(angry['low'] & disgust['low'] & happy['high']
                  & neutral['high'] & surprise['high'], kualifikasi['yes'])

rule9 = ctrl.Rule(angry['low'] & disgust['high'] & happy['low']
                  & neutral['low'] & surprise['low'], kualifikasi['no'])

rule10 = ctrl.Rule(angry['low'] & disgust['high'] & happy['low']
                   & neutral['low'] & surprise['high'], kualifikasi['no'])

rule11 = ctrl.Rule(angry['low'] & disgust['high'] & happy['low']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule12 = ctrl.Rule(angry['low'] & disgust['high'] & happy['low']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

rule13 = ctrl.Rule(angry['low'] & disgust['high'] & happy['high']
                   & neutral['low'] & surprise['low'], kualifikasi['no'])

rule14 = ctrl.Rule(angry['low'] & disgust['high'] & happy['high']
                   & neutral['low'] & surprise['high'], kualifikasi['no'])

rule15 = ctrl.Rule(angry['low'] & disgust['high'] & happy['high']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule16 = ctrl.Rule(angry['low'] & disgust['high'] & happy['high']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

rule17 = ctrl.Rule(angry['high'] & disgust['low'] & happy['low']
                   & neutral['low'] & surprise['low'], kualifikasi['no'])

rule18 = ctrl.Rule(angry['high'] & disgust['low'] & happy['low']
                   & neutral['low'] & surprise['high'], kualifikasi['yes'])

rule19 = ctrl.Rule(angry['high'] & disgust['low'] & happy['low']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule20 = ctrl.Rule(angry['high'] & disgust['low'] & happy['low']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

rule21 = ctrl.Rule(angry['high'] & disgust['low'] & happy['high']
                   & neutral['low'] & surprise['low'], kualifikasi['no'])

rule22 = ctrl.Rule(angry['high'] & disgust['low'] & happy['high']
                   & neutral['low'] & surprise['high'], kualifikasi['yes'])

rule23 = ctrl.Rule(angry['high'] & disgust['low'] & happy['high']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule24 = ctrl.Rule(angry['high'] & disgust['low'] & happy['high']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

rule25 = ctrl.Rule(angry['high'] & disgust['high'] & happy['low']
                   & neutral['low'] & surprise['low'], kualifikasi['no'])

rule26 = ctrl.Rule(angry['high'] & disgust['high'] & happy['low']
                   & neutral['low'] & surprise['high'], kualifikasi['no'])

rule27 = ctrl.Rule(angry['high'] & disgust['high'] & happy['low']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule28 = ctrl.Rule(angry['high'] & disgust['high'] & happy['low']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

rule29 = ctrl.Rule(angry['high'] & disgust['high'] & happy['high']
                   & neutral['low'] & surprise['low'], kualifikasi['no'])

rule30 = ctrl.Rule(angry['high'] & disgust['high'] & happy['high']
                   & neutral['low'] & surprise['high'], kualifikasi['no'])

rule31 = ctrl.Rule(angry['high'] & disgust['high'] & happy['high']
                   & neutral['high'] & surprise['low'], kualifikasi['no'])

rule32 = ctrl.Rule(angry['high'] & disgust['high'] & happy['high']
                   & neutral['high'] & surprise['high'], kualifikasi['no'])

# Set the control system
kontrol_keputusan = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                                       rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32])
keputusan = ctrl.ControlSystemSimulation(kontrol_keputusan)


keputusan.input['angry'] = angryJml
keputusan.input['disgust'] = disgustJml
keputusan.input['happy'] = happyJml
keputusan.input['neutral'] = neutralJml
keputusan.input['surprise'] = surpriseJml

# Crunch the numbers
keputusan.compute()
nilai = keputusan.output['kualifikasi']
print(keputusan.output['kualifikasi'])
kkm = 2
if nilai > kkm:
    print('\nPrediksi Keputusan Juri adalah "Yes" ')
else:
    print('\nPrediksi Keputusan Juri adalah "No" ')