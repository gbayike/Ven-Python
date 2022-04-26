from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import datetime
import os

def emotion_detection():
    then = datetime.datetime.now()
    now = datetime.datetime.now()
    duration = now - then
    duration_in_s = duration.total_seconds()

    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    face_classifier = cv2.CascadeClassifier(cascPathface)
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
    classifier = load_model('Models/ven_model_20epochs_final_v6.h5')

    emotions = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while duration_in_s < 10:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                if label == "Angry":
                    x = emotions.get("Angry")
                    emotions.update({"Angry": x + 1})

                elif label == "Disgust":
                    x = emotions.get("Disgust")
                    emotions.update({"Disgust": x + 1})

                elif label == "Fear":
                    x = emotions.get("Fear")
                    emotions.update({"Fear": x + 1})

                elif label == "Happy":
                    x = emotions.get("Happy")
                    emotions.update({"Happy": x + 1})

                elif label == "Neutral":
                    x = emotions.get("Neutral")
                    emotions.update({"Neutral": x + 1})

                elif label == "Sad":
                    x = emotions.get("Sad")
                    emotions.update({"Sad": x + 1})

                elif label == "Surprise":
                    x = emotions.get("Surprise")
                    emotions.update({"Surprise": x + 1})

            else:
                cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # print(now)
        print(duration_in_s)
        print(emotions)
        now = datetime.datetime.now()
        duration = now - then
        duration_in_s = duration.total_seconds()

        new_value = max(emotions, key=emotions.get)

        print("Highest value from dictionary:", new_value)

        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return max(emotions, key=emotions.get)


print("Emotion detected is: ", emotion_detection())
