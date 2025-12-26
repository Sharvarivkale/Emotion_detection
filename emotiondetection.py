import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img

# ==============================
# Load Emotion Detection Model
# ==============================
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# ==============================
# Load Haar Cascade for Face Detection
# ==============================
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# ==============================
# Emotion Labels
# ==============================
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# ==============================
# Feature Extraction Function
# ==============================
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ==============================
# Start Webcam
# ==============================
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        img = extract_features(face)
        pred = model.predict(img)
        emotion_label = labels[pred.argmax()]

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
