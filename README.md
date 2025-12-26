# Facial Emotion Detection Using Deep Learning

This project implements a Facial Emotion Detection system using
Convolutional Neural Networks (CNN) and Computer Vision techniques.
The system captures facial images from a webcam, detects the face,
and predicts the emotion displayed on the face in real time.

The model is trained on grayscale facial images of size 48×48 pixels
and classifies emotions into predefined categories.

---

## Emotions Recognized
The system can detect the following emotions:
Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

---

## Working Description
The application uses a Haar Cascade Classifier to detect human faces
from a live webcam feed. Once a face is detected, it is converted to
grayscale, resized to 48×48 pixels, and passed to a trained CNN model.
The CNN predicts the emotion, which is then displayed on the screen
along with a bounding box around the detected face.

---

## Technologies Used
Python, TensorFlow/Keras, OpenCV, NumPy

---

## Dataset Used
The model is trained using publicly available facial emotion datasets.
The most commonly used and recommended dataset is FER-2013.

Dataset URLs:
FER-2013: https://www.kaggle.com/datasets/msambare/fer2013  
CK+ Dataset: https://www.kaggle.com/datasets/shawon10/ckplus  
JAFFE Dataset: https://zenodo.org/record/3451524  

---

## How to Run
Execute the Python file that contains the emotion detection code.
The webcam will start automatically and display the predicted emotion
on the detected face. Press the ESC key to stop the program.

---

## Applications
Human Computer Interaction, Emotion Analysis, Mental Health Monitoring,
Smart Surveillance Systems, AI-based User Experience Analysis

---

## Conclusion
This project demonstrates how deep learning and computer vision can be
combined to recognize human emotions from facial expressions in real time.
It serves as a practical implementation of CNN-based image classification
and is suitable for academic, internship, and learning purposes.

---

## Author
Sharvari Kale  

