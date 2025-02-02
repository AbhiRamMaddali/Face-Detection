import cv2
import numpy as np

# Load Haar cascade for face detection
haarcascade_path='/home/abhiram/haarcascade_frontalface_default.xml.1'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load or train the recognizer (if you have a trained model)
# recognizer.read('lbph_model.yml')

# Function to detect faces and extract LBPH features
def detect_faces_and_extract_lbph(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]

        # Compute LBPH features (for recognition)
        label, confidence = recognizer.predict(face_roi)
        print(f"Label: {label}, Confidence: {confidence}")

        # Display the label and confidence
        cv2.putText(image, f"Label: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the output
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function
detect_faces_and_extract_lbph('img.jpg')
