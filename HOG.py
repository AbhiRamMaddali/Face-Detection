import cv2
import numpy as np
# this is HOG face detection algorithm
# HOG= Histiogram of oriented gradient
# Load the pre-trained HOG + SVM model for face detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load an image from file
image_path = 'img.jpg'
image = cv2.imread(image_path)

# Resize the image if it's too large
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# Detect faces in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.06)

# Draw rectangles around the detected faces
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



