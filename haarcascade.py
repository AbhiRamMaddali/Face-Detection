import cv2
import cv2 as cv
import sys
imagepath="img.jpg"
cascpath="haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascpath)
image =cv2.imread(imagepath)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=1,
    minSize=(15,15),
    flags=cv2.CASCADE_SCALE_IMAGE
 )
print("faces detected: {0}".format(len(faces)))
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Faces found ",image)
cv2.waitKey(0)

