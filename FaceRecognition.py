import face_recognition
import cv2
import numpy as np


imgaAK = face_recognition.load_image_file('SC1.jpg')
imgaAK = cv2.cvtColor(imgaAK,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Surbhi2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgaAK)[0]
encodeAK = face_recognition.face_encodings(imgaAK)[0]
cv2.rectangle(imgaAK, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # top, right, bottom, left
# print(encodeAK)

faceLoctest = face_recognition.face_locations(imgTest)[0]
encodetest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeAK], encodetest)
faceDis = face_recognition.face_distance([encodeAK], encodetest)
print(results, faceDis)

cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgaAK, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Surbhi Chauhan1',imgaAK)
cv2.imshow('Surbhi Chauhan',imgTest)


cv2.waitKey(0)
