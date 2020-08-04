import face_recognition
import cv2
import numpy as np
import os
from datetime import  datetime

path = 'images2'
name = []
image = []
personlist = os.listdir(path)

for person in personlist:
    img=cv2.imread(f'{path}/{person}')
    image.append(img)
    name.append(os.path.splitext(person)[0])




def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeing = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeing)
    return encodeList

EncodedList =findEncodings(image)

print('Encoding images')


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


    facesCurFrame = face_recognition.face_locations(gray)
    encodesCurFrame = face_recognition.face_encodings(gray, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # print(EncodedList.shape, encodeFace.shape)
        matches = face_recognition.compare_faces(EncodedList, encodeFace, tolerance=0.8)
        faceDis = face_recognition.face_distance(EncodedList, encodeFace)
        matchIndex = np.argmin(faceDis)



        if matches[matchIndex]:
            matchedname = name[matchIndex].upper()
            print(matchedname)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, matchedname, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            markAttendance(matchedname)

    # cv2.imshow('webcam', frame)
    # cv2.waitKey(0)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






