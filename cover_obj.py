from pycocotools.coco import COCO
import numpy as np
import cv2 as cv

CAM_WIDTH = 640 #1920
CAM_HEIGHT = 480 #1080

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

#face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')




while True:
    ret, frame = cam.read()

    # frame_cinza = cv.cvtColor(frame, cv.COLOR_BGR_GRAY)


  #  faces = face_cascade.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5)

    # for (x, y, w, h) in faces:
    #     cv.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)

       
    cv.imshow('cam', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
