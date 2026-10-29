from capture.webcam import WebcamSource
import cv2 as cv
from detect.detector import get_faces


cam = WebcamSource()
while True:
    isSuccess, frame = cam.get_frame()
    if not isSuccess:
        continue
    face = get_faces(frame)[0]
    if face is None:
        continue
    x, y, w, h = face[:4].astype(int)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow("A", frame)
    cv.waitKey(1)

cam.close()
