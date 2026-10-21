from detect.detector import get_faces
from capture.webcam import WebcamSource
from align.aligner import align_face
import cv2

cam = WebcamSource(width = 640, height = 480)
isSuccess, frame = cam.get_frame()
if isSuccess:
    face = get_faces(frame)[0]
    landmarks = face[4:14].reshape(5, 2)
    aligned_face = align_face(frame, landmarks)
    cv2.imwrite('test.png', aligned_face)


