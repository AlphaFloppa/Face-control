import cv2 as cv
from webcam import CameraSource, WebcamSource

cam = WebcamSource(width=640, height=480)
(provided_fps, actual_fps), capture_latency = cam.get_info()
print(
    str(provided_fps) + 'fps',
    str(actual_fps) + 'fps',
    str(capture_latency) + 'ms',
    sep = '\n'
)

isSuccess, frame = cam.get_frame()
print(frame.shape[:2])

#webcam = WebcamSource(width=1280, height=720)
#frame = webcam.get_frame()
#if frame is not None:
#    cv.imwrite('testframe.png', frame)
#max_fps, real_fps = webcam.get_fps()
#print(max_fps, real_fps, sep='\n')
