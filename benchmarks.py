from capture.webcam import WebcamSource
import cv2

TEST_WIDTH = 1280
TEST_HEIGHT = 720 
cam = WebcamSource(width = TEST_WIDTH, height = TEST_HEIGHT)
(provided_fps, actual_fps), capture_latency = cam.get_info()
print(
    str(provided_fps) + 'fps',
    str(actual_fps) + 'fps',
    str(capture_latency) + 'ms',
    sep = '\n'
)