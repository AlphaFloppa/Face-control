import cv2 as cv
from .camera import CameraSource

class WebcamSource(CameraSource):
    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        self.camera_connection = cv.VideoCapture(device_id)
        self.camera_connection.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.camera_connection.set(cv.CAP_PROP_FRAME_HEIGHT, height)
