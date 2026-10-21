from abc import ABC, abstractmethod
import numpy
import time
import cv2 as cv


class CameraSource(ABC):
    camera_connection: cv.VideoCapture

    @abstractmethod
    def __init__(self) -> None:
        pass

    def get_frame(self) -> tuple[bool, numpy.ndarray | None]:
        is_success, frame = self.camera_connection.read()
        return is_success, frame if is_success else None

    def close(self) -> None:
        self.camera_connection.release()

    def is_active(self) -> bool:
        return self.camera_connection.isOpened()

    def get_info(self) -> tuple[tuple[float, float], float]:
        measurement_start = time.perf_counter()
        delta = 8
        frames_counter = 0
        latencies = []
        while time.perf_counter() - measurement_start < delta:
            before = time.perf_counter()
            is_success, _ = self.get_frame()
            if is_success:
                frames_counter += 1
                latencies.append((time.perf_counter() - before) * 1000)

        return (
            (
                self.camera_connection.get(cv.CAP_PROP_FPS), 
                frames_counter / delta
            ),
            sum(latencies) / len(latencies)
        ) 
