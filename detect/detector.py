import cv2
import numpy


def get_faces(frame) -> [] | None:
    detector = cv2.FaceDetectorYN.create(
        model="models/face_detection_yunet_2023mar.onnx",  # путь к модели
        config="",
        input_size=(320, 320),   # начальный размер (потом можно менять)
        score_threshold=0.6,     # порог уверенности
        nms_threshold=0.3,
        top_k=5000
    )

# 2. Установка реального размера кадра
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

# 3. Детекция — frame это numpy.ndarray
    retval, faces = detector.detect(frame)

    return faces if retval else None