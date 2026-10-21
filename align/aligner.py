import cv2
import numpy as np

CROPDIMS = (112, 112)
# Эталонные точки для ArcFace 112×112
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(
    frame: np.ndarray,
    landmarks: np.ndarray
) -> np.ndarray | None:
    if landmarks is None or len(landmarks) != 5:
        return None
    
    src = landmarks.astype(np.float32)
    
    M, _ = cv2.estimateAffinePartial2D(src, REFERENCE_LANDMARKS)
    
    if M is None:
        return None
    
    aligned = cv2.warpAffine(
        frame,
        M,
        CROPDIMS,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    
    return aligned