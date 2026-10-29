
from detect.detector import get_faces
from capture.webcam import WebcamSource
from align.aligner import align_face
import cv2
import numpy as np
import onnxruntime as ort

def preprocess_crop(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Критично: BGR -> RGB
    img = (img.astype(np.float32) - 127.5) / 128.0 # Нормализация
    return img[np.newaxis, ...] # Форма: (1, 112, 112, 3)


def normalize_emb_vector(emb):
    return emb / np.linalg.norm(emb)

cam = WebcamSource(width = 1280, height = 720)
emb_maker_session = ort.InferenceSession("models/arcface.onnx")
input_name = emb_maker_session.get_inputs()[0].name
output_name = emb_maker_session.get_outputs()[0].name

for i in range(2):
    isSuccess, frame = cam.get_frame()
    if isSuccess:
        face = get_faces(frame)[0]
        landmarks = face[4:14].reshape(5, 2)
        aligned_face = align_face(frame, landmarks)
        cv2.imwrite('crop' + str(i+1) + '.jpg', aligned_face)

emb1 = emb_maker_session.run(
    [output_name], 
    {
        input_name: preprocess_crop("crop1.jpg")
    }
)[0][0]
emb2 = emb_maker_session.run(
    [output_name], 
    {
        input_name: preprocess_crop("crop2.jpg")
    }
)[0][0]

cosine_sim = np.dot(
    normalize_emb_vector(emb1),
    normalize_emb_vector(emb2)
)
print("Cosine similarity:", cosine_sim)