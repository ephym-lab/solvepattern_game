import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

import numpy as np
dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
result = detector.detect(mp_image)
print(dir(result))
print(result.hand_landmarks)
