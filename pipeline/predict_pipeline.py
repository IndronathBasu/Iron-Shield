import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import numpy as np

from detection.detect_eye import EyeDetector
from segmentation.predict_mask import predict_mask
from inference.predict_hb import predict_hb
from models.risk_model import classify_anemia

detector = EyeDetector()


def run_pipeline(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return

    # ------------------------
    # 1 Eye Detection
    # ------------------------

    boxes = detector.detect(image)

    if len(boxes) == 0:
        print("No eye detected")
        return

    x1, y1, x2, y2 = map(int, boxes[0])

    eye_crop = image[y1:y2, x1:x2]

    cv2.imwrite("outputs/eye_crop.jpg", eye_crop)

    # ------------------------
    # 2 Segmentation
    # ------------------------

    mask = predict_mask(eye_crop)

    mask = (mask > 0.5).astype("uint8")

    # Resize mask to match eye crop
    mask = cv2.resize(mask, (eye_crop.shape[1], eye_crop.shape[0]))

    conjunctiva = eye_crop * mask[:, :, None]

    cv2.imwrite("outputs/conjunctiva.jpg", conjunctiva)

    # ------------------------
    # 3 Hb Prediction
    # ------------------------

    hb = predict_hb("outputs/conjunctiva.jpg")

    # ------------------------
    # 4 Risk Classification
    # ------------------------

    risk = classify_anemia(hb)

    print("\nRESULT")
    print("--------------------")
    print(f"Hemoglobin : {hb:.2f} g/dL")
    print(f"Risk Level : {risk}")


if __name__ == "__main__":

    img_path = input("Enter image path: ")

    run_pipeline(img_path)