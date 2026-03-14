import cv2
import numpy as np
import sys
sys.path.append(".")

from detection.detect_eye import detect_eye
from segmentation.predict_mask import predict_mask
from inference.predict_hb import predict_hb_from_image
from models.risk_model import classify_anemia


def run_pipeline(image_path):

    # Step 1: Detect Eye
    eye_crop = detect_eye(image_path)

    # Step 2: Segment Conjunctiva
    mask = predict_mask(eye_crop)
    mask = (mask > 0.5).astype("uint8")

    # Resize mask to match eye crop
    mask = cv2.resize(mask, (eye_crop.shape[1], eye_crop.shape[0]))

    # Step 3: Extract conjunctiva
    conjunctiva = eye_crop * mask[:, :, None]

    # Step 4: Predict Hb
    hb = predict_hb_from_image(conjunctiva)

    # Step 5: Risk classification
    risk = classify_anemia(hb)

    print("\nRESULT")
    print("--------------------")
    print("Hemoglobin :", round(hb,2), "g/dL")
    print("Risk Level :", risk)

    return hb, risk