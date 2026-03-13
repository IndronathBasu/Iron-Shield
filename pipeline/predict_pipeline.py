from ultralytics import YOLO
import cv2
from inference.predict_hb import predict_hb
from segmentation.predict_mask import predict_mask

eye_detector = YOLO("eye_detector.pt")

def run_pipeline(image_path):

    image = cv2.imread(image_path)

    # Detect eye
    results = eye_detector(image)

    box = results[0].boxes.xyxy[0].cpu().numpy()

    x1,y1,x2,y2 = map(int,box)

    eye_crop = image[y1:y2,x1:x2]

    # Segment conjunctiva
    mask = predict_mask(seg_model, eye_crop)

    conjunctiva = eye_crop * mask[:,:,None]

    cv2.imwrite("temp_conjunctiva.jpg", conjunctiva)

    # Predict Hb
    hb = predict_hb("temp_conjunctiva.jpg")

    return hb