from ultralytics import YOLO
import os

def train_eye_detector():

    dataset_config = "configs/eye_dataset.yaml"

    model = YOLO("yolov8n.pt")

    model.train(
        data=dataset_config,
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",
        project="runs",
        name="eye_detector",
        workers=4
    )

if __name__ == "__main__":
    train_eye_detector()