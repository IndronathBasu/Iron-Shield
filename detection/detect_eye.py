from ultralytics import YOLO
import cv2
import os

class EyeDetector:

    def __init__(self, model_path="models/eye_detector.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):

        results = self.model(image)

        boxes = results[0].boxes.xyxy.cpu().numpy()

        return boxes


def crop_eye(image_path, detector):

    image = cv2.imread(image_path)

    boxes = detector.detect(image)

    if len(boxes) == 0:
        print("No eye detected")
        return None

    x1, y1, x2, y2 = map(int, boxes[0])

    eye_crop = image[y1:y2, x1:x2]

    return eye_crop


if __name__ == "__main__":

    detector = EyeDetector()

    img_path = input("Enter image path: ")

    crop = crop_eye(img_path, detector)

    if crop is not None:

        os.makedirs("outputs", exist_ok=True)

        save_path = "outputs/eye_crop.jpg"

        cv2.imwrite(save_path, crop)

        print(f"Eye crop saved to: {save_path}")