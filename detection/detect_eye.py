import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("models/eye_detector.pt")


def detect_eye(image_path):

    img = cv2.imread(image_path)

    results = model(img)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        raise Exception("No eye detected")

    x1, y1, x2, y2 = boxes[0]

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    eye_crop = img[y1:y2, x1:x2]

    return eye_crop


# CLI testing mode
if __name__ == "__main__":

    path = input("Enter image path: ")

    crop = detect_eye(path)

    cv2.imshow("Eye Crop", crop)
    cv2.waitKey(0)