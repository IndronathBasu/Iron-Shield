import cv2
import os

MASK_DIR = "dataset/segmentation/masks"

for file in os.listdir(MASK_DIR):

    path = os.path.join(MASK_DIR, file)

    img = cv2.imread(path)

    cv2.imwrite(path, img)

print("Masks cleaned")