import cv2
import os
import numpy as np

RAW_PATH = r"E:\anemia-ai\dataset\raw"
OUTPUT_PATH = r"E:\anemia-ai\dataset\processed\images"

os.makedirs(OUTPUT_PATH, exist_ok=True)

IMAGE_EXT = (".jpg", ".png", ".jpeg")


def extract_region(image_path, mask_path, save_path):

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        print("Failed to read:", image_path, mask_path)
        return

    # Resize mask to match image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    mask = mask / 255.0

    result = img * mask[:, :, None]

    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        print("Empty mask:", mask_path)
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = result[y0:y1, x0:x1]

    cv2.imwrite(save_path, cropped)

def process_country(country):

    country_path = os.path.join(RAW_PATH, country)

    print("Processing:", country)

    for folder in os.listdir(country_path):

        folder_path = os.path.join(country_path, folder)

        if not os.path.isdir(folder_path):
            continue

        image_file = None
        mask_file = None

        for file in os.listdir(folder_path):

            file_path = os.path.join(folder_path, file)

            # detect eye image
            if file.lower().endswith(IMAGE_EXT) and "palpebral" not in file.lower() and "forniceal" not in file.lower():
                image_file = file_path

            # detect palpebral mask
            if "palpebral" in file.lower():
                mask_file = file_path

        if image_file and mask_file:

            save_name = f"{country}_{folder}.jpg"
            save_path = os.path.join(OUTPUT_PATH, save_name)

            extract_region(image_file, mask_file, save_path)

            print("Saved:", save_name)

        else:
            print("Skipped:", folder)


process_country("India")
process_country("Italy")

print("Extraction finished")