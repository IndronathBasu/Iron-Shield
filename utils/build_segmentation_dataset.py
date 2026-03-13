import os
import shutil

RAW_DATA = "dataset/anemia/raw"
OUT_IMG = "dataset/segmentation/images"
OUT_MASK = "dataset/segmentation/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

count = 0

for country in ["India", "Italy"]:

    country_path = os.path.join(RAW_DATA, country)

    for folder in os.listdir(country_path):

        folder_path = os.path.join(country_path, folder)

        if not os.path.isdir(folder_path):
            continue

        image_file = None
        mask_file = None

        for file in os.listdir(folder_path):

            lower = file.lower()

            # original eye image
            if lower.endswith(".jpg") or lower.endswith(".png"):
                if "_forniceal" not in lower and "_palpebral" not in lower:
                    image_file = os.path.join(folder_path, file)

            # conjunctiva mask
            if "forniceal_palpebral" in lower:
                mask_file = os.path.join(folder_path, file)

        if image_file and mask_file:

            new_name = f"{country}_{folder}"

            shutil.copy(image_file, os.path.join(OUT_IMG, new_name + ".jpg"))
            shutil.copy(mask_file, os.path.join(OUT_MASK, new_name + ".png"))

            count += 1

print(f"Segmentation dataset created with {count} samples")