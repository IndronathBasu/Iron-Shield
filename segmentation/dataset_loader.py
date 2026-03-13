import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A


class ConjunctivaDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (224, 224))

        mask = mask / 255.0

        augmented = self.aug(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = image / 255.0

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask