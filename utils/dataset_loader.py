import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class AnemiaDataset(Dataset):

    def __init__(self, csv_file, image_dir):

        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = self.data.iloc[idx]["image"]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        hb = torch.tensor(self.data.iloc[idx]["hb"], dtype=torch.float32)

        return image, hb