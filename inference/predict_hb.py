import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(".")

from models.hb_model import HbModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = HbModel().to(device)
model.load_state_dict(torch.load("hb_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def predict_hb_from_image(image):

    # image is numpy array (conjunctiva crop)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        hb = model(image)

    hb = hb.item()

    return hb


# CLI testing
if __name__ == "__main__":

    path = input("Enter image path: ")

    img = cv2.imread(path)

    hb = predict_hb_from_image(img)

    print("Predicted Hb:", round(hb,2), "g/dL")