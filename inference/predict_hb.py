import sys
import os

# add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.hb_model import HbModel
from models.risk_model import classify_anemia


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = HbModel()
model.load_state_dict(torch.load("hb_model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def predict_hb(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image).item()

    return pred


if __name__ == "__main__":

    img = input("Enter image path: ").strip().strip('"').strip("'")

    hb = predict_hb(img)

    risk = classify_anemia(hb)

    print(f"\nPredicted Hemoglobin: {hb:.2f} g/dL")
    print("Anemia Risk:", risk)