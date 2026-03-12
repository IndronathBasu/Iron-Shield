import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from utils.dataset_loader import AnemiaDataset
from models.hb_model import HbModel

DATASET_PATH = os.path.join(ROOT, "dataset", "processed", "labels.csv")
IMAGE_PATH = os.path.join(ROOT, "dataset", "processed", "images")

dataset = AnemiaDataset(DATASET_PATH, IMAGE_PATH)

loader = DataLoader(dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = HbModel()
MODEL_PATH = os.path.join(ROOT, "hb_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


predictions = []
targets = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        preds = model(images).squeeze().cpu().numpy()

        predictions.extend(preds)
        targets.extend(labels.numpy())


predictions = np.array(predictions)
targets = np.array(targets)


mae = mean_absolute_error(targets, predictions)

rmse = np.sqrt(mean_squared_error(targets, predictions))

r2 = r2_score(targets, predictions)

mape = np.mean(np.abs((targets - predictions) / targets)) * 100


print("\nMODEL EVALUATION METRICS")
print("------------------------")

print(f"MAE  : {mae:.3f} g/dL")
print(f"RMSE : {rmse:.3f} g/dL")
print(f"R²   : {r2:.3f}")
print(f"MAPE : {mape:.2f}%")