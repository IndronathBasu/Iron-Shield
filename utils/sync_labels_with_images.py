import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils.dataset_loader import AnemiaDataset
from models.hb_model import HbModel


DATASET_PATH = "dataset/processed/labels.csv"
IMAGE_PATH = "dataset/processed/images"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


dataset = AnemiaDataset(DATASET_PATH, IMAGE_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


model = HbModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

EPOCHS = 20


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()

    train_losses = []
    train_preds = []
    train_labels = []

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).squeeze()

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        train_preds.extend(preds.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    train_loss = np.mean(train_losses)

    train_mae = np.mean(np.abs(np.array(train_preds) - np.array(train_labels)))

    train_rmse = np.sqrt(train_loss)

    model.eval()

    val_losses = []
    val_preds = []
    val_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).squeeze()

            loss = criterion(preds, labels)

            val_losses.append(loss.item())

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = np.mean(val_losses)

    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))

    val_rmse = np.sqrt(val_loss)

    print(
        f"Train Loss: {train_loss:.3f} | Train MAE: {train_mae:.3f} | Train RMSE: {train_rmse:.3f} || "
        f"Val Loss: {val_loss:.3f} | Val MAE: {val_mae:.3f} | Val RMSE: {val_rmse:.3f}"
    )


torch.save(model.state_dict(), "hb_model.pth")

print("\nTraining complete")
print("Model saved as hb_model.pth")