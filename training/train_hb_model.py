import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset_loader import AnemiaDataset
from models.hb_model import HbModel


DATASET_PATH = "dataset/processed/labels.csv"
IMAGE_PATH = "dataset/processed/images"


dataset = AnemiaDataset(DATASET_PATH, IMAGE_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


model = HbModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

EPOCHS = 20


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for images, labels in train_loader:

        preds = model(images).squeeze()

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            preds = model(images).squeeze()

            loss = criterion(preds, labels)

            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


torch.save(model.state_dict(), "hb_model.pth")

print("Model saved as hb_model.pth")