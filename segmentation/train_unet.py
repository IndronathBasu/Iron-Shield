import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset_loader import ConjunctivaDataset
from tqdm import tqdm
import os

IMAGE_PATH = "dataset/segmentation/images"
MASK_PATH = "dataset/segmentation/masks"

dataset = ConjunctivaDataset(IMAGE_PATH, MASK_PATH)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}\n")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

model = model.to(device)

dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 60

for epoch in range(epochs):

    model.train()

    running_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images, masks in progress_bar:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = dice_loss(preds, masks) + bce_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)

    print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/conjunctiva_unet.pth")

print("\nModel saved to models/conjunctiva_unet.pth")