import torch
import cv2
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

model.load_state_dict(torch.load("models/conjunctiva_unet.pth"))
model = model.to(device)
model.eval()

def predict_mask(image):

    image = cv2.resize(image,(224,224))
    image = image/255.0

    tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(tensor)

    mask = pred.squeeze().cpu().numpy()

    return mask