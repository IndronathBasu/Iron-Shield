import torch
import cv2

def predict_mask(model, image):

    image = cv2.resize(image,(224,224))

    image = image.transpose(2,0,1)/255.0
    image = torch.tensor(image).float().unsqueeze(0)

    with torch.no_grad():
        mask = model(image)

    mask = mask.squeeze().numpy()

    return mask