import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


class HbModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )

        self.model.classifier = nn.Linear(1024,1)

    def forward(self,x):

        return self.model(x)