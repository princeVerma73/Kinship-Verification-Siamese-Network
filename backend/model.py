import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet18 backbone
        self.backbone = models.resnet18(weights=None)

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Similarity classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, img1, img2):
        emb1 = self.forward_once(img1)
        emb2 = self.forward_once(img2)

        diff = torch.abs(emb1 - emb2)

        output = self.classifier(diff)

        return output