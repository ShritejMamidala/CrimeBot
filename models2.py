
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import numpy as np

#SAME AS IN TRAINING SCRIPT - class located here for easy acsess/to be able to put it into main server pipeline file


class FaceAttributeModel(nn.Module):
    def __init__(self, num_features):
        super(FaceAttributeModel, self).__init__()
        self.base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = nn.Sequential()  

        self.fc1 = nn.Linear(in_features, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_features)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)