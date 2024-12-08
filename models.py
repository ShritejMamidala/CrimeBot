import torch.nn as nn
from torchvision import models

#SAME AS IN TRAINING SCRIPT - class located here for easy acsess/to be able to put it into main server pipeline file

class MultiTaskModel(nn.Module):
    def __init__(self, device="cuda"):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True) 
        self.backbone.classifier = nn.Identity() 
        
        self.age_head = nn.Sequential(
    nn.Dropout(0.2), 
    nn.Linear(1280, 5)
    )
        self.gender_head = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 1)
    )
        self.race_head = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 6)
    )

    def forward(self, x):
        features = self.backbone(x) 
        age_out = self.age_head(features)
        gender_out = self.gender_head(features)
        race_out = self.race_head(features)
        return age_out, gender_out, race_out

    
