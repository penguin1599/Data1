import torch
import torch.nn as nn
import torchvision.models as models

class HyperIQA(nn.Module):
    def __init__(self):
        super(HyperIQA, self).__init__()
        # Simplified HyperIQA structure.
        # Uses ResNet50 backbone.
        # Assumption: Pretrained weights would likely match a specific implementation (e.g. Sujit Roy's).
        
        # Backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Remove AvgPool and FC
        
        # HyperNet would go here.
        # For this prototype, we produce a single scalar score from features.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        score = self.fc(x)
        return score
