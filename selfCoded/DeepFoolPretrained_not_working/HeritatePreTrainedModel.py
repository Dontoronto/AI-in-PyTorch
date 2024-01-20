import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet101, ResNet101_Weights

# Define your custom model inheriting from nn.Module
class DeepFoolHeritage(nn.Module):
    def __init__(self,weights):
        super(DeepFoolHeritage, self).__init__()
        # Load a pre-trained ResNet model
        self.weights = weights
        self.model = resnet101(pretrained=True,weights=self.weights.transforms(antialias=None))
        #self.model.requires_grad_(False)
        self.model.eval()

        #self.preprocess = self.weights.transforms()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


    def forward(self, img):
        _ = img.squeeze(0)
        print("raw input after squeeze")
        print(_.shape)
        #batch = self.preprocess(_)
        print("before unsqueeze")
        #print(batch.shape)
        batch = _.unsqueeze(0)
        #batch = torch.clamp(batch, 0, 1)
        #print(batch)
        # Define the forward pass, utilizing the pre-trained backbone
        return self.model(batch)

