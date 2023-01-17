#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class MyEnsemble(nn.Module):
    def __init__(self, num_classes=3):
        super(MyEnsemble, self).__init__()
        self.modelA = models.video.r3d_18(pretrained=True)
        # Remove last linear layer
        
        # self.modelA.fc = nn.Identity()

        # Create new classifier
        # self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2,
        #                     batch_first=True)
        
        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_3d):
        
        out = self.modelA(x_3d) # torch.Size([64, 400])
        x = F.relu(out)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x