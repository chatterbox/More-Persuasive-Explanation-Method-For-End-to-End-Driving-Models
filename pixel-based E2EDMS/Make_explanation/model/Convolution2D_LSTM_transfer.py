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
        self.modelA = models.resnet18(pretrained=True)
        # Remove last linear layer

        self.modelA.fc = nn.Identity()
        
        # Create new classifier
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2,
                            batch_first=True)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_3d):
        cnn_output_list = list()
        for t in range(x_3d.size(2)):
            # print(self.modelA(x_3d[:, :, t, :, :]).shape)  # torch.Size([64, 512])
            # exit()
            cnn_output_list.append(self.modelA(x_3d[:, :, t, :, :]))
            
        x = torch.stack(tuple(cnn_output_list), dim=1) # torch.Size([64, 2, 512])
        out, hidden = self.lstm(x)
        x = out[:, -1, :]
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
