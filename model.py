#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:01:03 2023

@author: vaishnavijanakiraman
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetCustom(nn.Module):
    def __init__(self):
        super(ResNetCustom, self).__init__()
        
        #Prep layer
        
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        #Layer 1
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        
        #Layer 2
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        #Layer 3
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.mp2 = nn.MaxPool2d(4, 4)
        self.output_linear = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.conv1(x)
        x = x + self.res1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.res2(x)
        x = self.mp2(x)
        x = x.squeeze()
        x = self.output_linear(x)
        return F.log_softmax(x, dim=-1)