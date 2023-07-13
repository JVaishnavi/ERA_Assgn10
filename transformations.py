#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:20:45 2023

@author: vaishnavijanakiraman
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40),
        A.RandomCrop(height=32, width=32),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8,
                        min_holes=1, min_height=8, min_width=8, 
                        fill_value=(0.4914, 0.4822, 0.4465)),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ToTensorV2()
])