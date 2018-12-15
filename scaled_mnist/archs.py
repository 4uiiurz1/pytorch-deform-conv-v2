# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from deform_conv_v2 import *


class ScaledMNISTNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        features = []
        inplanes = 1
        outplanes = 32
        for i in range(4):
            if args.deform and args.min_deform_layer <= i+1:
                features.append(DeformConv2d(inplanes, outplanes, 3, padding=1, bias=False, modulation=args.modulation))
            else:
                features.append(nn.Conv2d(inplanes, outplanes, 3, padding=1, bias=False))
            features.append(nn.BatchNorm2d(outplanes))
            features.append(self.relu)
            if i == 1:
                features.append(self.pool)
            inplanes = outplanes
            outplanes *= 2
        self.features = nn.Sequential(*features)

        self.fc = nn.Linear(256, 10)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)

        return output
