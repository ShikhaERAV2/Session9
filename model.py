from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import albumentations
import albumentations.pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1
        # Depthwise sperable layer
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1, groups=3), #### Depthwise convolution
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=1), ### Pointwise Convolution (using depthwise output) <-- This is called DEPTHWISE-SEPARABLE CONVOLUTION
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 2 - Dialated Convolution
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )

        self.dropout = nn.Dropout(0.01)

        # First fully connected layer
        self.fc1 = nn.Linear(128*8*8, 100) # [ [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].]
        self.fc2 = nn.Linear(100, 10) # [ [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].]

    def forward(self, x):
        x = self.input(x)
        #print('input layer - ',x.shape)
        x = self.convblock1(x)
        #print('conv1 - ',x.shape)
        x = self.convblock2(x)
        #print('conv2 - ',x.shape)
        x = self.convblock3(x)
        #print('conv3 - ',x.shape)
        x = self.convblock4(x)
        #print('conv4 - ',x.shape)
        x = self.gap(x)
        #print('Gap - ',x.shape)
        #print(x.view(-1, x.shape[-1]))
        x = x.view(-1, 128*8*8) #[CHANNEL_NUMBER] * [HEIGHT] * [WIDTH]
        #print('Flatten - ', x.view(-1, x.shape[-1]).shape[0])
        x = self.fc1(x)
        x = self.fc2(x)
        #print('Linear - ',x.shape)
        return F.log_softmax(x, dim=-1)
