## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 8, 7)),
            ('relu1', nn.ReLU()),
            ('mp1', nn.MaxPool2d(2,2)),
            ('drop1', nn.Dropout())]))
        
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(8, 16, 5)),
            ('relu2', nn.ReLU()),
            ('mp2', nn.MaxPool2d(2,2)),
            ('drop2', nn.Dropout())]))
        
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(16, 32, 3)),
            ('relu3', nn.ReLU()),
            ('mp3', nn.MaxPool2d(2,2)),
            ('drop3', nn.Dropout())]))
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(12*12*32, 1000)),
            ('relu4', nn.ReLU()),
            ('drop4', nn.Dropout())]))
        self.fc2 = nn.Linear(1000, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool1(x)
        x = x.view(-1, 12*12*32)
        x = self.fc1(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
