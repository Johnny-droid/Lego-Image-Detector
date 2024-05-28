import numpy as np
import torch
from torch import nn
from torchvision import models
np.random.seed(42)

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.pool_size = 2
        self.nb_filters = 32
        self.kernel_size = 3

        self.layers = nn.Sequential(
            nn.Conv2d(3, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.nb_filters, self.nb_filters, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),
            nn.Dropout(0.25),
            nn.Conv2d(self.nb_filters, self.nb_filters * 2, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.nb_filters * 2, self.nb_filters * 2, self.kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(179776, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 32)  # we have 32 classes to guess
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits



class VGG16Classifier(nn.Module):
    def __init__(self):
        super(VGG16Classifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 32)

    def forward(self, x):
        return self.vgg(x)
    
class VGG16Regression(nn.Module):
    def __init__(self):
        super(VGG16Regression, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 1)

    def forward(self, x):
        return self.vgg(x)

class ResNet18Classifier(nn.Module):
    def __init__(self):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 32)

    def forward(self, x):
        return self.resnet(x)