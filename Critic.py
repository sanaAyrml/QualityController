import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)



# model definition
class Critic(nn.Module):
    # define model elements
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.conv2 = conv3x3(32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.linear1 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(512, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.linear4 = nn.Linear(128, 1)
        
    # forward propagate input
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu1(x)
        # print(x.shape)
        x = self.mp1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        x = self.relu5(x)
        x = self.linear4(x)
        return x