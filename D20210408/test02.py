import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()   #0-1 转 0-255
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from torch.autograd import Variable
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


pic1 = './3.png'
pic2 = './4.png'
# image_1 = Image.open(pic1)
image_1 = cv2.imread(pic1)
image_2 = cv2.imread(pic2)
trans = transforms.ToTensor()

width = 28
height = 28

img1_resize = cv2.resize(image_1, (width, height))
img2_resize = cv2.resize(image_2, (width, height))

gray1 = cv2.cvtColor(img1_resize, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resize, cv2.COLOR_BGR2GRAY)

img1 = trans(gray1).unsqueeze(0)
img2 = trans(gray2).unsqueeze(0)

net = LeNet()
output1 = net(Variable(img1))
output2 = net(Variable(img2))

max_value1, max_index1 = torch.max(output1, dim=1)
max_value2, max_index2 = torch.max(output2, dim=1)
print(max_index1)
print(max_index2)


