#读取MNIST数据集
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                              ])
trainset=torchvision.datasets.CIFAR10(root='E:/HardDisk/data/CIFAR10/cifar-10-python', train=True, download=False, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# print(trainset[0].shape)
# (data,label)=trainset[0]            trainset[0] 直接调用      trainset[0]会报错   下标数字太大
for i,data in enumerate(trainloader, 0):
    inputs, labels = data