import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
filepath = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/JPEGImages/'  # 数据集目录

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

class CustomDataset(torch.utils.data.Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self,root,transform):
        # TODO
        # 1. Initialize file path or list of file names.
        self.root=root
        self.transform=transform
        self.img_path=os.listdir(self.root)
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data point
        image = Image.open(os.path.join(self.root, self.img_path[index]))
        # label = 0 if self.img_path[index].split('.')[0] == 'cat' else 1  # label, 猫为0，狗为1
        # if self.transform:
        #     image = self.transform(image)
        # label = torch.from_numpy(np.array([label]))
        image=self.transform(image)
        label=1
        return image,label

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_path)
if __name__ == '__main__':
    transform=transforms.ToTensor()
    train_dataset = CustomDataset(root=filepath, transform=transform)
    print(getStat(train_dataset))