
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os,sys
from PIL import Image
import matplotlib.pyplot as plt
import ipdb
import os
class SIM2MNIST_Dataset(Dataset):
    def __init__(self, data_dir,train):

        self.data_dir=data_dir
        self.train=train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ] )

        files = os.listdir(self.data_dir)
        for file in files:
            if self.train:
                if file == 'x_train.npy':
                    self.imgs = np.load(os.path.join(self.data_dir, file))
                if file == 'y_train.npy':
                    self.labels = np.load(os.path.join(self.data_dir, file))

            else:
                if file == 'x_test.npy':
                    self.imgs = np.load(os.path.join(self.data_dir, file))
                if file == 'y_test.npy':
                    self.labels = np.load(os.path.join(self.data_dir, file))

    def __getitem__(self, index):
        img=self.imgs[index,:,:]   # 1 42 42 numpy
        img=self.transform(img)
        temp_label=np.argmax(self.labels[index,:])   # numpy.int64
        # temp_label=temp_label.astype(np.ndarray)
        label=torch.tensor(temp_label)
        return img, label

    def __len__(self):
        return len(self.imgs)



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# data_dir='/media/n/SanDiskSSD/HardDisk/data/stn_mnist_RTSaug'
# train_data = SIM2MNIST_Dataset(data_dir=data_dir,train=True)
# train_loader = DataLoader(dataset=train_data,batch_size=64, shuffle=False)
#
# # # [1]使用epoch方法迭代，LfwDataset的参数repeat=1
# # for epoch in range(10):
# for  i,(data, label) in enumerate(train_loader):  # data :[64, 1, 28, 28]  target:[64]
#     ipdb.set_trace()
#     data, target = data.to(device), label.to(device)
#     a=1
#
#     # image = batch_image[0, :]
#     # image = image.numpy()  #
#     # # plt.imshow(image)
