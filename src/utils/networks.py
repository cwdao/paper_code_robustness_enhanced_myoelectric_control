# coding=utf8
# collect network function or classes
#####################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset


#####################################################
# for Ninapro db1 10ch data
class EMGDataset_2D(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        emgData = self.data[index, :, :, :]
        emgData = np.squeeze(emgData)
        emglabel = self.label[index]
        emglabel = emglabel.astype(np.int16)
        emgData = self.transforms(emgData)
        # 一维数据用下面的这个就行
        # emgData = torch.Tensor(emgData)
        return emgData, emglabel

    def __len__(self):
        return len(self.label)


#####################################################
# for Ninapro db1 10ch data
class EMGDataset_1D(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        emgData = self.data[index, :, :, :]
        emgData = np.squeeze(emgData)
        emglabel = self.label[index]
        emglabel = emglabel.astype(np.int16)
        # emgData = self.transforms(emgData)一维数据用下面的这个就行
        emgData = torch.Tensor(emgData)
        return emgData, emglabel

    def __len__(self):
        return len(self.label)


#####################################################
#####################################################
# combination of basic network parameters, alpha version ^_^
class Basic_Network_alpha(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=0
        )

        self.fc1 = nn.Linear(in_features=32 * 18 * 8, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=10)
        self.dr1 = nn.Dropout2d(0.2)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        # t = self.dr1(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (4) hidden linear layer
        t = t.reshape(-1, 32 * 18 * 8)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.dr1(t)

        # (5) output layer
        t = self.out(t)

        return t


#####################################################
# primal GAN generator
def GAN_Gen(latent_size, n_g_feature, n_channel):
    Gen = nn.Sequential(
        nn.Flatten(),
        nn.Linear(latent_size, n_g_feature * 2),  # 用线性变换将输入映射到64维
        nn.BatchNorm1d(n_g_feature * 2),
        nn.ReLU(True),  # relu激活
        nn.Dropout2d(0.2),
        nn.Linear(n_g_feature * 2, n_g_feature * 4),  # 线性变换
        nn.BatchNorm1d(n_g_feature * 4),
        nn.ReLU(True),  # relu激活
        nn.Dropout2d(0.2),
        nn.Linear(n_g_feature * 4, n_channel),  # 线性变换
    )
    return Gen


#####################################################
# primal GAN discriminator
def GAN_Dis(n_channel, n_d_feature):
    Dis = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_channel, n_d_feature * 2),  # 输入特征数为784，输出为256
        nn.BatchNorm1d(n_d_feature * 2),
        nn.LeakyReLU(0.2),  # 进行非线性映射
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature * 2, n_d_feature),  # 进行一个线性映射
        nn.BatchNorm1d(n_d_feature),
        nn.LeakyReLU(0.2),
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature, 1),
        # nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
        # sigmoid可以班实数映射到【0,1】，作为概率值，
        # 多分类用softmax函数
    )
    return Dis


#####################################################
# primal GAN generator alpha
def GAN_Gen_alpha(latent_size, n_g_feature, n_channel):
    Gen = nn.Sequential(
        nn.Flatten(),
        nn.Linear(latent_size, n_g_feature * 2),  # 用线性变换将输入映射到64维
        nn.BatchNorm1d(n_g_feature * 2),
        nn.ReLU(True),  # relu激活
        nn.Dropout2d(0.2),
        nn.Linear(n_g_feature * 2, n_g_feature * 4),  # 线性变换
        nn.BatchNorm1d(n_g_feature * 4),
        nn.ReLU(True),  # relu激活
        nn.Dropout2d(0.2),
        nn.Linear(n_g_feature * 4, n_channel),  # 线性变换
    )
    return Gen


#####################################################
# primal GAN discriminator alpha
def GAN_Dis_alpha(n_channel, n_d_feature):
    Dis = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_channel, n_d_feature * 2),  # 输入特征数为784，输出为256
        nn.BatchNorm1d(n_d_feature * 2),
        nn.LeakyReLU(0.2),  # 进行非线性映射
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature * 2, n_d_feature),  # 进行一个线性映射
        nn.BatchNorm1d(n_d_feature),
        nn.LeakyReLU(0.2),
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature, 1),
        # nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
        # sigmoid可以班实数映射到【0,1】，作为概率值，
        # 多分类用softmax函数
    )
    return Dis


#####################################################
# OpGan's CNN for shimmer signal,6 channels,10 classes, the first 6 classes
# will be regard as known class.
class Network_CNN_6ch_6cls_smr_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=0
        )

        self.fc1 = nn.Linear(in_features=32 * 4 * 198, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=6)
        self.dr1 = nn.Dropout2d(0.2)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        # t = self.dr1(t)
        t = F.max_pool2d(t, kernel_size=2, stride=1)

        # (4) hidden linear layer
        t = t.reshape(-1, 32 * 4 * 198)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.dr1(t)

        # (5) output layer
        t = self.out(t)

        return t


#####################################################
# primal GAN discriminator alpha
def GAN_Dis_opengan_alpha(n_channel, n_d_feature):
    Dis = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_channel, n_d_feature * 2),
        nn.BatchNorm1d(n_d_feature * 2),
        nn.LeakyReLU(0.2),  # 进行非线性映射
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature * 2, n_d_feature),  # 进行一个线性映射
        nn.BatchNorm1d(n_d_feature),
        nn.LeakyReLU(0.2),
        nn.Dropout2d(0.2),
        nn.Linear(n_d_feature, 1),
        # nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
        # sigmoid可以班实数映射到【0,1】，作为概率值，
        # 多分类用softmax函数
    )
    return Dis
