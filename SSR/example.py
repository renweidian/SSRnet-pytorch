# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:03:25 2020

@author: Dian
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:32:24 2020

@author: Dian
"""
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional as F
from torch import nn
import torch
import os
import torch.utils.data as data
import metrics
class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_hrms_all):
        self.train_hrhs_all  = train_hrhs_all
        self.train_hrms_all  = train_hrms_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms
    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data1(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr=init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
      lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

       



def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F

class SEBlock(nn.Module):
    def __init__(self):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(31,1,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(1,31,kernel_size=1,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.se(x)
        return x*x1 
    
class MyarcLoss(torch.nn.Module):
    def __init__(self):
        super(MyarcLoss, self).__init__()

    def forward(self, output, target):
        sum1=output*target
        sum2=torch.sum(sum1,dim=1)+1e-10
        norm_abs1=torch.sqrt(torch.sum(output*output,dim=1))+1e-10
        norm_abs2=torch.sqrt(torch.sum(target*target,dim=1))+1e-10
        aa=sum2/norm_abs1/norm_abs2
        aa[aa<-1]=-1
        aa[aa>1]=1
        spectralmap=torch.acos(aa)
        return torch.mean(spectralmap)
     
     
     
class AWCA(nn.Module):
    def __init__(self, channel=31):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, 1, bias=False),
#            nn.PReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h*w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=16, dimension=2, sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // reduction

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        f = self.count_cov_second(theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y


