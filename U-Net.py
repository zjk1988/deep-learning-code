https://arxiv.org/pdf/1505.04597.pdf

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:22:18 2020

@author: Lenovo
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class conv_block(nn.Module):
    def __init__(self,in_c,o_c):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.ReLU(),
                nn.Conv2d(o_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.ReLU()
                )
    def forward(self,x):
        out = self.conv(x)
        return out
class up_conv(nn.Module):
    def __init__(self,in_c,o_c):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_c,o_c,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(o_c),
                nn.ReLU()
                )
    def forward(self,x):
        out = self.up(x)
        return out
class U_net(nn.Module):
    def __init__(self,in_c=3,o_c=1):
        super(U_net,self).__init__()
        
        n1 = 64
        filters = [n1,n1*2,n1*4,n1*8,n1*16]
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.maxpool4 = nn.MaxPool2d(2,2)
        
        self.conv1 = conv_block(in_c,filters[0])
        self.conv2 = conv_block(filters[0],filters[1])
        self.conv3 = conv_block(filters[1],filters[2])
        self.conv4 = conv_block(filters[2],filters[3])
        self.conv5 = conv_block(filters[3],filters[4])
        
        self.Up5 = up_conv(filters[4],filters[3])
        self.up_conv5 = conv_block(filters[4],filters[3])
        
        self.Up4 = up_conv(filters[3],filters[2])
        self.up_conv4 = conv_block(filters[3],filters[2])

        self.Up3 = up_conv(filters[2],filters[1])
        self.up_conv3 = conv_block(filters[2],filters[1])
        
        self.Up2 = up_conv(filters[1],filters[0])
        self.up_conv2 = conv_block(filters[1],filters[0])
        
        self.conv = nn.Conv2d(filters[0],o_c,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        e1 = self.conv1(x)
        e2 = self.maxpool1(e1)
        e2 = self.conv2(e2)
        e3 = self.maxpool2(e2)
        e3 = self.conv3(e3)
        e4 = self.maxpool3(e3)
        e4 = self.conv4(e4)
        e5 = self.maxpool4(e4)
        e5 = self.conv5(e5)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4,d5),dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2,d3),dim=1)
        d3 = self.up_conv3(d3)
       
        d2 = self.Up2(d3)
        d2 = torch.cat((e1,d2),dim=1)
        d2 = self.up_conv2(d2)
    
        out = self.conv(d2)
        return out
unet = U_net()           
x = torch.randn(1,3,512,512)
out = unet(x)
out.shape                    

torch.Size([1, 1, 512, 512])