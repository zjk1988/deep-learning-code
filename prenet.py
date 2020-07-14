# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:24:35 2020

@author: Lenovo
"""


import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
#device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cup')
#hyper parameters
num_epoch = 5
lr = 0.0001
num_class = 10
batch_size = 2


train_x = torch.randn(300,1,256,256)
train_y = torch.randn(300,1,256,256)
train_z = torch.randn(300,1,256,256)
test_x = torch.randn(100,1,256,256)
test_y = torch.randn(100,1,256,256)
test_z = torch.randn(100,1,256,256)
class resblock(nn.Module):
    def __init__(self,out_c=16):
        super(resblock,self).__init__()
        self.l1 = nn.Sequential(
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
                )
    def forward(self,x):
        out = self.l1(x)
        out = out+x
        return out
class prenet(nn.Module):
    def __init__(self,out_c=16):
        super(prenet,self).__init__()
        self.l1 = nn.Sequential(
                nn.Conv2d(1,out_c,kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
                )
        self.res1 = resblock()
        self.res2 = resblock()
        self.res3 = resblock()
        self.res4 = resblock()
        self.last = nn.Conv2d(out_c,1,kernel_size=1,stride=1)
    def forward(self,x):
        out = self.l1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.last(out)
        return out  
    
    
net=prenet().to(device)

import torch.utils.data as Data
train_dataset = Data.TensorDataset(train_x, train_y, train_z)
train_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
test_dataset = Data.TensorDataset(test_x, test_y, test_z)
train_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

#loss and opt
criterion = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(),lr=lr)

#train step
for epoch in range(num_epoch):
  for i,(batch_x,batch_y,batch_z) in enumerate(train_loader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_z = batch_z.to(device)

    #forward pass
    out = net(batch_x)
    out = out + batch_z
    loss = criterion(out,batch_y)

    #backward and optimize
    opt.zero_grad()
    loss.backward()
    opt.step()

    #print sth
    if (i+1)%10==0:
        print('epoch %d, step %d, loss %f'%(epoch,i+1,loss.item()))



#
#import torch.utils.data as Data
#x = torch.linspace(1, 10, 10)
#y = torch.linspace(10, 1, 10)
## 把数据放在数据库中
#torch_dataset = Data.TensorDataset(x, y)
#loader = Data.DataLoader(
#    # 从数据库中每次抽出batch size个样本
#    dataset=torch_dataset,
#    batch_size=BATCH_SIZE,
#    shuffle=True,
#    num_workers=2,
#)
#
#
#
#
#
#
#
#"""
#    批训练，把数据变成一小批一小批数据进行训练。
#    DataLoader就是用来包装所使用的数据，每次抛出一批数据
#"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
z = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y, z)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y,batch_z) in enumerate(loader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{},batch_z:{}".format(step, batch_x, batch_y,batch_z))

show_batch()
