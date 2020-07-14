##https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf
#import torch
#import torch.nn as nn
#class mask_p(nn.Module):
#    def __init__(self,out_c):
#        super(mask_p,self).__init__()
#        self.c1 = nn.Sequential(
#                nn.Conv2d(1,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
#                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
#                nn.AvgPool2d(2,2)
#                )
#        self.c2 = nn.Sequential(
#                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
#                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
#                nn.AvgPool2d(2,2)
#                )
#        self.c3 = nn.Sequential(
#                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU()
#                )
#        self.t1 = nn.Sequential(
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
##                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
##                nn.ReLU()
#                )
#        self.t2 = nn.Sequential(
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
#                nn.ReLU(),
##                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
##                nn.ReLU()
#                )
#        self.d1 = nn.Sequential(
#                nn.Upsample(scale_factor=2,mode='nearest'),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2)
#                )
#        self.relu1 = nn.ReLU()
#        self.d2 = nn.Sequential(
#                nn.Upsample(scale_factor=2,mode='nearest'),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2)
#                )
#        self.relu2 = nn.ReLU()
#        self.d3 = nn.ConvTranspose2d(out_c,1,kernel_size=5,stride=1,padding=2)
#        self.relu3 = nn.ReLU()
#        self.reup3 = nn.Upsample(scale_factor=2,mode='nearest')
#        self.reup2 = nn.Upsample(scale_factor=2,mode='nearest')
#
#    def forward(self,x):
#        re1 = x.clone()
#        out = self.c1(x)
#        re2 = self.reup2(out.clone())
#        out = self.c2(out)
#        re3 = self.reup3(out.clone())
#        out = self.c3(out)
#        
#        out = self.d1(out)
#        print(out.shape,re3.shape)
#        out = out + re3
#        out = self.relu1(out)
#        out = self.t1(out)
#        out = self.d2(out)
#        out = out + re2
#        out = self.relu2(out)
#        out = self.t2(out)
#        out = self.d3(out)
#        out = out + re1
#        out = self.relu3(out)
#        
#        return out
#net = mask_p(16)
#x = torch.randn(1,1,64,64)
#out = net(x)    
#out.shape 




import torch
import torch.nn as nn
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
mask_train = torch.ones(300,1,256,256)
test_x = torch.randn(100,1,256,256)
test_y = torch.randn(100,1,256,256)
mask_test = torch.ones(100,1,256,256)
class mask_p(nn.Module):
    def __init__(self,in_c,out_c):
        super(mask_p,self).__init__()
        self.c1 = nn.Sequential(
                nn.Conv2d(in_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
                nn.AvgPool2d(2,2)
                )
        self.c3 = nn.Sequential(
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU()
                )
        self.t1 = nn.Sequential(
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
#                nn.ReLU()
                )
        self.t2 = nn.Sequential(
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2),
                nn.ReLU(),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
#                nn.ReLU()
                )
        self.d1 = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2)
                )
        self.relu1 = nn.ReLU()
        self.d2 = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=2)
                )
        self.relu2 = nn.ReLU()
        self.d3 = nn.ConvTranspose2d(out_c,1,kernel_size=5,stride=1,padding=2)
        self.relu3 = nn.ReLU()
        self.reup3 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reup2 = nn.Upsample(scale_factor=2,mode='nearest')
        
        self.c1_mask = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c2_mask = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c1x1_1 = nn.Conv2d(out_c+1,out_c,kernel_size=1,stride=1)
        self.c1x1_2 = nn.Conv2d(out_c+1,out_c,kernel_size=1,stride=1)
        

    def forward(self,x,mask):
        re1 = x.clone()
        out = self.c1(x)
        masked1 = self.c1_mask(mask)
        out = torch.cat((masked1,out),dim=1)#ch=2
        out = self.c1x1_1(out)#ch=1
        re2 = self.reup2(out.clone())
        out = self.c2(out)
        masked2 = self.c2_mask(masked1)
        out = torch.cat((masked2,out),dim=1)#ch=2
        out = self.c1x1_2(out)#ch=1
        re3 = self.reup3(out.clone())
        out = self.c3(out)
        
        out = self.d1(out)
        out = out + re3
        out = self.relu1(out)
        out = self.t1(out)
        out = self.d2(out)
        out = out + re2
        out = self.relu2(out)
        out = self.t2(out)
        out = self.d3(out)
        out = out + re1
        out = self.relu3(out)
        
        return out
#net = mask_p(1,16)
#x = torch.randn(1,1,64,64)
#mask = torch.ones(1,1,64,64)
#out = net(x,mask)    
#print(out.shape)

net=mask_p(1,16).to(device)

import torch.utils.data as Data
train_dataset = Data.TensorDataset(train_x, train_y, mask_train)
train_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
test_dataset = Data.TensorDataset(test_x, test_y, mask_test)
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
  for i,(batch_x,batch_y,batch_mask) in enumerate(train_loader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_mask = batch_mask.to(device)
    #forward pass
    out = net(batch_x,batch_mask)
    loss = criterion(out,batch_y)

    #backward and optimize
    opt.zero_grad()
    loss.backward()
    opt.step()

    #print sth
    if (i+1)%10==0:
        print('epoch %d, step %d, loss %f'%(epoch,i+1,loss.item()))


#root = 'E:\yjs\代码\mental mask'
#
#import os
#from PIL import Image
#import numpy as np
# 
#def resize(imgPath,savePath):
#    files = os.listdir(imgPath)
#    files.sort()
#    print('****************')
#    print('input :',imgPath)
#    print('start...')
#    for file in files:
#        print(imgPath+'\\'+file)
#        fileType = os.path.splitext(file)
#        if fileType[1] == '.png':
#            new_png = Image.open(imgPath+'\\'+file) #打开图片
#            #new_png = new_png.resize((20, 20),Image.ANTIALIAS) #改变图片大小
#            matrix = 255-np.asarray(new_png) #图像转矩阵 并反色
#            new_png = Image.fromarray(matrix) #矩阵转图像
#            new_png.save(savePath+'/'+file) #保存图片
#    print('down!')
#    print('****************')
# 
#
## 待处理图片地址
#dataPath = 'E:\\yjs\\代码\\mental mask'
##保存图片的地址
#savePath = 'E:\\yjs\\代码\\ms'
#resize(dataPath,savePath)









