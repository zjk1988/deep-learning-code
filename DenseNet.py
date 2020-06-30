文章https://arxiv.org/pdf/1608.06993.pdf
GitHub：https://github.com/liuzhuang13/DenseNet
参考：https://blog.csdn.net/weixin_41798111/article/details/86494353

"""
dense net 主要由两个部分组成，dense block和连接block的transition层
而dense block根据有无bottle neck也可分为两类Bottleneck和SingleLayer，
和SingleLayer相比Bottleneck只是在3*3的卷积前加了1*1的卷积
transition作用是通过1*1的卷积减少通道数目
"""
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Transition(nn.Module):
    def __init__(self,in_c,o_c):
        super(Transition,self).__init__()
        self.layer = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c,o_c,kernel_size=1,bias=False),
                nn.AvgPool2d(2,2)
                
                )
    def forward(self,x):
        out = self.layer(x)
        return out
class Bottelneck(nn.Module):
    def __init__(self,in_c,o_c):#这里和原版不同
        super(Bottelneck,self).__init__()
        self.layer = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c,o_c,kernel_size=1,bias=False),
                nn.BatchNorm2d(o_c),
                nn.ReLU(),
                nn.Conv2d(o_c,o_c,kernel_size=3,padding=1,bias=False)
                )
    def forward(self,x):
        out = self.layer(x)
        out = torch.cat((x,out),dim=1)#通道上的拼接
        return out
class SingleLayer(nn.Module):
    def __init__(self,in_c,o_c):
        super(SingleLayer,self).__init__()
        self.layer = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c,o_c,kernel_size=3,padding=1,bias=False)        
                )
    def forward(self,x):
        out = self.layer(x)
        out = torch.cat((x,out),dim=1)
        return out
class DenseNet(nn.Module):
    def __init__(self,growth,depth,reduction,nclass,bottleneck):
        super(DenseNet,self).__init__()
        num_dense = (depth-4)//3 #每个block内有多少层，depth总的层数
        if bottleneck:
            num_dense = num_dense//2 #有了bottleneck减半
        num_ch = growth*2
        self.conv1 = nn.Conv2d(3,num_ch,kernel_size=3,padding=1,bias=False)
        self.dense1 = self._make_block(num_ch,growth,num_dense,bottleneck)
        num_ch = num_ch + growth*num_dense
        out_c = int(math.floor((num_ch*reduction)))
        self.trans1 = Transition(num_ch,out_c)

        num_ch = out_c
        self.dense2 = self._make_block(num_ch,growth,num_dense,bottleneck)
        num_ch = num_ch + growth*num_dense
        out_c = int(math.floor((num_ch*reduction)))
        self.trans2 = Transition(num_ch,out_c)

        num_ch = out_c
        self.dense3 = self._make_block(num_ch,growth,num_dense,bottleneck)
        num_ch = num_ch + growth*num_dense
        self.bn1 = nn.BatchNorm2d(num_ch)
        self.fc = nn.Linear(num_ch,nclass)
    def forward(self,x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.bn1(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(out),8))
        out = F.log_softmax(self.fc(out))
        return out
    def _make_block(self,num_ch,growth,num_dense,bottleneck):
        layers = []
        for i in range(int(num_dense)):
            if bottleneck:
                layers.append(Bottelneck(num_ch,growth))
            else:
                layers.append(SingleLayer(num_ch,growth))
            num_ch += growth
        return nn.Sequential(*layers)
net = DenseNet(4,13,0.5,10,True)
x = torch.randn(1,3,48,48)
out = net(x)
out.shape        
        
        
        
        
        
        
        