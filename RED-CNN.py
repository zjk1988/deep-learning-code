https://arxiv.org/ftp/arxiv/papers/1702/1702.00288.pdf

class RED(nn.Module):
    def __init__(self,out_c):
        super(Decoder,self).__init__()
        self.c1 = nn.Sequential(
                nn.Conv2d(1,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU()
                )
        self.c2 = nn.Sequential(
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU()
                )
        self.c3 = nn.Sequential(
                nn.Conv2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU()
                )
        self.t1 = nn.Sequential(
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU(),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
#                nn.ReLU()
                )
        self.t2 = nn.Sequential(
                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=0),
                nn.ReLU(),
#                nn.ConvTranspose2d(out_c,out_c,kernel_size=5,padding=0),
#                nn.ReLU()
                )
        self.d1 = nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=0)
        self.relu1 = nn.ReLU()
        self.d2 = nn.ConvTranspose2d(out_c,out_c,kernel_size=5,stride=1,padding=0)
        self.relu2 = nn.ReLU()
        self.d3 = nn.ConvTranspose2d(out_c,1,kernel_size=5,stride=1,padding=0)
        self.relu3 = nn.ReLU()

    def forward(self,x):
        re1 = x.clone()
        out = self.c1(x)
        re2 = out.clone()
        out = self.c2(out)
        re3 = out.clone()
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
net = RED(32)
x = torch.randn(1,1,512,512)
out = net(x)    
out.shape 