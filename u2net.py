import torch
import torch.nn as nn
import torch.nn.functional as F

class CONBNRELU(nn.Module):
    '''Conv() -> BatchNorm() -> ReLU() == output'''
    def __init__(self, in_channel, out_channel, dilation=1) -> None:
        super(CONBNRELU, self).__init__()
        # defining basic blocks
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1*dilation, dilation=1*dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x))) 


class RSU7(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, middle_channel=32) -> None:
        super(RSU7, self).__init__()
 
        # helper functions
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x,target.shape[2:], mode="bilinear")

        # ENCODER parts
        self.conv = CONBNRELU(in_channel, out_channel, dilation=1)
        self.conv1 = CONBNRELU(out_channel, middle_channel, dilation=1)
        self.conv2 = CONBNRELU(middle_channel, middle_channel, dilation=1)
        self.conv3 = CONBNRELU(middle_channel, middle_channel, dilation=1)
        self.conv4 = CONBNRELU(middle_channel, middle_channel, dilation=1)
        self.conv5 = CONBNRELU(middle_channel, middle_channel, dilation=1)
        self.conv6 = CONBNRELU(middle_channel, middle_channel, dilation=1)

        ## BOTTLENECK parts
        self.conv7 = CONBNRELU(middle_channel, middle_channel, dilation=2)

        ## DECODER parts
        self.conv6d = CONBNRELU(middle_channel*2, middle_channel, dilation=1)
        self.conv5d = CONBNRELU(middle_channel*2, middle_channel, dilation=1)
        self.conv4d = CONBNRELU(middle_channel*2, middle_channel, dilation=1)
        self.conv3d = CONBNRELU(middle_channel*2, middle_channel, dilation=1)
        self.conv2d = CONBNRELU(middle_channel*2, middle_channel, dilation=1)
        self.conv1d = CONBNRELU(middle_channel*2, out_channel, dilation=1)

    def forward(self, x):
        ### ENCODER
        # eg:- in: 128 x 128 x 3
        xin = self.conv(x)
        # out: 128 x 128 x 3

        ## layer 1
        x1 = self.conv1(xin)
        # out: 128 x 128 x 32
        down = self.pool(x1)
        # out: 64 x 64 x 32

        ## layer 2
        x2 = self.conv2(down)
        # out: 64 x 64 x 32
        down = self.pool(x2)
        # out: 32 x 32 x 32

        ## layer 3
        x3 = self.conv3(down)
        # out: 32 x 32 x 32
        down = self.pool(x3)
        # out: 16 x 16 x 32

        ## layer 4
        x4 = self.conv4(down)
        # out: 16 x 16 x 32
        down = self.pool(x4)
        # out: 8 x 8 x 32

        ## layer 5
        x5 = self.conv5(down)
        # out: 8 x 8 x 32
        down = self.pool(x5)
        # out: 4 x 4 x 32

        ## layer 6
        ## BOTTLENECK
        x6 = self.conv6(down)
        # out: 4 x 4 x 32
        x7 = self.conv7(x6)
        # out: 4 x 4 x 32

        # DECODER
        x6d = self.conv6d(torch.cat([x7, x6], 1)) # when x7(4 x 4 x 32) CONCAT with x6(4 x 4 x 32) on axis 1 => out(4 x 4 x 64)
        # out: 4 x 4 x 32
        up = self.upsampler(x6d, x5)
        # out: 8 x 8 x 32

        x5d = self.conv5d(torch.cat([up, x5], 1))
        # out: 8 x 8 x 32
        up = self.upsampler(x5d, x4)
        # out: 16 x 16 x 32

        x4d = self.conv4d(torch.cat([up, x4], 1))
        # out: 16 x 16 x 32
        up = self.upsampler(x4d, x3)
        # out: 32 x 32 x 32

        x3d = self.conv3d(torch.cat([up, x3], 1))
        # out: 32 x 32 x 32
        up = self.upsampler(x3d, x2)
        # out: 64 x 64 x 32

        x2d = self.conv2d(torch.cat([up, x2], 1))
        # out: 64 x 64 x 32
        up = self.upsampler(x2d, x1)
        # out: 128 x 128 x 32

        x1d = self.conv1d(torch.cat([up, x1], 1))
        # out: 128 x 128 x 32
        
        return x1d + xin
    

class RSU6(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, middle_channel=32) -> None:
        super(RSU6, self).__init__()

         # helper functions
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x,target.shape[2:], mode="bilinear")

        # ENCODER part
        self.conv = CONBNRELU(in_channel, out_channel)
        self.conv1 = CONBNRELU(out_channel, middle_channel)
        self.conv2 = CONBNRELU(middle_channel, middle_channel)
        self.conv3 = CONBNRELU(middle_channel, middle_channel)
        self.conv4 = CONBNRELU(middle_channel, middle_channel)
        self.conv5 = CONBNRELU(middle_channel, middle_channel)

        # BOTTLENECK part
        self.conv6 = CONBNRELU(middle_channel, middle_channel, dilation=2)

        # DECODER part
        self.conv5d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv4d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv3d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv2d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv1d = CONBNRELU(2*middle_channel, out_channel)

    def forward(self, x):
        ## ENCODER
        # in : 128 x 128 x 3
        xin = self.conv(x)
        # out : 128 x 128 x 3

        x1 = self.conv1(xin)
        # out : 128 x 128 x 32
        down = self.pool(x1)
        # out : 64 x 64 x 32

        x2 = self.conv2(down)
        # out : 64 x 64 x 32
        down = self.pool(x2)
        # out : 32 x 32 x 32

        x3 = self.conv3(down)
        # out : 32 x 32 x 32
        down = self.pool(x3)
        # out : 16 x 16 x 32

        x4 = self.conv4(down)
        # out : 16 x 16 x 32
        down = self.pool(x4)
        # out : 8 x 8 x 32

        x5 = self.conv5(down)
        # out : 8 x 8 x 32

        ## BOTTLENECK
        x6 = self.conv6(x5)
        # out : 8 x 8 x 32

        ## DECODER
        x5d = self.conv5d(torch.cat([x6, x5], axis=1))
        # out : 8 x 8 x 32
        up = self.upsampler(x5d, x4)
        # out : 16 x 16 x 32

        x4d = self.conv4d(torch.cat([up, x4], axis=1))
        # out : 16 x 16 x 32
        up = self.upsampler(x4d, x3)
        # out : 32 x 32 x 32

        x3d = self.conv3d(torch.cat([up, x3], axis=1))
        up = self.upsampler(x3d, x2)
        # out : 64 x 64 x 32

        x2d = self.conv2d(torch.cat([up, x2], axis=1))
        up = self.upsampler(x2d, x1)
        # out : 128 x 128 x 32

        x1d = self.conv1d(torch.cat([up, x1], axis=1))
        # out : 128 x 128 x 3

        return x1d + xin


class RSU5(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, middle_channel=32) -> None:
        super(RSU5, self).__init__()

         # helper functions
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x, target.shape[2:], mode="bilinear")

        # ENCODER part
        self.conv = CONBNRELU(in_channel, out_channel)
        self.conv1 = CONBNRELU(out_channel, middle_channel)
        self.conv2 = CONBNRELU(middle_channel, middle_channel)
        self.conv3 = CONBNRELU(middle_channel, middle_channel)
        self.conv4 = CONBNRELU(middle_channel, middle_channel)

        # BOTTLENECK part
        self.conv5 = CONBNRELU(middle_channel, middle_channel, dilation=2)

        # DECODER part
        self.conv4d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv3d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv2d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv1d = CONBNRELU(2*middle_channel, out_channel)

    def forward(self, x):
        ## ENCODER
        # in : 128 x 128 x 3
        xin = self.conv(x)
        # out : 128 x 128 x 3

        x1 = self.conv1(xin)
        # out : 128 x 128 x 32
        down = self.pool(x1)
        # out : 64 x 64 x 32

        x2 = self.conv2(down)
        # out : 64 x 64 x 32
        down = self.pool(x2)
        # out : 32 x 32 x 32

        x3 = self.conv3(down)
        # out : 32 x 32 x 32
        down = self.pool(x3)
        # out : 16 x 16 x 32

        x4 = self.conv4(down)
        # out : 16 x 16 x 32

        ## BOTTLENECK
        x5 = self.conv5(x4)
        # out : 16 x 16 x 32

        ## DECODER
        x4d = self.conv4d(torch.cat([x5, x4], axis=1))
        # out : 16 x 16 x 32
        up = self.upsampler(x4d, x3)
        # out : 32 x 32 x 32

        x3d = self.conv3d(torch.cat([up, x3], axis=1))
        up = self.upsampler(x3d, x2)
        # out : 64 x 64 x 32

        x2d = self.conv2d(torch.cat([up, x2], axis=1))
        up = self.upsampler(x2d, x1)
        # out : 128 x 128 x 32

        x1d = self.conv1d(torch.cat([up, x1], axis=1))
        # out : 128 x 128 x 3

        return x1d + xin
    

class RSU4(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, middle_channel=32) -> None:
        super(RSU4, self).__init__()

         # helper functions
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x,target.shape[2:], mode="bilinear")

        # ENCODER part
        self.conv = CONBNRELU(in_channel, out_channel)
        self.conv1 = CONBNRELU(out_channel, middle_channel)
        self.conv2 = CONBNRELU(middle_channel, middle_channel)
        self.conv3 = CONBNRELU(middle_channel, middle_channel)

        # BOTTLENECK part
        self.conv4 = CONBNRELU(middle_channel, middle_channel, dilation=2)

        # DECODER part
        self.conv3d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv2d = CONBNRELU(2*middle_channel, middle_channel)
        self.conv1d = CONBNRELU(2*middle_channel, out_channel)    

    def forward(self, x):
        ## ENCODER
        # in : 128 x 128 x 3
        xin = self.conv(x)
        # out : 128 x 128 x 3

        x1 = self.conv1(xin)
        # out : 128 x 128 x 32
        down = self.pool(x1)
        # out : 64 x 64 x 32

        x2 = self.conv2(down)
        # out : 64 x 64 x 32
        down = self.pool(x2)
        # out : 32 x 32 x 32

        x3 = self.conv3(down)
        # out : 32 x 32 x 32

        ## BOTTLENECK
        x4 = self.conv4(x3)
        # out : 32 x 32 x 32

        ## DECODER
        x3d = self.conv3d(torch.cat([x4, x3], axis=1))
        up = self.upsampler(x3d, x2)
        # out : 64 x 64 x 32

        x2d = self.conv2d(torch.cat([up, x2], axis=1))
        up = self.upsampler(x2d, x1)
        # out : 128 x 128 x 32

        x1d = self.conv1d(torch.cat([up, x1], axis=1))
        # out : 128 x 128 x 3

        return x1d + xin


class RSU4F(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, middle_channel=32) -> None:
        super(RSU4F, self).__init__()

         # helper functions
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x,target.shape[2:], mode="bilinear")

        # ENCODER part
        self.conv = CONBNRELU(in_channel, out_channel)
        self.conv1 = CONBNRELU(out_channel, middle_channel)
        self.conv2 = CONBNRELU(middle_channel, middle_channel, dilation=2)
        self.conv3 = CONBNRELU(middle_channel, middle_channel, dilation=4)

        # BOTTLENECK part
        self.conv4 = CONBNRELU(middle_channel, middle_channel, dilation=8)

        # DECODER part
        self.conv3d = CONBNRELU(2*middle_channel, middle_channel, dilation=4)
        self.conv2d = CONBNRELU(2*middle_channel, middle_channel, dilation=2)
        self.conv1d = CONBNRELU(2*middle_channel, out_channel) 

    def forward(self, x):
        ## ENCODER
        # in : 128 x 128 x 3
        xin = self.conv(x)
        # out : 128 x 128 x 3

        x1 = self.conv1(xin)
        # out : 128 x 128 x 32

        x2 = self.conv2(x1)
        # out : 128 x 128 x 32

        x3 = self.conv3(x2)
        # out : 128 x 128 x 32

        ## BOTTLENECK
        x4 = self.conv4(x3)
        # out : 128 x 128 x 32

        ## DECODER
        x3d = self.conv3d(torch.cat([x4, x3], axis=1))
        # out : 128 x 128 x 32

        x2d = self.conv2d(torch.cat([x3d, x2], axis=1))
        # out : 128 x 128 x 32

        x1d = self.conv1d(torch.cat([x2d, x1], axis=1))
        # out : 128 x 128 x 3

        return x1d + xin
    

########### U2NET ######################
class U2NET(nn.Module):
    def __init__(self, in_channel=3, out_channel=3) -> None:
        super(U2NET, self).__init__()
        # helper
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upsampler = lambda x, target: F.interpolate(x,target.shape[2:], mode="bilinear")

        # ENCODER
        self.en_1 = RSU7(in_channel=in_channel, out_channel=64, middle_channel=32)
        self.en_2 = RSU6(in_channel=64, out_channel=128, middle_channel=32)
        self.en_3 = RSU5(in_channel=128, out_channel=256, middle_channel=64)
        self.en_4 = RSU4(in_channel=256, out_channel=512, middle_channel=128)
        self.en_5 = RSU4F(in_channel=512, out_channel=512, middle_channel=256)

        # BOTTLENECK
        self.en_6 = RSU4F(in_channel=512, out_channel=512, middle_channel=256)

        # DECODER
        self.de_5 = RSU4F(in_channel=1024, out_channel=512, middle_channel=256) 
        self.de_4 = RSU4(in_channel=1024, out_channel=256, middle_channel=128)
        self.de_3 = RSU5(in_channel=512, out_channel=128, middle_channel=64)
        self.de_2 = RSU6(in_channel=256, out_channel=64, middle_channel=32)
        self.de_1 = RSU7(in_channel=128, out_channel=64, middle_channel=16)

        # side connection
        self.convS6 = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=3, padding=1)
        self.convS5 = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=3, padding=1)
        self.convS4 = nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=3, padding=1)
        self.convS3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=3, padding=1)
        self.convS2 = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=3, padding=1)
        self.convS1 = nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=3, padding=1)

        self.out = nn.Conv2d(6*out_channel,out_channel, kernel_size=1)

    def forward(self, x):
        ## ENCODER
        # in : 128 x 128 x 3
        e1 = self.en_1(x)
        # out : 128 x 128 x 64
        down = self.pool(e1)
        # out : 64 x 64 x 64

        e2 = self.en_2(down)
        # out : 64 x 64 x 128
        down = self.pool(e2)
        # out : 32 x 32 x 128

        e3 = self.en_3(down)
        # out : 32 x 32 x 256
        down = self.pool(e3)
        # out : 16 x 16 x 256

        e4 = self.en_4(down)
        # out : 16 x 16 x 512
        down = self.pool(e4)
        # out : 8 x 8 x 512

        e5 = self.en_5(down)
        # out : 8 x 8 x 512
        down = self.pool(e5)
        # out : 4 x 4 x 512

        ## BOTTLENECK
        e6 = self.en_6(down)
        up =self.upsampler(e6, e5)
        # out : 8 x 8 x 512

        ## DECODER
        d5 = self.de_5(torch.cat([up, e5], axis=1))
        # out : 8 x 8 x 512
        up =self.upsampler(d5, e4)
        # out : 16 x 16 x 512

        d4 = self.de_4(torch.cat([up, e4], axis=1))
        # out : 16 x 16 x 256
        up =self.upsampler(d4, e3)
        # out : 32 x 32 x 256

        d3 = self.de_3(torch.cat([up, e3], axis=1))
        # out : 32 x 32 x 128
        up =self.upsampler(d3, e2)
        # out : 64 x 64 x 128

        d2 = self.de_2(torch.cat([up, e2], axis=1))
        # out : 64 x 64 x 64
        up =self.upsampler(d2, e1)
        # out : 128 x 128 x 64

        d1 = self.de_1(torch.cat([up, e1], axis=1))
        # out : 128 x 128 x 64
        
        # side connections
        s1 = self.convS1(d1)
        # no need to upsampe as it have same size as input

        s2 = self.convS2(d2)
        s2 = self.upsampler(s2, d1)

        s3 = self.convS3(d3)
        s3 = self.upsampler(s3, d1)

        s4 = self.convS4(d4)
        s4 = self.upsampler(s4, d1)

        s5 = self.convS5(d5)
        s5 = self.upsampler(s5, d1)

        s6 = self.convS6(e6)
        s6 = self.upsampler(s6, d1)

        s0 = self.out(torch.cat([s1, s2, s3, s4, s5, s6],1))
        return (F.sigmoid(s0), F.sigmoid(s1), F.sigmoid(s2), F.sigmoid(s3), F.sigmoid(s4), F.sigmoid(s5), F.sigmoid(s6))
