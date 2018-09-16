import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)  # <- make local residual by 'concat' (important!!)
        # print(out.szie())  # print for sanity check
        return out


# Residual Dense Block
class ResDenseBlock(nn.Module):
    '''
    good explaination of how it works:
    https://discuss.pytorch.org/t/resolved-how-to-understand-the-densenet-implementation/3964/4
    '''
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(ResDenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(BasicBlock(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)  # <- key step, nn.Sequential can hold previous output
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# Residual Dense Network
class SRNet(nn.Module):
    def __init__(self, args):
        super(SRNet, self).__init__()
        self.num_channel = args.num_channel
        self.num_dense = args.num_dense
        self.num_feat = args.num_feat
        self.num_RDB = args.num_RDB
        self.scale = args.scale
        self.growth_rate = args.growth_rate

        # F-1
        self.conv1 = nn.Conv2d(self.num_channel, self.num_feat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(self.num_feat, self.num_feat, kernel_size=3, padding=1, bias=True)

        # RDBs
        self.RDBs = []
        for i in range(self.num_RDB):
            tmp_RDB = ResDenseBlock(self.num_feat, self.num_dense, self.growth_rate)
            setattr(self, 'RDB%i' % i, tmp_RDB)  # set layer to the Module (very important)
            self.RDBs.append(tmp_RDB)

        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(self.num_feat*self.num_RDB, self.num_feat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(self.num_feat, self.num_feat, kernel_size=3, padding=1, bias=True)

        # Upsampler
        self.conv_up = nn.Conv2d(self.num_feat, self.num_feat*self.scale*self.scale,
                                 kernel_size=3, padding=1, bias=True)
        self.upsample = nn.PixelShuffle(self.scale)
        self.conv3 = nn.Conv2d(self.num_feat, self.num_channel, kernel_size=3, padding=1, bias=True)

    def forward(self, x):

        # shallow
        F_ = self.conv1(x)
        F_N = self.conv2(F_)

        # RDBs
        F_s = []
        for i in range(self.num_RDB):
            F_N = self.RDBs[i](F_N)
            F_s.append(F_N)

        # GFF
        FF = torch.cat(F_s, 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)

        # DFF
        FDF = FGF + F_

        # upscale
        up = self.conv_up(FDF)
        up = self.upsample(up)
        output = self.conv3(up)

        return output
