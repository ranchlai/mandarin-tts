import torch
import torch.nn as nn

class double_res_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,bn=False):
        super(double_res_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),

        )


        self.relu = nn.LeakyReLU(0.1)


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)



        return x3





class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,bn=True):
        super(inconv, self).__init__()



        self.conv = double_res_conv(in_ch, out_ch,bn)

    def forward(self, x):

        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,bn=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_res_conv(in_ch, out_ch,bn)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True,bn=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        self.bilinear = bilinear
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_res_conv(in_ch, out_ch,bn)

    def forward(self, x1, x2):
        if not self.bilinear:
            x1 = self.up(x1)
        else:
            x1 =  nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)


        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1,padding=0)#nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x
    
    
class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            set_trace()
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths
    

class UNet(nn.Module):
    def __init__(self, n_channels=1,
                 scale=1,use_sigmoid=False):
        super(UNet, self).__init__()


        #self.linear = nn.Linear(256,256)
        self.inc = inconv(n_channels, 64//scale)
        self.down1 = down(64//scale, 128//scale)
        self.down2 = down(128//scale, 256//scale)
        self.down3 = down(256//scale, 512//scale)
        self.down4 = down(512//scale, 512//scale)



        self.up1 = up(1024//scale, 256//scale)
        self.up2 = up(512//scale, 128//scale)
        self.up3 = up(256//scale, 64//scale)
        self.up4 = up(128//scale, 32//scale)
        
        self.reduce=outconv(32//scale,1)


    def forward(self,x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.reduce(x)

        return x
