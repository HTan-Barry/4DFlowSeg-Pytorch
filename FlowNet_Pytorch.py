import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union


class douConv(nn.Module):
    def __init__(self,
                 kernel_size: Union[Sequence[int]] = [3, 3],
                 in_channel: int = 3,
                 out_channel: Union[Sequence[int]] = [64, 64],
                 stride: int = 1,
                 padding: Union[Sequence[int]] = [1, 1],
                 padding_mode: str = 'replicate',
                 act: str = 'relu',
                 ):
        super(douConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel,
                               out_channels=out_channel[0],
                               kernel_size=kernel_size[0],
                               stride=stride,
                               padding=padding[0],
                               padding_mode=padding_mode)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channel[0],
                               out_channels=out_channel[1],
                               kernel_size=kernel_size[1],
                               stride=stride,
                               padding=padding[1],
                               padding_mode=padding_mode)
        self.act = act
        if act == 'relu':
            self.act2 = nn.ReLU()
        elif act == 'tanh' or act is None:
            self.act2 = nn.Tanh()
        elif act == 'LeakyReLU':
            self.act2 = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        if not self.act == 'linear':
            out = self.act2(out)
        return out


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 64,
                 padding_mode: str = 'replicate', ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode=padding_mode)
        self.leakyrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = out + x
        out = self.leakyrelu(out)
        return out


class FlowNet(nn.Module):
    def __init__(self,
                 res_increase: int = 2,
                 num_low_res: int = 8,
                 num_hi_res: int = 4,
                 last_act='tanh'
                 ):
        super(FlowNet, self).__init__()
        self.conv_vol = douConv(kernel_size=[3, 3],
                                in_channel=3,
                                out_channel=[64, 64],
                                )
        self.conv_mag = douConv(kernel_size=[3, 3],
                                in_channel=3,
                                out_channel=[64, 64],
                                )
        self.conv_lr = douConv(kernel_size=[1, 3],
                               in_channel=128,
                               out_channel=[64, 64],
                               padding=[0, 1]
                               )
        self.res_low = nn.Sequential(*[ResBlock(in_channels=64)] * num_low_res)
        self.res_high = nn.Sequential(*[ResBlock(in_channels=64)] * num_hi_res)
        self.up = nn.Upsample(scale_factor=res_increase, mode='trilinear')
        self.conv_hr_u = douConv(kernel_size=[3, 3],
                                 in_channel=64,
                                 out_channel=[64, 1],
                                 padding=[1, 1],
                                 act=last_act
                                 )
        self.conv_hr_v = douConv(kernel_size=[3, 3],
                                 in_channel=64,
                                 out_channel=[64, 1],
                                 padding=[1, 1],
                                 act=last_act
                                 )
        self.conv_hr_w = douConv(kernel_size=[3, 3],
                                 in_channel=64,
                                 out_channel=[64, 1],
                                 padding=[1, 1],
                                 act=last_act
                                 )

    def forward(self, input):
        pc = input[:, 0:3]
        vol = input[:, 3:]
        pc = self.conv_mag(pc)
        vol = self.conv_vol(vol)
        out = torch.cat((pc, vol), dim=1)
        out = self.conv_lr(out)
        out = self.res_low(out)
        out = self.up(out)
        out = self.res_high(out)
        vx = self.conv_hr_u(out)
        vy = self.conv_hr_v(out)
        vz = self.conv_hr_w(out)
        out = torch.cat((vx, vy, vz), dim=1)
        return out
