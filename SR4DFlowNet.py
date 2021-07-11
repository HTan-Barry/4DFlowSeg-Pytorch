import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
import torch.nn.functional as F
from monai.networks.layers import Conv, Act
from monai.networks.blocks import Upsample
from monai.networks.layers.simplelayers import Reshape
from monai.utils import alias, export
from collections import OrderedDict

class douConv(nn.Module):
    def __init__(self,
                 kernel_size: Union[Sequence[int], int] = 1,
                 in_channel: int = 3,
                 out_channel: int = 64,
                 stride: int = 1,
                 padding: int = 0,
                 padding_mode: str = 'replicate',
                 low: bool = True,
                 ):
        conv_type = Conv[Conv.CONV, 3]
        act_type = Act[Act.PRELU]
        self.layers = OrderedDict()
        super(douConv, self).__init__()

        if low:
            in_feature = in_channel
            out_feature = out_channel
            for i in range(2):
                self.layers["conv{}".format(i)] = conv_type(
                    in_channels=in_feature,
                    out_channels=out_feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode
                )
                in_feature = out_channel
                self.layers[f"act{i}"] = act_type(
                    num_parameters=out_feature
                )

        else:

            in_feature = in_channel
            out_feature = in_channel
            for i in range(2):
                self.layers[f"conv{i}"] = conv_type(
                    in_channels=in_feature,
                    out_channels=out_feature,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode
                    )
                self.layers[f"act{i}"] = act_type(
                    num_parameters=out_feature
                )
                out_feature = out_channel

    def forward(self, x):
        out = x
        for i in range(2):
            out = self.layers[f"conv{i}"](out)
            out = self.layers[f"act{i}"](out)
        return out


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 ):
        super(ResBlock, self).__init__()
        conv_type = Conv[Conv.CONV, 3]
        act_type = Act[Act.LEAKYRELU]
        self.conv = conv_type(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              padding_mode='replicate'
                              )
        self.Relu = act_type()


    def forward(self, x):
        out = self.conv(x)
        out = self.Relu(out)
        out = self.conv(out)
        out += x
        out = self.Relu(out)
        return out


class FlowNet(nn.Module):
    def __init__(self,
                 res_increase: int = 2,
                 num_low_res: int = 8,
                 num_hi_res: int = 4,
                 last_act = 'tanh'
                 ):
        super(FlowNet, self).__init__()
        self.res_increase = res_increase
        self.k3Conv_1_pc = douConv(kernel_size=3, in_channel=3, out_channel=64, padding=1)
        self.k3Conv_1_phase = douConv(kernel_size=3, in_channel=3, out_channel=64, padding=1)
        self.k1Conv = douConv(kernel_size=1, in_channel=64, out_channel=64, padding=0)
        if last_act == 'tanh':
            self.k3Conv_2_x = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.PRELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            Act[Act.TANH](),
                                            )
            self.k3Conv_2_y = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.PRELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            Act[Act.TANH](),
                                            )
            self.k3Conv_2_z = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.PRELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            Act[Act.TANH](),
                                            )
            self.k3Conv_2_y = douConv(kernel_size=3, in_channel=64, out_channel=1, padding=1, low=False)
            self.k3Conv_2_z = douConv(kernel_size=3, in_channel=64, out_channel=1, padding=1, low=False)
        elif last_act == 'linear':
            self.k3Conv_2_x = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.RELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            )
            self.k3Conv_2_y = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.RELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            )
            self.k3Conv_2_z = nn.Sequential(Conv[Conv.CONV, 3](in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1,
                                                              padding_mode='replicate'
                                                              ),
                                            Act[Act.RELU](),
                                            Conv[Conv.CONV, 3](in_channels=64,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1,
                                                               padding_mode='replicate'
                                                               ),
                                            )
        else:
            raise ValueError('wrong type of activation')
        self.res_low = nn.Sequential(*[ResBlock(in_channels=64)] * num_low_res)
        self.res_high = nn.Sequential(*[ResBlock(in_channels=64)] * num_hi_res)
        self.up = Upsample(dimensions=3, in_channels=64, scale_factor=2, mode='deconv', interp_mode='bilinear')


    def forward(self, pc, phase):
        pc = self.k3Conv_1_pc(pc)
        phase = self.k3Conv_1_phase(phase)
        # out = torch.cat((pc, phase), dim=1)
        out = pc + phase
        out = self.k1Conv(out)
        out = self.res_low(out)
        out = self.up(out)
        vx = self.k3Conv_2_x(out)
        vy = self.k3Conv_2_y(out)
        vz = self.k3Conv_2_z(out)
        return torch.cat((vx, vy, vz), dim=1)


    """
    Old version of pre-processing, will move to dataset setting
    def forward(self, u, v, w, u_mag, v_mag, w_mag, low_resblock=8, hi_resblock=4, channel_nr=64):
        channel_nr = 64

        speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 2
        pcmr = mag * speed

        phase = torch.cat([u, v, w], dim=1)
        pc = torch.cat([pcmr, mag, speed], dim=1)

        # Conv for separate layer
    """
