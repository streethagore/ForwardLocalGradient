import torch
from torch import nn
from model.auxnet import get_aux_constructor
from model.utils import Block
from args import AuxiliaryLinearArgs


class SmallBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        if downsample:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


def small_net_block(block_n: int, in_channel: int, n_class: int, aux_loss=None):
    if block_n < 5:
        auxnet = get_aux_constructor(aux_loss.auxnet)
    elif block_n == 5:
        auxnet_args = AuxiliaryLinearArgs(pool_size=1)
        auxnet = get_aux_constructor(auxnet_args)
    else:
        raise "Small net only contains 6 blocks."

    if block_n == 0:
        in_c, out_c = in_channel, 64
    elif block_n == 1:
        in_c, out_c = 64, 64
    elif block_n == 2:
        in_c, out_c = 64, 128
    elif block_n == 3:
        in_c, out_c = 128, 128
    elif block_n == 4:
        in_c, out_c = 128, 256
    elif block_n == 5:
        in_c, out_c = 256, 256

    return Block(model=SmallBlock(in_channel=in_c, out_channel=out_c), aux_net=auxnet(out_c, n_class))


class SmallNet(nn.Module):
    def __init__(self, in_channel, n_class, aux_loss=None):
        super().__init__()
        self.blocks = nn.ModuleList([small_net_block(k, in_channel, n_class, aux_loss) for k in range(6)])

    def forward(self, x: torch.Tensor):
        for k in range(len(self.blocks)):
            x = self.blocks[k](x)
        return self.blocks[-1].auxnet(x)

    def upto(self, x: torch.Tensor, j: int):
        for b in range(j + 1):
            x = self.blocks[b](x)
        return x
