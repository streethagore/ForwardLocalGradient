import torch
from torch import nn, Tensor
from model.auxnet import get_aux_constructor, AuxiliaryProjector
from model.utils import Block
from utils.tools import print_debug
from args import AuxiliaryLinearArgs


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resnet18_block(block_n: int, in_channel: int, n_class: int, aux_loss=None, downsample: str = 'conv', widen_factor=1, image_size_over_32=False):
    if block_n > 15:
        raise "Only 16 blocks in resnet"

    if block_n != 15:
        auxnet = get_aux_constructor(aux_loss.auxnet)
    else:
        auxnet_args = AuxiliaryLinearArgs(pool_size=1)
        auxnet = get_aux_constructor(auxnet_args)

    # in_c = 3 if not do_permute_loss else 4
    if block_n == 0: in_c = in_channel
    if block_n in [1, 2, 3, 4]: in_c = 64
    if block_n in [5, 6, 7, 8]: in_c = 128
    if block_n in [9, 10, 11, 12]: in_c = 256
    if block_n in [13, 14, 15]: in_c = 512

    if block_n in [0, 1, 2, 3]: out_c = 64
    if block_n in [4, 5, 6, 7]: out_c = 128
    if block_n in [8, 9, 10, 11]: out_c = 256
    if block_n in [12, 13, 14, 15]: out_c = 512


    if block_n != 0: in_c = int(in_c*widen_factor)
    out_c = int(out_c*widen_factor)

    if block_n == 0:
        if image_size_over_32:
            block_module = nn.Sequential(
                nn.Conv2d(in_channel, out_c, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            block_module = nn.Sequential(
                conv3x3(in_c, out_c, 1),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False),
                conv3x3(out_c, out_c, 1),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False)
            )
    else:
        if block_n in [4, 8, 12]:
            block_module = nn.Sequential(
                conv3x3(in_c, out_c, 2),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False)
            )
        else:
            block_module = nn.Sequential(
                conv3x3(in_c, out_c, 1),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False)
            )

    b = Block(
        block_module,
        auxnet(out_c, n_class)
    )

    # if aux_loss.auxnet.algorithm == 'fg-tgt-e2e-guess-localgrad-activity-map':
        # b.identity.lin_proj = nn.Linear(out_c, out_c, bias=False)

    return b


class ResNet18(nn.Module):
    def __init__(self, in_channel: int, n_class: int = 10, aux_loss=None, downsample: str = 'conv', widen_factor=1, image_size_over_32=False):
        super(ResNet18, self).__init__()
        self.blocks = nn.ModuleList([resnet18_block(k, in_channel, n_class, aux_loss, downsample, widen_factor, image_size_over_32) for k in range(16)])

    def forward(self, x: torch.Tensor):
        for k in range(len(self.blocks)):
            x = self.blocks[k](x)
        return self.blocks[-1].auxnet(x)

    def upto(self, x: torch.Tensor, j: int):
        for b in range(j + 1):
            x = self.blocks[b](x)
        return x
