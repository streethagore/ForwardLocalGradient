import torch
from torch import nn, Tensor
from model.auxnet import get_aux_constructor, AuxiliaryProjector
from model.utils import Block
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


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #    raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


def resnet18_block(block_n: int, in_channel: int, n_class: int, aux_loss=None, downsample: str = 'conv', widen_factor=1,
                   image_size_over_32=False):
    if block_n > 7:
        raise "Only 8 blocks in resnet"

    if block_n != 7:
        auxnet = get_aux_constructor(aux_loss.auxnet)
    else:
        auxnet_args = AuxiliaryLinearArgs(pool_size=1)
        auxnet = get_aux_constructor(auxnet_args)

    # in_c = 3 if not do_permute_loss else 4
    if block_n == 0: in_c = in_channel
    if block_n in [1, 2]: in_c = 64
    if block_n in [3, 4]: in_c = 128
    if block_n in [5, 6]: in_c = 256
    if block_n == 7: in_c = 512

    if block_n in [0, 1]: out_c = 64
    if block_n in [2, 3]: out_c = 128
    if block_n in [4, 5]: out_c = 256
    if block_n in [6, 7]: out_c = 512

    if block_n != 0: in_c = int(in_c * widen_factor)
    out_c = int(out_c * widen_factor)

    if block_n == 0:
        if image_size_over_32:
            block_module = nn.Sequential(
                nn.Conv2d(in_channel, out_c, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                BasicBlock(out_c, out_c, 1, None, base_width=64), )
        else:
            block_module = nn.Sequential(
                nn.Conv2d(in_channel, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=False),
                BasicBlock(out_c, out_c, 1, None, base_width=64),
            )
    else:
        if block_n in [2, 4, 6]:
            if downsample == 'conv':
                downsample = nn.Sequential(conv1x1(in_c, out_c, 2), nn.BatchNorm2d(out_c))
            elif downsample == 'avg-pool':
                downsample = nn.Sequential(nn.AvgPool2d(2, 2), conv1x1(in_c, out_c, 1), nn.BatchNorm2d(out_c))
            else:
                raise ValueError(f'Wrong downsampling option ({downsample})')
            block_module = BasicBlock(in_c, out_c, 2, downsample=downsample, base_width=widen_factor)
        else:
            block_module = BasicBlock(in_c, out_c, 1, None, base_width=64)

    b = Block(
        block_module,
        auxnet(out_c, n_class)
    )

    if aux_loss.auxnet.algorithm == 'fg-tgt-e2e-guess-localgrad-activity-map':
        b.identity.lin_proj = nn.Linear(out_c, out_c, bias=False)

    return b


def resnet18_2block(block_n: int, in_channel: int, n_class: int, aux_loss=None, downsample: str = 'conv',
                    widen_factor=1, image_size_over_32=False):
    if block_n > 1:
        raise "Only 8 blocks in resnet 2blocks (used for testing the alignment of deep auxiliary nets)"

    auxnet_l = get_aux_constructor(aux_loss.auxnet)

    # in_c = 3 if not do_permute_loss else 4
    if block_n == 0:
        in_c = in_channel
        out_c = 64
    else:
        in_c = 64
        out_c = 128

    if block_n == 0:
        block_module = nn.Sequential(
            nn.Conv2d(in_channel, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            BasicBlock(out_c, out_c, 1, None, base_width=64),
            BasicBlock(out_c, out_c, 1, None, base_width=64)
        )

        auxnet = nn.Sequential(nn.Sequential(
            BasicBlock(out_c, 2 * out_c, 2,
                       downsample=nn.Sequential(conv1x1(out_c, 2 * out_c, 2), nn.BatchNorm2d(2 * out_c)),
                       base_width=64),
            BasicBlock(2 * out_c, 2 * out_c, 1, None, base_width=64)
        ),
            auxnet_l(2 * out_c, n_class))
    else:
        block_module = nn.Sequential(
            BasicBlock(in_c, out_c, 2, downsample=nn.Sequential(conv1x1(in_c, out_c, 2), nn.BatchNorm2d(out_c)),
                       base_width=64),
            BasicBlock(out_c, out_c, 1, None, base_width=64)
        )

        auxnet = nn.Sequential(auxnet_l(out_c, n_class))

    b = Block(
        block_module,
        auxnet
    )

    return b


class ResNet18(nn.Module):
    def __init__(self, in_channel: int, n_class: int = 10, aux_loss=None, downsample: str = 'conv', widen_factor=1,
                 image_size_over_32=False):
        super(ResNet18, self).__init__()
        self.blocks = nn.ModuleList(
            [resnet18_block(k, in_channel, n_class, aux_loss, downsample, widen_factor, image_size_over_32) for k in
             range(8)])

    def forward(self, x: torch.Tensor):
        for k in range(len(self.blocks)):
            x = self.blocks[k](x)
        return self.blocks[-1].auxnet(x)

    def upto(self, x: torch.Tensor, j: int):
        for b in range(j + 1):
            x = self.blocks[b](x)
        return x

    def fromtoend(self, x: torch.Tensor, j: int):
        for b in range(j, len(self.blocks)):
            x = self.blocks[b](x)
        return self.blocks[-1].auxnet(x)


class ResNet18_2blocks(nn.Module):
    def __init__(self, in_channel: int, n_class: int = 10, aux_loss=None, downsample: str = 'conv', widen_factor=1,
                 image_size_over_32=False):
        super(ResNet18_2blocks, self).__init__()
        self.blocks = nn.ModuleList(
            [resnet18_2block(k, in_channel, n_class, aux_loss, downsample, widen_factor, image_size_over_32) for k in
             range(2)])
        # Same initialization parameters for the first block auxnet and second block main module. Only difference in the small auxnet

    def forward(self, x: torch.Tensor):
        for k in range(len(self.blocks)):
            x = self.blocks[k](x)
        return self.blocks[-1].auxnet(x)

    def upto(self, x: torch.Tensor, j: int):
        for b in range(j + 1):
            x = self.blocks[b](x)
        return x
