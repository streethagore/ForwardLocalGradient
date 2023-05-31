import torch
from torch import nn
from functools import partial
from model.utils import DenseLayer, Conv1Layer, Conv3Layer
import torch.nn.functional as tf
import math
from trainer.utils import similarity_matrix


def get_aux_constructor(auxnet):
    if auxnet.aux_type == 'linear':
        constructor = AuxiliaryLinear
    elif auxnet.aux_type == 'mlp':
        constructor = AuxiliaryMLP
    elif auxnet.aux_type == 'cnn':
        constructor = AuxiliaryCNN
    elif auxnet.aux_type == 'generic':
        constructor = AuxiliaryGeneric
    elif auxnet.aux_type == 'predsim':
        constructor = AuxiliaryPredSim
    else:
        def _return_none(*args, **kwargs):
            return None

        return _return_none
    return partial(constructor, **vars(auxnet))


class AuxiliaryGeneric(nn.Module):
    def __init__(self, in_channel, n_class, **kwargs):
        super(AuxiliaryGeneric, self).__init__()
        h_channel, n_conv_1, n_conv_3, pool_size, n_mlp = kwargs['h_channel'], kwargs['n_conv_1'], kwargs['n_conv_3'], \
        kwargs['pool_size'], kwargs['n_mlp']
        self.pooling = nn.AvgPool2d(2)
        self.compression = Conv1Layer(in_channel, h_channel)
        self.cnn_1x1 = nn.Sequential(*[Conv1Layer(h_channel, h_channel) for _ in range(n_conv_1)])
        self.cnn_3x3 = nn.Sequential(*[Conv3Layer(h_channel, h_channel) for _ in range(n_conv_3)])
        self.pooling_flatten = nn.Sequential(nn.AdaptiveAvgPool2d(pool_size), nn.Flatten(1, -1))
        self.mlp = nn.Sequential(DenseLayer(h_channel * pool_size ** 2, h_channel * pool_size ** 2))
        self.projection = nn.Linear(h_channel * pool_size ** 2, n_class)

    def forward(self, x: torch.Tensor):
        x = self.pooling(x)
        x = self.compression(x)
        x = self.cnn_1x1(x)
        x = self.cnn_3x3(x)
        x = self.pooling_flatten(x)
        x = self.mlp(x)
        return self.projection(x)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.flatten(2, -1).mean(dim=2)


class AuxiliaryLinear(nn.Module):
    def __init__(self, in_channel, n_class, **kwargs):
        super(AuxiliaryLinear, self).__init__()
        self.pool_size = kwargs['pool_size']
        if self.pool_size > 1:
            pooling = nn.AdaptiveAvgPool2d(self.pool_size)
        else:
            pooling = GlobalAvgPool()

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            pooling,
            nn.Flatten(1, -1),
            nn.Linear(in_channel * self.pool_size ** 2, n_class)
        )
        
    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class AuxiliaryMLP(nn.Module):
    def __init__(self, in_channel=64, n_class=10, **kwargs):
        super(AuxiliaryMLP, self).__init__()
        self.size = kwargs['pool_size']
        self.pooling = nn.AdaptiveAvgPool2d((self.size, self.size))
        self.flatten = nn.Flatten(1, -1)
        h_channel = in_channel if kwargs['h_channel'] == 0 else kwargs['h_channel']
        h_channel = int(h_channel)
        self.mlp = [
            DenseLayer(in_channel * self.size ** 2, h_channel),
        ]
        for k in range(kwargs['n_layers'] - 1):
            self.mlp.append(DenseLayer(h_channel, h_channel))
        self.mlp.append(nn.Linear(h_channel, n_class))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor):
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x


class AuxiliaryCNN(nn.Module):
    def __init__(self, in_channel=64, n_class=10, **kwargs):
        super(AuxiliaryCNN, self).__init__()
        self.pool_size = kwargs['pool_size']
        h_channel = in_channel if kwargs['h_channel'] == 0 else kwargs['h_channel']
        h_channel = int(h_channel)
        self.compression = Conv1Layer(in_channel, h_channel)
        cnn = [
            nn.Sequential(
                nn.Conv2d(h_channel, h_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(h_channel),
                nn.ReLU()
            ) for _ in range(kwargs['n_layers'])
        ]
        self.cnn = nn.Sequential(*cnn)
        self.pooling = nn.AdaptiveAvgPool2d(self.pool_size)
        self.flatten = nn.Flatten(1, -1)
        self.projection = nn.Linear(h_channel * (self.pool_size ** 2), n_class)

    def forward(self, x: torch.Tensor):
        x = self.compression(x)
        x = self.cnn(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x


class AuxiliaryPredSim(nn.Module):
    def __init__(self, in_channel=64, n_class=10, **kwargs):
        super(AuxiliaryPredSim, self).__init__()
        self.size = kwargs['pool_size']
        self.conv_loss = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False)

        self.dim_in_decoder = 2048  # 4096
        self.n_class = n_class

        dim_out = int(2048) // in_channel  # For ResNet 18

        ks_h, ks_w = 1, 1
        dim_out_h, dim_out_w = dim_out, dim_out
        dim_in_decoder = in_channel * dim_out_h * dim_out_w
        while dim_in_decoder > self.dim_in_decoder and ks_h < dim_out:
            ks_h *= 2
            dim_out_h = math.ceil(dim_out / ks_h)
            dim_in_decoder = in_channel * dim_out_h * dim_out_w
            if dim_in_decoder > self.dim_in_decoder:
                ks_w *= 2
                dim_out_w = math.ceil(dim_out / ks_w)
                dim_in_decoder = in_channel * dim_out_h * dim_out_w
        if ks_h > 1 or ks_w > 1:
            pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
            pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
            self.avg_pool = nn.AvgPool2d((ks_h, ks_w), padding=(pad_h, pad_w))

        self.decoder_y = nn.Linear(dim_in_decoder, n_class)
        # self.decoder_y.weight.data.zero_()

    def forward(self, x: torch.Tensor):
        h_loss = self.conv_loss(x)
        Rh = similarity_matrix(h_loss)

        h = self.avg_pool(x)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))

        return [y_hat_local, Rh]


class AuxiliaryProjector(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(AuxiliaryProjector, self).__init__()

        self.out_to_in_proj = nn.Sequential(
            nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, in_size: int):
        x = tf.interpolate(x, size=[in_size, in_size],
                           mode='bilinear', align_corners=True)
        return self.out_to_in_proj(x)
