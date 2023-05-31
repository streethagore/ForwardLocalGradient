import math
import torch
from torch import nn
from functools import partial
from utils.profiler import get_model_profile


def get_model(args):
    if args.model.arch == 'resnet18-8b':
        from model.resnet import ResNet18 as Net
        Net = partial(Net, downsample=args.model.downsample, widen_factor=args.model.widen_factor,
                      image_size_over_32=args.image_size_over_32)
    elif args.model.arch == 'resnet18-16b':
        from model.resnet_no_skip import ResNet18 as Net
        Net = partial(Net, downsample=args.model.downsample, widen_factor=args.model.widen_factor,
                      image_size_over_32=args.image_size_over_32)
    elif args.model.arch == 'resnet18-2b':
        from model.resnet import ResNet18_2blocks as Net
        Net = partial(Net, downsample=args.model.downsample, widen_factor=args.model.widen_factor,
                      image_size_over_32=args.image_size_over_32)
    elif args.model.arch == 'small-net':
        from model.small_net import SmallNet as Net

    net = Net(in_channel=args.dataset.n_channel, n_class=args.dataset.n_class, aux_loss=args.aux_loss)
    return net


class MaxMarginProjection(nn.Module):
    def __init__(self, n_class: int = 10):
        super(MaxMarginProjection, self).__init__()
        matrix = torch.eye(n_class - 1, n_class)
        matrix[-1, -1] = -1
        for k in range(2, n_class):
            p_sub_k = math.sqrt(1 - 1 / (k ** 2)) * matrix[-(k - 1):, -k:]

            p_k = torch.zeros(k, k + 1)
            p_k[0, 0] = 1
            p_k[0, 1:] = 1 / k
            p_k[1:, 1:] = p_sub_k

            matrix[-k:, -(k + 1):] = p_k
        self.matrix = torch.nn.Parameter(matrix, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return torch.matmul(self.matrix.T, x.T).T


class ClosedFormProjection(nn.Module):
    def __init__(self, in_features: int, n_class: int = 10):
        super(ClosedFormProjection, self).__init__()
        self.compression = nn.Linear(in_features, n_class - 1)
        self.projection = MaxMarginProjection(n_class)

    def forward(self, x: torch.Tensor):
        x = self.compression(x)
        x = self.projection(x)
        return x


class Conv1Layer(nn.Module):
    def __init__(self, in_chanel: int, out_channel: int):
        super(Conv1Layer, self).__init__()
        self.in_channel = in_chanel
        self.out_channel = out_channel
        # Linear transform
        self.conv = nn.Conv2d(in_chanel, out_channel, 1, 1, 0, bias=False)
        # Normalization
        self.norm = nn.BatchNorm2d(out_channel)
        # Activation
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class Conv3Layer(nn.Module):
    def __init__(self, in_chanel: int, out_channel: int):
        super(Conv3Layer, self).__init__()
        self.in_channel = in_chanel
        self.out_channel = out_channel
        # Linear transform
        self.conv = nn.Conv2d(in_chanel, out_channel, 3, 1, 1, bias=False)
        # Normalization
        self.norm = nn.BatchNorm2d(out_channel)
        # Activation
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DenseLayer(nn.Module):
    def __init__(self, in_chanel: int, out_channel: int):
        super(DenseLayer, self).__init__()
        self.in_channel = in_chanel
        self.out_channel = out_channel
        # Linear transform
        self.linear = nn.Linear(in_chanel, out_channel, bias=False)
        # Normalization
        self.norm = nn.BatchNorm1d(out_channel)
        # Activation
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.norm(x)
        return self.act(x)


class Block(nn.Module):
    def __init__(self, model: nn.Module, aux_net: nn.Module):
        super(Block, self).__init__()
        self.block = model
        self.auxnet = aux_net
        self.buffer = None
        self.max_buffer_size = None
        self.identity = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.identity(self.block(x))

    def upto(self, x: torch.Tensor):
        return self.auxnet(x)


def init_weights(m, mode: str = 'kaiming-uniform'):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if mode == 'xavier-uniform':
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'xavier-normal':
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'kaiming-uniform':
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'kaiming-normal':
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'uniform':
            torch.nn.init.uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'normal':
            torch.nn.init.normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif mode == 'ones':
            torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(1.)
        elif mode == 'zeros':
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.)
        elif mode is None:
            pass
        else:
            raise NotImplementedError(f"Wrong weight initialization parameter {mode}")


def get_resnet_blocks_flops_ratio_8b(model):
    flops_ratio = []

    b = 0
    input_shape = (1, 3, 32, 32)
    output_shape = (1, 64, 32, 32)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 1
    input_shape = (1, 64, 32, 32)
    output_shape = (1, 64, 32, 32)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 2
    input_shape = (1, 64, 32, 32)
    output_shape = (1, 128, 16, 16)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 3
    input_shape = (1, 128, 16, 16)
    output_shape = (1, 128, 16, 16)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 4
    input_shape = (1, 128, 16, 16)
    output_shape = (1, 256, 8, 8)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 5
    input_shape = (1, 256, 8, 8)
    output_shape = (1, 256, 8, 8)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 6
    input_shape = (1, 256, 8, 8)
    output_shape = (1, 512, 4, 4)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 7
    input_shape = (1, 512, 4, 8)
    output_shape = (1, 512, 8, 8)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    return flops_ratio


def get_resnet_blocks_flops_ratio_16b(model):
    flops_ratio = []

    b = 0
    input_shape = (1, 3, 32, 32)
    output_shape = (1, 64, 32, 32)
    flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False, detailed=False,
                                    as_string=False)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    for b in range(1, 4):
        input_shape = (1, 64, 32, 32)
        output_shape = (1, 64, 32, 32)
        main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                             detailed=False, as_string=False)
        aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                            detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 4
    input_shape = (1, 64, 32, 32)
    output_shape = (1, 128, 16, 16)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    for b in range(5, 8):
        input_shape = (1, 128, 16, 16)
        output_shape = (1, 128, 16, 16)
        main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                             detailed=False, as_string=False)
        aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                            detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 8
    input_shape = (1, 128, 16, 16)
    output_shape = (1, 256, 8, 8)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    for b in range(9, 12):
        input_shape = (1, 256, 8, 8)
        output_shape = (1, 256, 8, 8)
        main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                             detailed=False, as_string=False)
        aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                            detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    b = 12
    input_shape = (1, 256, 8, 8)
    output_shape = (1, 512, 4, 4)
    main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                         detailed=False, as_string=False)
    aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                        detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    for b in range(13, 16):
        input_shape = (1, 512, 4, 4)
        output_shape = (1, 512, 4, 4)
        main_flops, _, _ = get_model_profile(model=model.blocks[b], input_shape=input_shape, print_profile=False,
                                             detailed=False, as_string=False)
        aux_flops, _, _ = get_model_profile(model=model.blocks[b].auxnet, input_shape=output_shape, print_profile=False,
                                            detailed=False, as_string=False)
    flops_ratio.append(aux_flops / main_flops)

    return flops_ratio
