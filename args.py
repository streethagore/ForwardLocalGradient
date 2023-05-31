import os
from dataclasses import dataclass
import simple_parsing as sp
import torch.cuda
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable
import hashlib
import warnings

datasets = ['cifar10', 'cifar100', 'imagenet', 'mnist', 'fashion-mnist', 'k-mnist', 'svhn', 'imagenet32', 'imagenette']
data_augmentation = ['standard', 'none']
architectures = ['resnet18-8b', 'resnet18-16b', 'resnet18-2b', 'small-net']
aux_type = ['mlp', 'cnn', 'linear', 'predsim']
weight_init = ['xavier-uniform', 'xavier-normal', 'kaiming-uniform', 'kaiming-normal', 'normal', 'uniform', 'ones',
               'zeros', None]
algorithms = ['std', 'localonly', 'avg-weight-fast', 'generic']
schedulers = ['steplr', 'plateau', 'onecycle', 'linear']


# ===== Datasets =====
@dataclass
class DatasetArgs(Serializable):
    dataset: str = sp.field(default='cifar10', choices=datasets)
    num_train_sample: int = 0
    num_test_sample: int = 0
    image_size: int = 0
    augmentation: str = sp.field(default='standard', choices=data_augmentation)
    path: str = ''


# ===== Model Specifications =====
@dataclass
class ModelArgs(Serializable):
    arch: str = sp.field(default='resnet18-8b', choices=architectures)
    downsample: str = sp.field(default='conv', choices=['conv', 'avg-pool'])
    widen_factor: float = 1.
    num_block: int = 0
    weight_init: str = sp.field(default='kaiming-uniform', choices=weight_init)


# ===== AuxNet Types Specifications =====
@dataclass
class AuxiliaryLinearArgs(Serializable):
    aux_type: str = 'linear'
    pool_size: int = 2
    h_channel: int = 0  # not used


@dataclass
class AuxiliaryMLPArgs(Serializable):
    aux_type: str = 'mlp'
    pool_size: int = 2
    h_channel: int = 1024
    n_layers: int = 3


@dataclass
class AuxiliaryCNNArgs(Serializable):
    aux_type: str = 'cnn'
    pool_size: int = 2
    h_channel: int = 32
    n_layers: int = 3


@dataclass
class AuxiliaryPredSim(Serializable):
    aux_type: str = 'predsim'
    pool_size: int = 0  # not used
    h_channel: int = 0  # not used


# ===== AuxNet Specifications =====
@dataclass
class AuxLossArgs(Serializable):
    auxnet: object = sp.subgroups(
        {'linear': AuxiliaryLinearArgs,
         'mlp': AuxiliaryMLPArgs,
         'cnn': AuxiliaryCNNArgs,
         'predsim': AuxiliaryPredSim},
        default='mlp')


# ===== Training Algorithms =====
@dataclass
class AlgorithmArgs(Serializable):
    algorithm: str = sp.field(default='generic', choices=algorithms)
    target: str = sp.field(default='global', choices=['global', 'local', 'sum', 'above'])
    guess: str = sp.field(default='local', choices=['global', 'local', 'sum', 'above', 'random', 'ntk', 'fixed-ntk'])
    space: str = sp.field(default='weight', choices=['weight', 'avg-weight', 'f-weight', 'span-weight', 'sign-weight',
                                                     'random-sign-weight',
                                                     'activation', 'activation-2', 'f-activation', 'span-activation',
                                                     'constrained-weight'])
    n_epoch: int = 101
    noise_type: str = sp.field(default='gaussian', choices=['gaussian', 'rademacher'])
    eps: float = 1e-16


# ===== Optimizers =====
@dataclass
class SGDMomentumArgs(Serializable):
    batchsize: int = 128
    lr: float = 5e-2
    wd: float = 5e-4
    momentum: float = 0.9


@dataclass
class AdamArgs(Serializable):
    batchsize: int = 512
    lr: float = 1e-3
    wd: float = 1e-6


@dataclass
class OptimizerArgs(Serializable):
    optimizer: object = sp.subgroups(
        {'sgd': SGDMomentumArgs, 'adam': AdamArgs},
        default='sgd')


# ===== Learning Rate Schedulers =====
@dataclass
class ConstantLRArgs(Serializable):
    step_size: int = 30
    gamma: float = 0.05


@dataclass
class LinearLRArgs(Serializable):
    start_factor: float = 0.01
    end_factor: float = 10
    total_iters: int = 10 * 50000 / 256


@dataclass
class StepLRArgs(Serializable):
    step_size: int = 30
    gamma: float = 0.2


@dataclass
class OneCycleLRArgs(Serializable):
    step_per_epoch_time_bs: int = 45000
    maxlr: float = 0.1


@dataclass
class ReduceOnPlateauArgs(Serializable):
    factor: float = 0.2
    patience: int = 15


@dataclass
class SchedulerArgs(Serializable):
    scheduler: object = sp.subgroups(
        {'steplr': StepLRArgs, 'plateau': ReduceOnPlateauArgs, 'onecycle': OneCycleLRArgs, 'linear': LinearLRArgs,
         'constant': ConstantLRArgs},
        default='steplr')


# ===== WandB =====
@dataclass
class WandbArgs(Serializable):
    status: bool = sp.field(action='store_true')
    project: str = 'Forward Gradient'
    run_name: str = 'default-name'
    entity: str = 'streethagore'


def print_args(args):
    print()
    print(args.dataset, '\n')
    print(args.model, '\n')
    print(args.aux_loss, '\n')
    print(args.training, '\n')
    print(args.optimizer, '\n')
    print(args.scheduler, '\n')
    print(args.wandb, '\n')


def convert_args(args):
    kwargs = dict()
    for k, v in vars(args).items():
        if isinstance(v, Serializable):
            kwargs.update(convert_args(v))
        else:
            kwargs.update({k: v})
    return kwargs


def get_args():
    parser = ArgumentParser(description='Local loss gradient guess for target gradient approximation')

    parser.add_arguments(DatasetArgs, dest='dataset')
    parser.add_arguments(ModelArgs, dest='model')
    parser.add_arguments(AuxLossArgs, dest='aux_loss')
    parser.add_arguments(AlgorithmArgs, dest='training')
    parser.add_arguments(OptimizerArgs, dest='optimizer')
    parser.add_arguments(SchedulerArgs, dest='scheduler')
    parser.add_arguments(WandbArgs, dest='wandb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--checkpoint_time', type=int, default=0)
    parser.add_argument('--log-freq', type=int, default=39)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--image_size_over_32', action='store_true')
    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--save_model', type=int, default=0)

    args = parser.parse_args()

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.device}')
    else:
        args.device = torch.device('cpu')
        warnings.warn("Using CPU only... This will be slow !\n\n")

    # Default channel values for auxnet
    if args.model.arch == 'resnet18-16b':
        if args.aux_loss.auxnet.aux_type == 'mlp':
            args.aux_loss.auxnet.h_channel = 512
        elif args.aux_loss.auxnet.aux_type == 'cnn':
            args.aux_loss.auxnet.h_channel = 16

    args.aux_loss.auxnet.h_channel *= args.model.widen_factor  # scale the auxiliary net with the block

    # Histogram folder
    if args.histogram:
        args.histogram_folder = os.path.join('output/histogram', f'{args.model.arch}-{args.aux_loss.auxnet.aux_type}',
                                             f'{args.training.target}-{args.training.guess}-{args.training.space}')
    else:
        args.histogram_folder = None
    del args.histogram

    # Reformat parser members
    if not args.dataset.path:
        args.dataset.path = None
    if args.dataset.augmentation == 'none':
        args.dataset.augmentation = None
    args.optimizer = args.optimizer.optimizer
    args.scheduler = args.scheduler.scheduler
    args.subgroups['optimizer'] = args.subgroups.pop('optimizer.optimizer')
    args.subgroups['scheduler'] = args.subgroups.pop('scheduler.scheduler')

    args.aux_loss.auxnet.algorithm = args.training.algorithm

    return args
