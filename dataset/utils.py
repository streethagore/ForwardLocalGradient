import os
import torch
import torchvision.transforms as tf


def get_dataset(args):
    """
    Prepare the proper data loader given the argument from the parsed command line arguments.
    :param args: parsed command line arguments
    :return: train_loader, test_loader
    """
    if args.dataset == 'cifar10':
        from torchvision.datasets.cifar import CIFAR10

        if args.augmentation == 'standard':
            transform_train = tf.Compose([
                tf.RandomCrop(32, padding=4),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
            ])
        else:
            transform_train = tf.Compose([
                tf.ToTensor(),
                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
            ])

        transform_test = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
        ])

        root_dir = args.path if args.path is not None else 'data'
        train_data = CIFAR10(root=root_dir, train=True, transform=transform_train, download=True)
        test_data = CIFAR10(root=root_dir, train=False, transform=transform_test, download=True)
        n_class = 10
        n_channel = 3

    elif args.dataset == 'cifar100':
        from torchvision.datasets.cifar import CIFAR100

        if args.augmentation == 'standard':
            transform_train = tf.Compose([
                tf.RandomCrop(32, padding=4),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = tf.Compose([
                tf.ToTensor(),
                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        root_dir = args.path if args.path is not None else 'data'
        train_data = CIFAR100(root=root_dir, train=True, transform=transform_train, download=True)
        test_data = CIFAR100(root=root_dir, train=False, transform=transform_test, download=True)
        n_class = 100
        n_channel = 3

    elif args.dataset == 'mnist':
        from torchvision.datasets.mnist import MNIST
        root_dir = args.path if args.path is not None else 'data'
        train_data = MNIST(root=root_dir, train=True, transform=tf.ToTensor(), download=True)
        test_data = MNIST(root=root_dir, train=False, transform=tf.ToTensor(), download=True)
        n_class = 10
        n_channel = 1

    elif args.dataset == 'k-mnist':
        from torchvision.datasets import KMNIST
        root_dir = args.path if args.path is not None else 'data'
        train_data = KMNIST(root=root_dir, train=True, transform=tf.ToTensor(), download=True)
        test_data = KMNIST(root=root_dir, train=False, transform=tf.ToTensor(), download=True)
        n_class = 10
        n_channel = 1

    elif args.dataset == 'fashion-mnist':
        from torchvision.datasets.mnist import FashionMNIST
        root_dir = args.path if args.path is not None else 'data'
        train_data = FashionMNIST(root=root_dir, train=True, transform=tf.ToTensor(), download=True)
        test_data = FashionMNIST(root=root_dir, train=False, transform=tf.ToTensor(), download=True)
        n_class = 10
        n_channel = 1

    elif args.dataset == 'svhn':
        from torchvision.datasets import SVHN
        root_dir = args.path if args.path is not None else 'data/'
        train_data = SVHN(root=root_dir, split='train', transform=tf.ToTensor(), download=True)
        test_data = SVHN(root=root_dir, split='test', transform=tf.ToTensor(), download=True)
        n_class = 10
        n_channel = 3

    elif args.dataset == 'imagenette':
        from torchvision.datasets import ImageFolder

        if args.augmentation == 'standard':
            transform_train = tf.Compose([
                tf.RandomResizedCrop(args.image_size),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])
        else:
            transform_train = tf.Compose([
                tf.Resize(args.image_size),
                tf.CenterCrop(args.image_size),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])

        transform_test = tf.Compose([
            tf.Resize(args.image_size),
            tf.CenterCrop(args.image_size),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

        root_dir = args.path if args.path is not None else 'data/imagenette2'
        train_folder = os.path.join(root_dir, 'train')
        val_folder = os.path.join(root_dir, 'val')
        train_data = ImageFolder(root=train_folder, transform=transform_train)
        test_data = ImageFolder(root=val_folder, transform=transform_test)
        n_class = 10
        n_channel = 3

    elif args.dataset == 'imagenet':
        from torchvision.datasets import ImageFolder

        if args.augmentation == 'standard':
            transform_train = tf.Compose([
                tf.RandomResizedCrop(args.image_size),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])
        else:
            transform_train = tf.Compose([
                tf.Resize(args.image_size),
                tf.CenterCrop(args.image_size),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])

        transform_test = tf.Compose([
            tf.Resize(args.image_size),
            tf.CenterCrop(args.image_size),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

        root_dir = args.path if args.path is not None else 'data/imagenet'
        train_folder = os.path.join(root_dir, 'train')
        val_folder = os.path.join(root_dir, 'val')
        train_data = ImageFolder(root=train_folder, transform=transform_train)
        test_data = ImageFolder(root=val_folder, transform=transform_test)
        n_class = 1000
        n_channel = 3

    elif args.dataset == 'imagenet32':
        # Data loading code
        from dataset.imagenet32 import Imagenet32

        if args.augmentation == 'standard':
            transform_train = tf.Compose([
                tf.RandomResizedCrop(args.image_size),
                tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])
        else:
            transform_train = tf.Compose([
                tf.Resize(args.image_size),
                tf.CenterCrop(args.image_size),
                tf.ToTensor(),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            ])

        transform_test = tf.Compose([
            tf.Resize(args.image_size),
            tf.CenterCrop(args.image_size),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

        root_dir = args.path if args.path is not None else 'data/imagenet32'
        train_data = Imagenet32(root_dir, train=True, transform=transform_train)
        test_data = Imagenet32(root_dir, train=False, transform=transform_test)

        n_class = 1000
        n_channel = 3

    else:
        raise ValueError(f"Wrong dataset argument ({args.dataset})")

    if args.num_train_sample > 0:
        tot_example = len(train_data.targets)
        num_example = min(args.num_train_sample, tot_example)
        ids = torch.multinomial(torch.ones(tot_example), num_example)
        train_data.data = train_data.data[ids]
        train_data.targets = [train_data.targets[k] for k in ids]

    if args.num_test_sample > 0:
        tot_example = len(test_data.targets)
        num_test_example = min(args.num_test_sample, tot_example)
        ids = torch.multinomial(torch.ones(tot_example), num_test_example)
        test_data.data = test_data.data[ids]
        test_data.targets = [test_data.targets[k] for k in ids]

    setattr(args, 'n_channel', n_channel)
    setattr(args, 'n_class', n_class)

    return train_data, test_data
