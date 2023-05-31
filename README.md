# Can Forward Gradient Match Backpropagation?

<img src="./utils/forward%20gradient%20estimation%20figure.png" width="200">

## Introduction

This is the repository for the code for the ICML 2023 submission "Can Forward Gradient Match Backpropagation?"

The presented code can be used to train our different algorithms for image classification on CIFAR10 and Imagenet32.

The main.py function is used to launch the training, using the arguments given in args.
All the different trainers are available in the trainer subfolder.

## Running instructions

To reproduce the experiments of our articles, use the following command:
```
python -u main.py --algo [ALGO] --arch [ARCH] --auxnet [AUX] --auxnet.h_channel [N_HIDDEN] --n_epoch 100  --optimizer.lr 5e-2 --optimizer.wd 5e-4 --eps_norm 1e-14
```

Two models are available: ['resnet18','resnet18-16b'] for a 8 or 16-blocks divided ResNet18.

Three auxiliary nets are available: ['cnn', 'mlp', 'linear']. (We use a hidden channel size (h_channel) of 32 for 8 blocks and 16 for 16 blocks, to keep FLOPS consistently under 10\%.).

All algorithms presented are available, using directly global target and guess (end-to-end training 'std'), local target and guess ('localonly'),
and more generally using a guess and a target (and updating with activaty or weight perturbations).
For exemple 'fg-tgt-e2e-guess-localgrad-weight' uses forward gradient with Global (end-to-end gradient) Target, and Local (local loss gradient) Guess, with weight perturbation.


For example, to use a training of a CNN Local Guess of Global Target, for a Resnet split in 8 blocks on CIFAR10, use:
```
python -u main.py --algo 'fg-tgt-e2e-guess-localgrad-weight' --arch resnet18 --auxnet cnn   --auxnet.h_channel 32 --n_epoch 100  --optimizer.lr 5e-2 --optimizer.wd 5e-4 --eps_norm 1e-14
```


