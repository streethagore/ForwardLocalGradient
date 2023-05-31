from trainer.utils import AverageMeter, set_bn_train, set_bn_eval
from tqdm import tqdm
from trainer.gradients import save_gradient, set_gradient, get_gradient_computation_function, save_auxiliary_gradients, \
    set_auxiliary_gradients
from trainer.estimators import compute_avg_projection
import torch
from functools import partial
from model.utils import init_weights


def process_gradient(net, guess):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    if guess == 'random':
                        if module.weight.requires_grad:
                            scalar_prod = (module.weight.saved_grad * module.weight.grad).sum()
                            module.weight.grad = scalar_prod * module.weight.grad
                        if module.bias is not None and module.bias.requires_grad:
                            scalar_prod = (module.bias.saved_grad * module.bias.grad).sum()
                            module.bias.grad = scalar_prod * module.bias.grad
                    else:
                        if module.weight.requires_grad:
                            module.weight.grad = compute_avg_projection(module.weight.saved_grad, module.weight.grad)
                        if module.bias is not None and module.bias.requires_grad:
                            module.bias.grad = compute_avg_projection(module.bias.saved_grad, module.bias.grad)


def train_epoch(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                args=None):
    compute_target = partial(get_gradient_computation_function(target), dest='target')
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess')
    if args.training.guess == 'random':
        compute_guess = partial(compute_guess, noise_type=args.training.noise_type)

    net.train()
    net.to(device)
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)

        if guess == 'ntk':
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    block.auxnet.apply(lambda m: init_weights(m, mode=args.model.weight_init))

        # target gradient
        optimizer.zero_grad()
        net.apply(set_bn_train)

        _, _ = compute_target(net, data, target, criterion)

        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                save_gradient(block.block)
        save_gradient(net.blocks[-1])
        if args.training.target == 'local':
            save_auxiliary_gradients(net)

        # guess gradient
        optimizer.zero_grad()
        net.apply(set_bn_eval)
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                block.auxnet.apply(set_bn_train)

        _, _ = compute_guess(net, data, target, criterion)

        if guess == 'local':
            save_auxiliary_gradients(net)

        # compute projection
        process_gradient(net, guess)
        set_gradient(net.blocks[-1])
        if guess in ['fixed-ntk', 'ntk']:
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    block.auxnet.zero_grad()
        elif target == 'local' or guess == 'local':
            set_auxiliary_gradients(net)
        optimizer.step()

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg
