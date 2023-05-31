import os
import torch
from functools import partial
from trainer.utils import AverageMeter, accuracy, set_bn_train, set_bn_eval
from trainer.hooks import set_net_hook
from trainer.gradients import save_auxiliary_gradients, save_gradient, set_gradient, set_auxiliary_gradients, \
    get_gradient_computation_function
from trainer.estimators import get_estimator
from trainer.trackers import reset_tracker, print_tracker, log_perf_to_wandb, print_perf
from tqdm import tqdm
from model.utils import init_weights


def _train_epoch(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                 args=None):
    # target guess and estimator
    compute_target = partial(get_gradient_computation_function(target), dest='target', space=space)
    compute_guess = partial(get_gradient_computation_function(guess), dest='guess', space=space)
    if args.training.guess == 'random':
        compute_guess = partial(compute_guess, noise_type=args.training.noise_type)
    compute_estimator = get_estimator(space)

    # tracker
    net.train()
    net.to(device)
    criterion.to(device)
    reset_tracker(net)

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
        handles = set_net_hook(net, space, 'target')

        compute_target(net, data, target, criterion)

        save_gradient(net.blocks[-1])
        if args.training.target == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()

        # guess gradient
        optimizer.zero_grad()
        net.apply(set_bn_eval)
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                block.auxnet.apply(set_bn_train)
        handles = set_net_hook(net, space, 'guess')

        compute_guess(net, data, target, criterion)

        if guess == 'local':
            save_auxiliary_gradients(net)
        for h in handles:
            h.remove()

        # compute projection
        compute_estimator(net)
        set_gradient(net.blocks[-1])
        if guess in ['fixed-ntk', 'ntk']:
            for k, block in enumerate(net.blocks):
                if k < len(net.blocks) - 1:
                    block.auxnet.zero_grad()
        elif target == 'local' or guess == 'local':
            set_auxiliary_gradients(net)

        optimizer.step()

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg


@torch.no_grad()
def validate(dataloader, net, criterion, device, args):
    net.to(device)
    net.eval()
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        x = data
        for block in net.blocks:
            x = block(x)
            pred = block.auxnet(x)
            loss = criterion(pred, target)
            accs = accuracy(pred, target)
            block.auxnet.loss.update(loss.item())
            block.auxnet.accs.update(accs.item())

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg


def get_train_epoch_function(algorithm):
    if algorithm == 'std':
        from trainer.backprop import train_epoch
        return train_epoch
    elif algorithm == 'localonly':
        from trainer.localonly import train_epoch
        return train_epoch
    elif algorithm == 'avg-weight-fast':
        from trainer.fast_avg_weight_pertub import train_epoch
        return train_epoch
    elif algorithm == 'generic':
        return _train_epoch
    else:
        return None


def train(train_loader, test_loader, net, criterion, optimizer, scheduler, n_epoch, args=None):
    train_epoch = get_train_epoch_function(args.training.algorithm)
    target, guess, space, device = args.training.target, args.training.guess, args.training.space, args.device

    for epoch in range(n_epoch):
        train_loss, train_acc = train_epoch(train_loader, net, criterion, optimizer, target=target, guess=guess,
                                            space=space, device=device, args=args)
        print_perf(net, epoch, train_loss, train_acc, scheduler.get_last_lr()[0], 'train')
        if args is not None and args.wandb.status:
            log_perf_to_wandb(net, epoch, scheduler, 'train')

        test_loss, test_acc = validate(test_loader, net, criterion, device, args)
        print_perf(net, epoch, test_loss, test_acc, scheduler.get_last_lr()[0], 'test')
        if args is not None and args.wandb.status:
            log_perf_to_wandb(net, epoch, scheduler, 'test')

        if args.histogram_folder is not None and not epoch % 20:
            print_tracker(net, space, os.path.join(args.histogram_folder, f'epoch_{epoch}'))

        if args.save_model > 0 and not epoch % args.save_model:
            net.cpu()
            fpath = f'output/{args.model.arch}-{args.aux_loss.auxnet.aux_type}-{args.training.target}-{args.training.guess}-{args.training.space}-epoch_{epoch}.weights'
            torch.save({'args': args, 'weights': net.state_dict()}, fpath)

        scheduler.step()
