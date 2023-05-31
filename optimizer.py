import torch
from functools import partial


def get_optimizer(net, args):
    if args.subgroups['optimizer'] == 'sgd':
        lr = args.optimizer.lr
        wd = args.optimizer.wd
        momentum = args.optimizer.momentum
        optim_constructor = partial(torch.optim.SGD, lr=lr, weight_decay=wd, momentum=momentum)

    elif args.subgroups['optimizer'] == 'adam':
        lr = args.optimizer.lr
        wd = args.optimizer.wd
        optim_constructor = partial(torch.optim.Adam, lr=lr, weight_decay=wd)

    else:
        raise 'Wrong optimizer'

    if args.training.algorithm == 'fg-tgt-e2e-guess-localgrad-activity-map':
        parameters = [
            {
                "params": [p for n, p in net.named_parameters() if not "lin_proj" in n],
                "weight_decay": wd,
            },
            {
                "params": [p for n, p in net.named_parameters() if "lin_proj" in n],
                "weight_decay": 0.,
            }]
    else:
        parameters = net.parameters()
    optimizer = optim_constructor(parameters)

    return optimizer
