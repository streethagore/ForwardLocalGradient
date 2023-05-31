import torch
from functools import partial


def get_scheduler(optimizer, args):
    if args.subgroups['scheduler'] == 'constant':
        step_size = 1000
        gamma = 1.0
        scheduler = partial(torch.optim.lr_scheduler.StepLR,
                            step_size=step_size,
                            gamma=gamma)

    elif args.subgroups['scheduler'] == 'linear':
        start_factor = args.scheduler.start_factor
        end_factor = args.scheduler.end_factor
        total_iters = args.scheduler.total_iters
        scheduler = partial(torch.optim.lr_scheduler.LinearLR,
                            start_factor=start_factor,
                            end_factor=end_factor,
                            total_iters=total_iters)

    elif args.subgroups['scheduler'] == 'steplr':
        gamma = args.scheduler.gamma
        step_size = args.scheduler.step_size
        scheduler = partial(torch.optim.lr_scheduler.StepLR,
                            step_size=step_size,
                            gamma=gamma)

    elif args.subgroups['scheduler'] == 'multistep':
        milestones = args.scheduler.milestones
        gamma = args.scheduler.gamma
        scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                            milestones=milestones,
                            gamma=gamma)

    elif args.subgroups['scheduler'] == 'onecycle':
        t_max = args.scheduler.t_max
        scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                            T_max=t_max)

    elif args.subgroups['scheduler'] == 'plateau':
        factor = args.scheduler.factor
        patience = args.scheduler.patience
        scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau,
                            factor=factor,
                            patience=patience)

    else:
        raise NotImplementedError(
            f"Unrecognized scheduler (args.subgroups['scheduler'] = {args.subgroups['scheduler']})"
        )

    if type(optimizer) == list:
        lr_scheduler = [scheduler(optim) for optim in optimizer]
    else:
        lr_scheduler = scheduler(optimizer)

    return lr_scheduler
