# Dependencies
import os
import torch
import torch.nn as nn
import wandb
import torch.utils.data
import torch.backends.cudnn
import random

# Local import
from args import get_args, print_args, convert_args
from dataset.utils import get_dataset
from model.utils import init_weights, get_model
from scheduler import get_scheduler
from trainer.utils import PredSimLoss
from optimizer import get_optimizer
from trainer.forward_gradient import train


def main_worker(args):
    # ========== Datasets ==========
    train_data, test_data = get_dataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.optimizer.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )
    if args.dataset.dataset == 'imagenette':
        batchsize = args.optimizer.batchsize * 2
    else:
        batchsize = 2048
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )

    # ========== Model ==========
    net = get_model(args)

    # weight initialization
    net.apply(lambda m: init_weights(m, mode=args.model.weight_init))
    if args.training.algorithm == 'fg-tgt-e2e-guess-localgrad-activity-map':
        for b in range(len(net.blocks)):
            nn.init.eye_(net.blocks[b].identity.lin_proj.weight)
            if not net.blocks[b].identity.lin_proj.bias is None:
                nn.init.zeros_(net.blocks[b].identity.lin_proj.bias)

    if args.model.arch == 'resnet18-2b':
        net.blocks[0].auxnet[0].load_state_dict(net.blocks[1].block.state_dict())
        # net.blocks[0].auxnet[1].load_state_dict(net.blocks[1].auxnet[0].state_dict())
    print(net, '\n')

    if args.training.guess == 'random':
        for k, block in enumerate(net.blocks):
            if k < len(net.blocks) - 1:
                if args.training.space == 'activation':
                    block.identity.random_flag = args.training.noise_type

    # ========== Loss ==========
    if args.aux_loss.auxnet.aux_type == 'predsim':
        criterion = PredSimLoss(n_class=args.dataset.n_class, gpu_ids=args.gpu_ids)
    else:
        criterion = nn.CrossEntropyLoss()

    # ========== Optimization ==========
    optimizer = get_optimizer(net, args)
    scheduler = get_scheduler(optimizer, args)

    # GPU optimization
    torch.backends.cudnn.benchmark = True

    # WandB
    if args.wandb.status:
        print("# Init wandb")
        wandb.init(project=args.wandb.project, entity=args.wandb.entity, name=args.wandb.run_name,
                   config=convert_args(args))

    # Training
    train(train_loader, test_loader, net, criterion, optimizer, scheduler, args.training.n_epoch, args)

    # Saving trained model
    if args.save_model:
        net.cpu()
        fpath = f'output/{args.model.arch}-{args.aux_loss.auxnet.aux_type}-{args.training.target}-{args.training.guess}-{args.training.space}.weights'
        torch.save({'args': args, 'weights': net.state_dict()}, fpath)

    if args.wandb.status:
        wandb.finish()
    return None


if __name__ == '__main__':
    import os
    from utils.tools import get_git_revision_hash

    opt = get_args()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.device.index}"
    if opt.wandb.status:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_API_KEY"] = "[WANDB KEY]"

    print()
    # print('Git commit hash -->', get_git_revision_hash(), '\n')
    print_args(opt)

    main_worker(opt)

    print_args(opt)
    # print('Git commit hash -->', get_git_revision_hash(), '\n')
