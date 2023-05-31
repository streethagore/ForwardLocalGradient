import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils.tools import Chrono
from trainer.utils import AverageMeter, accuracy


def train_epoch(dataloader, net, criterion, optimizer, target='global', guess='local', space='weight', device=None,
                args=None):

    net.train()
    net.to(device)
    for block in net.blocks:
        block.auxnet.loss = AverageMeter()
        block.auxnet.accs = AverageMeter()

    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x = data
        for block in net.blocks:
            x = block(x.detach())
            pred = block.auxnet(x)
            loss = criterion(pred, target)
            accs = accuracy(pred, target)
            loss.backward()

            block.auxnet.loss.update(loss.item())
            block.auxnet.accs.update(accs.item())

        optimizer.step()

    return net.blocks[-1].auxnet.loss.avg, net.blocks[-1].auxnet.accs.avg


class LocalOnlyTrainer:
    """This trainer class handles standard backprop training."""

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 args):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Create log folders
        os.makedirs(args.logdir, exist_ok=True)
        # best model
        self.best_check_dir = os.path.join(args.logdir, 'best_model')
        os.makedirs(self.best_check_dir, exist_ok=True)
        # prompt output
        self.wandb = args.wandb.status

    def train(self,
              train_loader: DataLoader,
              test_loader: DataLoader,
              criterion: nn.Module,
              n_epoch: int,
              args):

        # Training loop
        best_acc = 0.0
        train_chrono = Chrono()
        epoch_chrono = Chrono()
        
        train_chrono.start()
        for epoch in range(n_epoch):
            epoch_chrono.start()
            
            self.train_epoch(train_loader, criterion, epoch, args)
            loss, acc = self.validate(test_loader, criterion, args)
            
            # prompt output
            self.print_test(loss, acc, epoch, n_epoch)
            epoch_chrono.stop('Epoch duration:')
            train_chrono.stop('Total Time:')

            # WandB logging
            if self.wandb:
                log_index = (epoch + 1) * (len(train_loader)) - 1
                self.wandb_log(loss, acc, log_index, train=False)
                wandb.log({'Duration': train_chrono.duration}, step=log_index)

            if args.subgroups['scheduler'] == 'plateau':
                self.lr_scheduler.step(loss)
            elif args.subgroups['scheduler'] != 'onecycle' and args.training.algorithm != 'lr-finder':
                self.lr_scheduler.step()

            # save best model
            if acc[-1].avg > best_acc:
                self.checkpoint(epoch, best=True)
                best_acc = acc[-1].avg
                if args.wandb.status:
                    wandb.run.summary['Best Test Accuracy'] = best_acc
        
        # Logging total training duration
        train_chrono.stop('Total Training Time:')
        if args.wandb.status:
            wandb.run.summary['Cumulated Time'] = train_chrono.duration

    def train_epoch(self,
                    data_loader: DataLoader,
                    criterion: nn.Module,
                    epoch: int,
                    args):
        # Set training mode
        self.model.train()

        # defining the device used for training
        device = torch.device(f'cuda:{args.gpu_ids}') if args.gpu_ids is not None else torch.device('cpu')
        self.model.to(device=device)
        criterion.to(device=device)

        # trackers
        loss_meter = [AverageMeter() for _ in range(len(self.model.blocks))]
        acc_meter = [AverageMeter() for _ in range(len(self.model.blocks))]
        time_meter = AverageMeter()
        chrono = Chrono()
        
        chrono.start()
        for k, (data, target) in enumerate(data_loader):
            self.optimizer.zero_grad()
            data, target = data.to(device=device), target.to(device=device)
            representation = data
            for b, block in enumerate(self.model.blocks):
                representation = representation.detach()
                representation = block(representation)
                prediction = block.auxnet(representation)
                loss = criterion(prediction, target)
                loss.backward()
                
                # update trackers
                loss_meter[b].update(loss.item())
                acc_meter[b].update(accuracy(prediction, target).item())
            
            self.optimizer.step()
            
            if args.subgroups['scheduler'] == 'onecyclelr':
                self.lr_scheduler.step()
            
            # update trackers
            chrono.stop()
            time_meter.update(chrono.duration)

            # log
            if (k % args.log_freq) == (args.log_freq - 1):
                # define log index
                log_index = len(data_loader) * epoch + k

                # WandB log
                if self.wandb:
                    self.wandb_log(loss_meter, acc_meter, log_index, train=True)

                # prompt output
                self.print_train(loss_meter[-1].avg, acc_meter[-1].avg, k, len(data_loader), epoch, args.training.n_epoch)
                print('Average Mini-Batch Time:', time_meter.avg, 'secs', '\n')

                # reset trackers
                for b in range(len(self.model.blocks)):
                    loss_meter[b].reset()
                    acc_meter[b].reset()
                time_meter.reset()
            chrono.reset()
            chrono.start()
    
    @torch.no_grad()
    def validate(self,
                 data_loader: DataLoader,
                 criterion: nn.Module,
                 args):
        # Set training mode
        self.model.eval()

        # defining the device used for training
        device = torch.device(f'cuda:{args.gpu_ids}') if args.gpu_ids is not None else torch.device('cpu')
        self.model.to(device=device)
        criterion.to(device=device)

        # trackers
        loss_meter = [AverageMeter() for _ in range(len(self.model.blocks))]
        acc_meter = [AverageMeter() for _ in range(len(self.model.blocks))]

        for k, (data, target) in enumerate(data_loader):
            data, target = data.to(device=device), target.to(device=device)
            representation = data
            for b, block in enumerate(self.model.blocks):
                representation = block(representation)
                prediction = block.auxnet(representation)
                loss = criterion(prediction, target)

                # update trackers
                loss_meter[b].update(loss.item(), n=target.size(0))
                acc_meter[b].update(accuracy(prediction, target).item(), n=target.size(0))

        return loss_meter, acc_meter

    @staticmethod
    def print_train(loss, acc, batch_idx, num_batch, epoch, n_epoch):
        print(
            f'[Epoch: {epoch}/{n_epoch}] [Batch: {batch_idx}/{num_batch}] -- Loss {loss: .3f} -- Acc {acc * 100: .3f}%')

    def print_test(self, loss, acc, epoch, n_epoch):
        print()
        print(f'========== Epoch [{epoch}/{n_epoch}] -- Validation ==========', '\n')
        for b in range(len(self.model.blocks)):
            print(f'    Block {b}: Loss {loss[b].avg: .3f} -- Acc {acc[b].avg * 100: .3f}%')
        print()
        print(f'Global Model: Loss {loss[-1].avg: .3f} -- Acc {acc[-1].avg * 100: .3f}%', '\n')

    def wandb_log(self, loss_meter, acc_meter, step, train=True):
        if train:
            for b in range(len(self.model.blocks)):
                wandb.log({f'Block {b}/Train Loss': loss_meter[b].avg for b in range(len(self.model.blocks))},
                          step=step)
                wandb.log({f'Block {b}/Train Accuracy': acc_meter[b].avg for b in range(len(self.model.blocks))},
                          step=step)
            wandb.log({'Train/Loss': loss_meter[-1].avg, 'Train/Accuracy': acc_meter[-1].avg}, step=step)
            wandb.log({'Train/Learning Rate': self.lr_scheduler.get_last_lr()[0]}, step=step)

        else:
            for b in range(len(self.model.blocks)):
                wandb.log({f'Block {b}/Test Loss': loss_meter[b].avg for b in range(len(self.model.blocks))}, step=step)
                wandb.log({f'Block {b}/Test Accuracy': acc_meter[b].avg for b in range(len(self.model.blocks))},
                          step=step)
            wandb.log({'Test/Loss': loss_meter[-1].avg, 'Test/Accuracy': acc_meter[-1].avg}, step=step)

    def checkpoint(self, epoch, best=False, **kwargs):
        self.model.cpu()
        # define checkpoint dictionary
        check_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

        # save checkpoint dictionary

        check_path = os.path.join(self.best_check_dir, f'best_model')
        torch.save(check_dict, check_path)
