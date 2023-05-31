import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import torch.nn as nn
import torch.nn.functional as tf
from time import time


def get_trainer_constructor(args):
    if args.training.algorithm == 'std':
        from trainer.backprop import BackPropTrainer as Trainer
    elif args.training.algorithm == 'localonly':
        from trainer.localonly import LocalOnlyTrainer as Trainer
    elif args.training.algorithm == 'generic':
        from trainer.forward_gradient import ForwardGradientTrainer as Trainer

    else:
        raise NotImplementedError(f"Wrong algorithm option (args.training = {args.training})")
    return Trainer


def get_gradient_per_block(net):
    return [[p.grad.clone() for p in block.block.parameters() if p.requires_grad] for block in net.blocks]


def batch_scalar_product(x: torch.Tensor, y: torch.Tensor):
    assert x.size() == y.size(), f"Tensors must be of same size (x.size() = {x.size()}, y.size() = {y.size()}"
    shape = x.size()
    n = shape[0]
    scalar_prod = torch.bmm(x.view(n, 1, -1), y.view(n, -1, 1)).squeeze()
    return scalar_prod


def per_sample_product(factor, data_batch):
    # WARNING: only works for (N, C, H, W) tensors
    assert factor.ndim == 1 and factor.size(0) == data_batch.size(0)
    ext_shape = [1 for _ in range(data_batch.ndimension())]
    ext_shape[0] = -1
    with torch.no_grad():
        factor = factor.view(*ext_shape)
    normed_data_batch = factor * data_batch
    return normed_data_batch


def random_shuffle(x, nb_classes, device):
    sh = x.shape
    target_n = torch.randint(low=0, high=nb_classes, size=(sh[0],), device=torch.device('cuda', device))
    shuffle_index = (torch.arange(sh[1], device=torch.device('cuda', device))).view((1, sh[1], 1, 1)).repeat(
        (sh[0], 1, sh[2], sh[3]))
    target_s = torch.div(target_n * sh[1], nb_classes, rounding_mode='floor')
    shuffle_index -= target_s.view((sh[0], 1, 1, 1)).repeat((1, sh[1], sh[2], sh[3]))
    return torch.gather(x, 1, shuffle_index % sh[1]), target_n


def set_bn_eval(m):  # Do not log two times for the batchnorm running stats
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = True


@torch.no_grad()
def accuracy(prediction, target):
    if type(prediction) == list:
        prediction = prediction[0]
    with torch.no_grad():
        acc = prediction.argmax(dim=1).eq(target).float().mean()
    return acc


def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size (x.size(0), x.size(0)). '''
    if x.dim() == 4:
        z = x.view(x.size(0), x.size(1), -1)
        x = z.std(dim=2)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


def to_one_hot(y, n_dims=None, device=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1).to(device)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


class PredSimLoss(nn.Module):
    def __init__(self, n_class=10, gpu_ids=0):
        super(PredSimLoss, self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.beta = 0.99
        self.n_class = n_class
        self.device = torch.device(f'cuda:{gpu_ids}') if gpu_ids is not None else torch.device('cpu')

    def forward(self, x, labels):
        if type(x) == list:
            pred_loss = self.crossentropy(x[0], labels)
            Rh = x[1]
            y_onehot = to_one_hot(labels, self.n_class, device=self.device)
            Ry = similarity_matrix(y_onehot).detach()
            sim_loss = tf.mse_loss(Rh, Ry)
            return (1 - self.beta) * pred_loss + self.beta * sim_loss
        # Last layer
        return self.crossentropy(x, labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum_squared = 0
        self.var = 0
        self.sum = 0
        self.count = 0
        self.real_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.var = 0
        self.sum = 0
        self.count = 0
        self.real_count = 0

    def update(self, val, n=1, val_l=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not val_l is None:  # To log variance, not useful in practice
            val_s = torch.square(val_l).sum().item()
            if len(val_l.shape) == 0:
                real_n = 1
            else:
                real_n = val_l.shape[0]
            self.real_count += real_n
            self.sum_squared += val_s
            self.var = self.sum_squared / self.real_count - self.avg ** 2


class CountMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def update(self, n=1):
        self.count += n


class ChronoMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def start(self):
        self.start_time = time()

    def stop(self, n=1):
        self.end_time = time()
        self.duration = self.end_time - self.start_time
