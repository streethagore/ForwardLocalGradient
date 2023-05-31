import os
import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm


def compute_avg_projection(target, guess, eps=1e-14):
    scalar_prod = (target * guess).sum()
    guess_norm = guess.norm()
    weight = scalar_prod / ((guess_norm ** 2) + eps)
    weighted_grad = weight * guess
    return weighted_grad


def compute_projection(target, guess, eps=1e-14):
    scalar_prod = torch.einsum('n...,n...->n', target, guess)
    guess_norm_2 = torch.einsum('n...,n...->n', guess, guess)
    weight = scalar_prod / (guess_norm_2 + eps)
    weighted_grad = torch.einsum('n,n...->n...', weight, guess)
    return weighted_grad


def compute_span_projection(target, guess):
    n = target.size(0)
    weighted_grad = ((target.view(n, -1) @ guess.view(n, -1).T) @ torch.linalg.pinv(guess.view(n, -1).T)).view(
        guess.shape)
    return weighted_grad


def _create_tracker_dict(x_label, y_label, x, y, eps=1e-14):
    scalar_prod = torch.einsum('n...,n...->n', x, y)
    x_norm = torch.einsum('n...,n...->n', x, x).sqrt()
    y_norm = torch.einsum('n...,n...->n', y, y).sqrt()
    cosine = scalar_prod / (x_norm * y_norm + eps)
    norm_diff = torch.einsum('n...,n...->n', x - y, x - y).sqrt()
    norm_diff_normed = norm_diff / x_norm
    norm_ratio = y_norm / x_norm

    tracks = dict(
        scalar_prod=scalar_prod,
        cosine=cosine,
        norm_diff=norm_diff,
        norm_diff_normed=norm_diff_normed,
        norm_ratio=norm_ratio
    )

    old_keys = list(tracks.keys())
    for key in old_keys:
        new_key = f'{key}_{x_label}_{y_label}'
        tracks[new_key] = tracks.pop(key)
    key = f'{x_label}_norm'
    tracks[key] = x_norm
    key = f'{y_label}_norm'
    tracks[key] = y_norm

    x, y = x.sum(dim=0), y.sum(dim=0)
    scalar_prod = (x * y).sum()
    x_norm = x.norm()
    y_norm = y.norm()
    cosine = scalar_prod / (x_norm * y_norm + eps)
    norm_diff = (x - y).norm()
    norm_diff_normed = norm_diff / x_norm
    norm_ratio = y_norm / x_norm

    tracks_avg = dict(
        scalar_prod=scalar_prod.unsqueeze(0),
        cosine=cosine.unsqueeze(0),
        norm_diff=norm_diff.unsqueeze(0),
        norm_diff_normed=norm_diff_normed.unsqueeze(0),
        norm_ratio=norm_ratio.unsqueeze(0)
    )

    old_keys = list(tracks_avg.keys())
    for key in old_keys:
        new_key = f'{key}_{x_label}_avg_{y_label}_avg'
        tracks_avg[new_key] = tracks_avg.pop(key)
    key = f'{x_label}_avg_norm'
    tracks_avg[key] = x_norm.unsqueeze(0)
    key = f'{y_label}_avg_norm'
    tracks_avg[key] = y_norm.unsqueeze(0)

    tracks.update(tracks_avg)
    return tracks


def weight_perturb_module_tracker(module):
    def udpate_trackers(module, tracker_dict, weight_key):
        for key, value in tracker_dict.items():
            getattr(module, key)[weight_key].append(value)
        return None

    if module.weight.requires_grad:
        target = module.target['weight']
        guess = module.guess['weight']
        proj = compute_projection(target, guess)
        span_proj = compute_span_projection(target, guess)

        target_guess = _create_tracker_dict('target', 'guess', target, guess)
        target_proj = _create_tracker_dict('target', 'proj', target, proj)
        target_span_proj = _create_tracker_dict('target', 'span_proj', target, span_proj)
        guess_proj = _create_tracker_dict('guess', 'proj', guess, proj)
        guess_span_proj = _create_tracker_dict('guess', 'span_proj', guess, span_proj)
        proj_span_proj = _create_tracker_dict('proj', 'span_proj', proj, span_proj)

        udpate_trackers(module, target_guess, 'weight')
        udpate_trackers(module, target_proj, 'weight')
        udpate_trackers(module, target_span_proj, 'weight')
        udpate_trackers(module, guess_proj, 'weight')
        udpate_trackers(module, guess_span_proj, 'weight')
        udpate_trackers(module, proj_span_proj, 'weight')

    if module.bias is not None and module.bias.requires_grad:
        target = module.target['bias']
        guess = module.guess['bias']
        proj = compute_projection(target, guess)
        span_proj = compute_span_projection(target, guess)

        target_guess = _create_tracker_dict('target', 'guess', target, guess)
        target_proj = _create_tracker_dict('target', 'proj', target, proj)
        target_span_proj = _create_tracker_dict('target', 'span_proj', target, span_proj)
        guess_proj = _create_tracker_dict('guess', 'proj', guess, proj)
        guess_span_proj = _create_tracker_dict('guess', 'span_proj', guess, span_proj)
        proj_span_proj = _create_tracker_dict('proj', 'span_proj', proj, span_proj)

        udpate_trackers(module, target_guess, 'bias')
        udpate_trackers(module, target_proj, 'bias')
        udpate_trackers(module, target_span_proj, 'bias')
        udpate_trackers(module, guess_proj, 'bias')
        udpate_trackers(module, guess_span_proj, 'bias')
        udpate_trackers(module, proj_span_proj, 'bias')


def tracker(net, space='weight'):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    weight_perturb_module_tracker(module)


def reset_tracker(net, space='weight'):
    def reset_module_tracker(module, space):
        vector_list = ['target', 'guess', 'proj', 'span_proj']
        tracks = ['scalar_prod', 'cosine', 'norm_diff', 'norm_diff_normed', 'norm_ratio']
        for i in range(len(vector_list)):
            # per-sample quantities
            label = vector_list[i] + '_norm'
            if 'weight' in space:
                setattr(module, label, {'weight': [], 'bias': []})
            elif 'activation' in space:
                setattr(module, label, [])

            for track in tracks:
                for j in range(i + 1, len(vector_list)):
                    label = track + '_' + vector_list[i] + '_' + vector_list[j]
                    if 'weight' in space:
                        setattr(module, label, {'weight': [], 'bias': []})
                    elif 'activation' in space:
                        setattr(module, label, [])

            # average quantities
            label = vector_list[i] + '_avg_norm'
            if 'weight' in space:
                setattr(module, label, {'weight': [], 'bias': []})
            elif 'activation' in space:
                setattr(module, label, [])

            for track in tracks:
                for j in range(i + 1, len(vector_list)):
                    label = track + '_' + vector_list[i] + '_avg_' + vector_list[j] + '_avg'
                    if 'weight' in space:
                        setattr(module, label, {'weight': [], 'bias': []})
                    elif 'activation' in space:
                        setattr(module, label, [])

    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    reset_module_tracker(module, space)
    return None


def print_tracker(net, space, output_root=''):
    os.makedirs(output_root, exist_ok=True)
    for k, block in enumerate(tqdm(net.blocks)):
        if k < len(net.blocks) - 1:
            block_dir = os.path.join(output_root, f'block_{k}')
            os.makedirs(block_dir, exist_ok=True)
            kk = -1
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    kk += 1
                    for track in ['scalar_prod_target_guess', 'target_norm', 'guess_norm', 'cosine_target_guess',
                                  'norm_diff_target_guess', 'norm_diff_normed_target_guess', 'norm_ratio_target_guess',

                                  'scalar_prod_target_proj', 'proj_norm', 'cosine_target_proj',
                                  'norm_diff_target_proj', 'norm_diff_normed_target_proj', 'norm_ratio_target_proj',

                                  'scalar_prod_guess_proj', 'cosine_guess_proj',
                                  'norm_diff_guess_proj', 'norm_diff_normed_guess_proj', 'norm_ratio_guess_proj',

                                  'scalar_prod_target_avg_guess_avg', 'target_avg_norm', 'guess_avg_norm',
                                  'cosine_target_avg_guess_avg', 'norm_diff_target_avg_guess_avg',
                                  'norm_diff_normed_target_avg_guess_avg', 'norm_ratio_target_avg_guess_avg',

                                  'scalar_prod_target_avg_proj_avg', 'proj_avg_norm', 'cosine_target_avg_proj_avg',
                                  'norm_diff_target_avg_proj_avg', 'norm_diff_normed_target_avg_proj_avg',
                                  'norm_ratio_target_avg_proj_avg',

                                  'scalar_prod_guess_avg_proj_avg', 'cosine_guess_avg_proj_avg',
                                  'norm_diff_guess_avg_proj_avg', 'norm_diff_normed_guess_avg_proj_avg',
                                  'norm_ratio_guess_avg_proj_avg'
                                  ]:

                        track_dir = os.path.join(block_dir, track)
                        os.makedirs(track_dir, exist_ok=True)

                        if 'activation' in space:
                            if module.weight.requires_grad:
                                tracked = getattr(module, track)
                                plt.clf()
                                plt.hist(torch.cat(tracked).cpu(), bins=100, histtype='step')
                                plt.title(f'{kk} - {module}')
                                plt.savefig(os.path.join(track_dir, f'{kk} - cosine - {module}.pdf'))

                        elif 'weight' in space:
                            if module.weight.requires_grad:
                                tracked = getattr(module, track)['weight']
                                plt.clf()
                                plt.hist(torch.cat(tracked).cpu(), bins=100, histtype='step')
                                plt.title(f'{kk} - {module} - weight')
                                plt.savefig(os.path.join(track_dir, f'{kk} - {module} - weight.pdf'))

                            if module.bias is not None and module.bias.requires_grad:
                                tracked = getattr(module, track)['bias']
                                plt.clf()
                                plt.hist(torch.cat(tracked).cpu(), bins=100, histtype='step')
                                plt.title(f'{kk} - {module} - bias')
                                plt.savefig(os.path.join(track_dir, f'{kk} - {module} - bias.pdf'))


def log_perf_to_wandb(net, epoch, scheduler, mode):
    stats = dict()
    # learning rate
    if mode == 'train':
        stats.update({f'learning rate': scheduler.get_last_lr()[0]})

    # block metrics
    for k, block in enumerate(net.blocks):
        stats.update({f'block-{k}/{mode}-loss': block.auxnet.loss.avg,
                      f'block-{k}/{mode}-accuracy': block.auxnet.accs.avg})

    # model metrics
    stats.update({f'{mode}-loss': net.blocks[-1].auxnet.loss.avg, f'{mode}-accuracy': net.blocks[-1].auxnet.accs.avg})

    wandb.log(stats, step=epoch)
    return None


def print_perf(net, epoch, loss, acc, lr, mode):
    if mode == 'train':
        print(f'Epoch {epoch} - Train Acc {acc * 100: .3f}% - Train Loss {loss: .3f} - Learning Rate {lr: .6f}')
        print('Accuracy : ' + ' - '.join([f'{block.auxnet.accs.avg * 100: .3f}%' for block in net.blocks]))
        print('Loss : ' + ' - '.join([f'{block.auxnet.loss.avg: .3f}' for block in net.blocks]))
    elif mode == 'test':
        print(f'Epoch {epoch} - Test Acc {acc * 100: .3f}% - Test Loss {loss: .3f}')
        print('Accuracy : ' + ' - '.join([f'{block.auxnet.accs.avg * 100: .3f}%' for block in net.blocks]))
        print('Loss : ' + ' - '.join([f'{block.auxnet.loss.avg: .3f}' for block in net.blocks]))
        print()


def log_histogram_to_wandb(net, mode):
    return None
