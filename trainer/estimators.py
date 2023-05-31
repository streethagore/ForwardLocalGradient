import torch
from trainer.per_sample_gradients import linear_gradient, conv_gradient, batchnorm_gradient


def get_estimator(label='weight'):
    if label == 'weight':
        return weight_perturb_estimator
    elif label == 'avg-weight':
        return avg_weight_perturb_estimator
    elif label == 'span-weight':
        return span_weight_perturb_estimator
    elif label == 'sign-weight':
        return sign_weight_perturb_estimator
    elif label == 'random-sign-weight':
        return random_sign_weight_perturb_estimator
    elif label == 'activation':
        return activity_perturb_estimator
    elif label == 'span-activation':
        return span_activity_perturb_estimator
    elif label == 'activation-2':
        return activity_perturb_estimator_2


# ========== Algebra Routine ==========
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


def _compute_projection(target, guess, eps=1e-16):
    scalar_prod = torch.einsum('n...,n...->n', target, guess)
    guess_norm_2 = torch.einsum('n...,n...->n', guess, guess)
    weight = scalar_prod / (guess_norm_2 + eps)
    weighted_grad = torch.einsum('n,n...->n...', weight, guess)
    return weighted_grad


def compute_projection(target, guess, eps=1e-16):
    scalar_prod = batch_scalar_product(target, guess)
    guess_norm_2 = batch_scalar_product(guess, guess)
    weight = scalar_prod / (guess_norm_2 + eps)
    weighted_grad = per_sample_product(weight, guess)
    return weighted_grad


def compute_random_sign_projection(target, guess, eps=1e-16):
    weight = torch.sign(torch.rand(target.size(0), device=target.device) - 0.5)
    weighted_grad = torch.einsum('n,n...->n...', weight, guess)
    return weighted_grad


def compute_sign_projection(target, guess, eps=1e-16):
    scalar_prod = torch.einsum('n...,n...->n', target, guess)
    weight = torch.sign(scalar_prod)
    weighted_grad = torch.einsum('n,n...->n...', weight, guess)
    return weighted_grad


def compute_avg_projection(target, guess, eps=1e-16):
    scalar_prod = (target * guess).sum()
    guess_norm_2 = (guess * guess).sum()
    weight = scalar_prod / (guess_norm_2 + eps)
    weighted_grad = weight * guess
    return weighted_grad


def compute_span_projection(target, guess):
    n = target.size(0)
    weighted_grad = ((target.view(n, -1) @ guess.view(n, -1).T) @ torch.linalg.pinv(guess.view(n, -1).T)).view(
        guess.shape)
    return weighted_grad


# ========== Average Weight Perturb Estimator ==========
def _avg_weight_perturb_estimator(module):
    if module.weight.requires_grad:
        weighted_grad = compute_avg_projection(module.target['weight'].sum(dim=0), module.guess['weight'].sum(dim=0))
        module.weight.grad = weighted_grad
    if module.bias is not None and module.bias.requires_grad:
        weighted_grad = compute_avg_projection(module.target['bias'].sum(dim=0), module.guess['bias'].sum(dim=0))
        module.weight.grad = weighted_grad


def avg_weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _avg_weight_perturb_estimator(module)


# ========== Weight Perturb Estimator ==========
def _weight_perturb_estimator(module):
    if module.weight.requires_grad:
        weighted_grad = compute_projection(module.target['weight'],
                                           module.guess['weight'])
        module.weight.grad = weighted_grad.sum(dim=0)
    if module.bias is not None and module.bias.requires_grad:
        weighted_grad = compute_projection(module.target['bias'], module.guess['bias'])
        module.weight.grad = weighted_grad.sum(dim=0)
    return None


def weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _weight_perturb_estimator(module)
    return None


# ========== Random Sign Perturb Estimator ==========
def _random_sign_weight_perturb_estimator(module):
    if module.weight.requires_grad:
        weighted_grad = compute_random_sign_projection(module.target['weight'], module.guess['weight'])
        module.weight.grad = weighted_grad.sum(dim=0)
    if module.bias is not None and module.bias.requires_grad:
        weighted_grad = compute_random_sign_projection(module.target['bias'], module.guess['bias'])
        module.weight.grad = weighted_grad.sum(dim=0)
    return None


def random_sign_weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _random_sign_weight_perturb_estimator(module)
    return None


# ========== Sign Weight Perturb Estimator ==========
def _sign_weight_perturb_estimator(module):
    if module.weight.requires_grad:
        weighted_grad = compute_sign_projection(module.target['weight'], module.guess['weight'])
        module.weight.grad = weighted_grad.sum(dim=0)

    if module.bias is not None and module.bias.requires_grad:
        weighted_grad = compute_sign_projection(module.target['bias'], module.guess['bias'])
        module.weight.grad = weighted_grad.sum(dim=0)
    return None


def sign_weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _weight_perturb_estimator(module)
    return None


# ========== Span Weight Perturb Estimator ==========
def _span_weight_perturb_estimator(module):
    if module.weight.requires_grad:
        sample_grad = compute_span_projection(module.target['weight'], module.guess['weight'])
        module.weight.grad = sample_grad.sum(dim=0)
    if module.bias is not None and module.bias.requires_grad:
        sample_grad = compute_span_projection(module.target['bias'], module.guess['bias'])
        module.bias.grad = sample_grad.sum(dim=0)
    return None


def span_weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _weight_perturb_estimator(module)
    return None


# ========== Filtered Weight Perturb Estimator ==========
def _filetered_weight_perturb_estimator(module):
    if module.weight.requires_grad:
        scalar_prod = torch.einsum('n...,n...->n', module.target['weight'], module.guess['weight'])
        pos_index = scalar_prod > 0
        sample_grad = module.guess['weight'][pos_index] * pos_index.size(0) / pos_index.sum()
        module.weight.grad = sample_grad.sum(dim=0)
    if module.bias is not None and module.bias.requires_grad:
        scalar_prod = torch.einsum('n...,n...->n', module.target['weight'], module.guess['weight'])
        pos_index = scalar_prod > 0
        sample_grad = module.guess['bias'][pos_index] * pos_index.size(0) / pos_index.sum()
        module.bias.grad = sample_grad.sum(dim=0)


def filtered_weight_perturb_estimator(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    _filetered_weight_perturb_estimator(module)


# ========== Activity Perturb Estimator ==========
def activity_perturb_estimator(net, eps=1e-16):
    return None


def activity_perturb_estimator_2(net, eps=1e-16):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            scalar_prod = batch_scalar_product(block.identity.target, block.identity.guess)
            guess_norm_2 = batch_scalar_product(block.identity.guess, block.identity.guess)
            weight = scalar_prod / (guess_norm_2 + eps)
            for module in block.block.modules():
                if isinstance(module, torch.nn.modules.Linear) or \
                        isinstance(module, torch.nn.modules.conv._ConvNd) or \
                        isinstance(module, torch.nn.modules.batchnorm._BatchNorm):

                    if isinstance(module, torch.nn.modules.Linear):
                        sample_grad = linear_gradient(module.inputs, module.guess)
                    if isinstance(module, torch.nn.modules.conv._ConvNd):
                        sample_grad = conv_gradient(module, module.inputs, module.guess)
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        sample_grad = batchnorm_gradient(module, module.inputs, module.guess)

                    if module.weight.requires_grad:
                        module.weight.grad = per_sample_product(weight, sample_grad['weight']).sum(dim=0)
                    if module.bias is not None and module.bias.requires_grad:
                        module.bias.grad = per_sample_product(weight, sample_grad['bias']).sum(dim=0)
    return None


# ========== Span Activity Perturb Estimator ==========
def span_activity_perturb_estimator(net, eps=1e-16):
    return None