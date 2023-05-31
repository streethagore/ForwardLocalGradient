from functools import partial
import torch
from trainer.per_sample_gradients import linear_gradient, conv_gradient, batchnorm_gradient
from trainer.estimators import compute_projection, compute_span_projection, batch_scalar_product, per_sample_product


def forward_hook(module, input, output):
    module.inputs = input[0]
    return None


def activity_perturb_backward_hook(module, grad_input, grad_output, dest='target'):
    grad_output = grad_output[0].contiguous()
    setattr(module, dest, grad_output)
    if dest == 'guess':
        if hasattr(module, 'random_flag'):
            if module.random_flag == 'gaussian':
                noise = torch.randn_like(module.target)
            elif module.random_flag == 'rademacher':
                noise = torch.randint_like(module.target, 0, 2) * 2 - 1
            else:
                raise ValueError('Wrong noise type:', module.random_flag)
            scalar_prod = batch_scalar_product(module.target, noise)
            proj = per_sample_product(scalar_prod, noise)
        else:
            proj = compute_projection(module.target, grad_output)

        return proj,


def activity_perturb_backward_hook_2(module, grad_input, grad_output, dest='target'):
    setattr(module, dest, grad_output[0].contiguous())
    return None


def span_activity_perturb_backward_hook(module, grad_input, grad_output, dest='target'):
    grad_output = grad_input[0].contiguous().detach()
    setattr(module, dest, grad_output)
    if dest == 'target':
        return None
    else:
        return compute_span_projection(module.target, grad_output),


def weight_perturb_backward_hook(module, grad_input, grad_output, dest='target'):
    grad_output = grad_output[0].contiguous()
    if isinstance(module, torch.nn.modules.Linear):
        sample_grad = linear_gradient(module.inputs, grad_output)
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        sample_grad = conv_gradient(module, module.inputs, grad_output)
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sample_grad = batchnorm_gradient(module, module.inputs, grad_output)
    setattr(module, dest, sample_grad)
    return None


def set_hook(net, hook):
    handles = []
    for module in net.modules():
        if isinstance(module, torch.nn.modules.Linear) or \
                isinstance(module, torch.nn.modules.conv._ConvNd) or \
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(hook))
    return handles


def set_net_hook(net, space='weight', dest='target'):
    handles = []
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            if 'weight' in space:
                handles.extend(set_hook(block.block, partial(weight_perturb_backward_hook, dest=dest)))
            elif space == 'activation':
                handles.append(block.identity.register_full_backward_hook(partial(activity_perturb_backward_hook, dest=dest)))
            elif space == 'activation-2':
                handles.extend(set_hook(block.block, partial(activity_perturb_backward_hook_2, dest=dest)))
                handles.append(block.identity.register_full_backward_hook(partial(activity_perturb_backward_hook_2, dest=dest)))
            elif space == 'span-activation':
                handles.append(block.identity.register_full_backward_hook(partial(span_activity_perturb_backward_hook, dest=dest)))
    return handles
