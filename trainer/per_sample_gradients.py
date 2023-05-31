import torch
from trainer.utils import per_sample_product, batch_scalar_product
from opt_einsum import contract


def linear_gradient(inputs, grad_output):
    grad_weight = torch.einsum('np,nq->nqp', inputs, grad_output)
    grad_bias = torch.einsum("nk...->nk", grad_output)
    return dict(weight=grad_weight, bias=grad_bias)


def conv_gradient(module, inputs, grad_output):
    n = inputs.shape[0]
    activations = torch.nn.functional.unfold(
        inputs,
        kernel_size=module.kernel_size,
        padding=module.padding,
        stride=module.stride,
        dilation=module.dilation,
    )

    grad_output = grad_output.reshape(n, -1, activations.shape[-1])

    # Weight gradient
    # n = batch_sz; o = num_out_channels; p = (num_in_channels/groups)*kernel_sz
    # grad_weight = torch.einsum("npq,noq->nop", activations, grad_output)
    grad_weight = contract("npq,noq->nop", activations, grad_output)
    # rearrange the above tensor and extract diagonals.
    grad_weight = grad_weight.view(
        n,
        module.groups,
        -1,
        module.groups,
        int(module.in_channels / module.groups),
        torch.prod(torch.tensor(module.kernel_size)),
    )
    # grad_weight = torch.einsum("ngrg...->ngr...", grad_weight)
    grad_weight = contract("ngrg...->ngr...", grad_weight)
    shape = [n] + list(module.weight.shape)
    grad_weight = grad_weight.reshape(shape)

    # Bias gradient
    grad_bias = grad_output.sum(dim=2)

    return dict(weight=grad_weight, bias=grad_bias)


def batchnorm_gradient(module, inputs, grad_output):
    # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py

    # calculate running estimates
    if module.training:
        mean = inputs.mean([0, 2, 3])
        # use biased var in train
        var = inputs.var([0, 2, 3], unbiased=False)
    else:
        mean = module.running_mean
        var = module.running_var

    activations_afterbn = (inputs - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + module.eps))
    grad_weight = torch.einsum("ni...,ni...->ni", activations_afterbn, grad_output)
    grad_bias = torch.einsum("nk...->nk", grad_output)

    return dict(weight=grad_weight, bias=grad_bias)
