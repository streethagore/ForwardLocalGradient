import torch
from trainer.utils import accuracy, set_bn_train, set_bn_eval


def save_module_gradient(module):
    if module.weight.requires_grad:
        module.weight.saved_grad = module.weight.grad.detach().clone()
    if module.bias is not None and module.bias.requires_grad:
        module.bias.saved_grad = module.bias.grad.detach().clone()


def set_module_gradient(module):
    if module.weight.requires_grad:
        module.weight.grad = module.weight.saved_grad
    if module.bias is not None and module.bias.requires_grad:
        module.bias.grad = module.bias.saved_grad


def save_gradient(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.Linear) or \
                isinstance(module, torch.nn.modules.conv._ConvNd) or \
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            save_module_gradient(module)


def set_gradient(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.Linear) or \
                isinstance(module, torch.nn.modules.conv._ConvNd) or \
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            set_module_gradient(module)


def save_auxiliary_gradients(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            save_gradient(block.auxnet)
    return None


def set_auxiliary_gradients(net):
    for k, block in enumerate(net.blocks):
        if k < len(net.blocks) - 1:
            set_gradient(block.auxnet)


def compute_random_directions(net, data, target, criterion, dest='guess', space='weight', noise_type='gaussian'):
    if 'activation' in space:
        x = data

    for k, block in enumerate(net.blocks):
        if dest == 'target' or k < len(net.blocks) - 1:
            if 'weight' in space:
                for module in block.block.modules():
                    if isinstance(module, torch.nn.modules.Linear) or \
                            isinstance(module, torch.nn.modules.conv._ConvNd) or \
                            isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        if hasattr(module, 'target'):
                            if noise_type == 'gaussian':
                                random_weights = torch.randn_like(
                                    getattr(module, 'target')['weight']
                                ) if module.weight.requires_grad else None
                                random_bias = torch.randn_like(
                                    getattr(module, 'target')['bias']
                                ) if module.bias is not None and module.bias.requires_grad else None
                            elif noise_type == 'rademacher':
                                random_weights = torch.randint_like(
                                    getattr(module, 'target')['weight'], 0, 2
                                ) * 2 - 1 if module.weight.requires_grad else None
                                random_bias = torch.randint_like(
                                    getattr(module, 'target')['bias'], 0, 2
                                ) * 2 - 1 if module.bias is not None and module.bias.requires_grad else None
                            setattr(module, dest, dict(weight=random_weights, bias=random_bias))
                        else:
                            if module.weight.requires_grad:
                                if noise_type == 'gaussian':
                                    module.weight.grad = torch.randn_like(module.weight)
                                elif noise_type == 'rademacher':
                                    module.weight.grad = torch.randint_like(module.weight, 0, 2) * 2 - 1
                            if module.bias is not None and module.bias.requires_grad:
                                if noise_type == 'gaussian':
                                    module.bias.grad = torch.randn_like(module.bias)
                                elif noise_type == 'rademacher':
                                    module.bias.grad = torch.randint_like(module.bias, 0, 2) * 2 - 1

            elif 'activation' in space:
                x = block(x.detach())
                x.backward(torch.zeros_like(x))
    return None, None


def compute_global_gradients(net, data, target, criterion, dest=None, space='weight'):
    pred = net(data)
    loss = criterion(pred, target)
    accs = accuracy(pred, target)
    _ = loss.item()
    loss.backward()

    net.blocks[-1].auxnet.loss.update(loss.item())
    net.blocks[-1].auxnet.accs.update(accs.item())
    return loss, accs


def compute_local_gradients(net, data, target, criterion, dest='guess', space='weight'):
    x = data
    for k, block in enumerate(net.blocks):
        if dest == 'target' or k < len(net.blocks) - 1:
            x = block(x.detach())
            pred = block.auxnet(x)
            loss = criterion(pred, target)
            accs = accuracy(pred, target)
            _ = loss.item()
            loss.backward()

            block.auxnet.loss.update(loss.item())
            block.auxnet.accs.update(accs.item())

    return loss, accs


def compute_local_above_gradients(net, data, target, criterion, dest=None, space='weight'):
    x, losses, accs = data, [], []
    loss = torch.tensor(0.0, device=x.device)
    for k, block in enumerate(net.blocks):
        block.apply(set_bn_train)
        x = block(x.detach())
        if k == 0:
            pred = block.auxnet(x.detach())
            loss = criterion(pred, target)
            accs = accuracy(pred, target)
            if loss.requires_grad:
                loss.backward()
            block.auxnet.loss.update(loss.item())
            block.auxnet.accs.update(accs.item())
        if k < len(net.blocks) - 1:
            x_above = net.blocks[k + 1](x)
            pred = net.blocks[k + 1].auxnet(x_above)
            net.blocks[k + 1].apply(set_bn_eval)

            loss_k = criterion(pred, target)
            loss += loss_k
            accs = accuracy(pred, target)

            net.blocks[k + 1].auxnet.loss.update(loss_k.item())
            net.blocks[k + 1].auxnet.accs.update(accs.item())

    loss.backward()

    return loss, accs


def get_gradient_computation_function(label='global'):
    if label == 'global':
        return compute_global_gradients
    elif label in ['local', 'ntk', 'fixed-ntk']:
        return compute_local_gradients
    elif label == 'above':
        return compute_local_above_gradients
    elif label == 'random':
        return compute_random_directions
