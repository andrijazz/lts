import numpy as np
import torch
import torch.nn.functional as F


def ash_b(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    x.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x.zero_().scatter_(dim=1, index=i, src=v)
    return x


def ash_s(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x.zero_().scatter_(dim=1, index=i, src=v)
    s2 = v.sum(dim=1)
    scale = s1 / s2
    return x * torch.exp(scale[:, None])


def scale(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    x = F.relu(x)
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    s2 = v.sum(dim=1)
    scale = s1 / s2
    return x * torch.exp(scale[:, None])


def react(x, threshold):
    x = x.clip(max=threshold)
    return x


def lts(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    s2 = v.sum(dim=1)
    scale = s1 / s2
    apply_to_logits = True
    return scale[:, None] ** 2


def get_ood_detector(method):
    try:
        [fn, p] = method.split('@')
        if fn == 'ash_b':
            return lambda x: ash_b(x, int(p))
        if fn == 'ash_s':
            return lambda x: ash_s(x, int(p))
        if fn == 'lts':
            return lambda x: lts(x, int(p))
        if fn == 'ash_p':
            return lambda x: ash_p(x, int(p))
        if fn == 'scale':
            return lambda x: scale(x, int(p))
        if fn == 'react':
            return lambda x: react(x, int(p))

    except Exception as e:
        print(e)
        exit('Unsupported ood method')
