import numpy as np
import torch
import torch.nn.functional as F


def get_msp_score(logits):
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores

def get_energy_score(logits):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_score(logits, method):
    if method == "msp":
        return get_msp_score(logits)
    if method == "energy":
        return get_energy_score(logits)
    exit('Unsupported scoring method')


def ash_b_2d(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    # calculate the sum of the input per sample
    x = F.relu(x)
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    x.zero_().scatter_(dim=1, index=i, src=fill)
    return x

def ash_p_2d(x, percentile=65):
    assert x.dim() == 2
    assert 0 <= percentile <= 100

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x.zero_().scatter_(dim=1, index=i, src=v)
    return x

def losh_2d(x, percentile=65):
    # assert x.dim() == 2
    assert 0 <= percentile <= 100
    x = F.relu(x)
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    high = torch.zeros_like(x)
    high.scatter_(dim=1, index=i, src=torch.ones_like(v))
    s2 = v.sum(dim=1)
    return s1 / s2


def kurtosis(t, dim):
    mean = torch.mean(t, dim=dim, keepdim=True)
    diffs = t - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=dim, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    # skews = torch.mean(torch.pow(zscores, 3.0))
    k = torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdim=True) - 3.0
    return k.squeeze(1).detach().cpu()


def scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    return input * torch.exp(scale[:, None, None, None])


def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.clone()
    t = t.reshape((b, c * h * w))
    # t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    t = t.reshape((b, c, h, w))
    x = t

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])
    return x


def ash_rand(x, percentile=65, r1=0, r2=10):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    v = v.uniform_(r1, r2)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x


def react(x, threshold):
    x = x.clip(max=threshold)
    return x


def react_and_ash(x, clip_threshold, pruning_percentile):
    x = x.clip(max=clip_threshold)
    x = ash_s(x, pruning_percentile)
    return x


def apply_ash(x, method):
    if method.startswith('react_and_ash@'):
        [fn, t, p] = method.split('@')
        return eval(fn)(x, float(t), int(p))

    if method.startswith('react@'):
        [fn, t] = method.split('@')
        return eval(fn)(x, float(t))

    if method.startswith('ash'):
        [fn, p] = method.split('@')
        return eval(fn)(x, int(p))

    if method.startswith('scale@'):
        [fn, t] = method.split('@')
        return eval(fn)(x, float(t))

    return x
