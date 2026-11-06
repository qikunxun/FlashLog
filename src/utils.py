import torch
from torch.nn import functional as F

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src, index, dim=-1, dim_size=None):
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def log_loss(p_score, label, E, tau_2, one, thr=1e-7):
    one_hot = F.one_hot(label, E).float()
    loss = -torch.sum(
        one_hot * torch.log(torch.maximum(p_score / tau_2, one * thr)),
        dim=-1)
    loss = torch.mean(loss)
    return loss


def log_loss_common_sp(p_score, label, E, tau_2, thr=1e-7):
    i_y = label.long()
    i_x = torch.arange(0, i_y.shape[0]).to(i_y.device)
    i = torch.stack([i_x, i_y], dim=0)
    v = torch.ones_like(label).float()
    one_hot = torch.sparse.FloatTensor(i.long(), v, torch.Size([i_y.shape[0], E]))
    logits = torch.sparse.sum(p_score * one_hot, dim=-1).to_dense()
    loss = -torch.mean(
        torch.log(torch.maximum(logits / tau_2, torch.ones_like(logits) * thr)),
        dim=-1)
    return loss
