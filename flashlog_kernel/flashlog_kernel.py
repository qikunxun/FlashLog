import numpy as np
import torch
from torch import autograd
from torch.utils import cpp_extension
import os
import sys
import subprocess
import time

_FLASHLOG_CPU_THREADS_CONFIGURED = False


def _flashlog_available_logical_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def _flashlog_physical_cores():
    try:
        cores = set()
        phys_id = None
        core_id = None
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                    phys_id = None
                    core_id = None
                    continue
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
        if phys_id is not None and core_id is not None:
            cores.add((phys_id, core_id))
        return len(cores) if cores else None
    except Exception:
        return None


def _choose_flashlog_cpu_threads():
    logical = max(1, _flashlog_available_logical_cpus())
    physical = _flashlog_physical_cores()
    if physical is None:
        physical = max(1, logical // 2 if logical >= 8 else logical)
    usable_physical = min(physical, logical)
    if logical >= 96:
        return min(32, max(16, usable_physical // 2))
    if logical >= 48:
        return min(24, max(12, usable_physical // 2))
    if logical >= 24:
        return min(16, max(8, usable_physical))
    return max(1, logical)


def configure_flashlog_cpu_threads(force=False):
    """Configure CPU thread count for FlashLog sparse kernels.

    FLASHLOG_CPU_THREADS:
      - unset/off/0/false: do not modify torch thread settings
      - auto: choose a conservative memory-bandwidth-friendly value
      - off/0/false: do not modify torch thread settings
      - integer: force that many intra-op threads

    FLASHLOG_CPU_INTEROP_THREADS optionally sets inter-op threads. It must be set
    before PyTorch starts parallel work; RuntimeError is ignored if too late.
    """
    global _FLASHLOG_CPU_THREADS_CONFIGURED
    if _FLASHLOG_CPU_THREADS_CONFIGURED and not force:
        return torch.get_num_threads()

    setting = os.environ.get("FLASHLOG_CPU_THREADS", "off").strip().lower()
    if setting in ("", "auto"):
        target = _choose_flashlog_cpu_threads()
    elif setting in ("off", "none", "false", "0"):
        _FLASHLOG_CPU_THREADS_CONFIGURED = True
        return torch.get_num_threads()
    else:
        try:
            target = int(setting)
        except ValueError:
            target = _choose_flashlog_cpu_threads()

    target = max(1, min(target, _flashlog_available_logical_cpus()))
    if torch.get_num_threads() != target:
        torch.set_num_threads(target)

    interop = os.environ.get("FLASHLOG_CPU_INTEROP_THREADS")
    if interop is not None:
        try:
            interop_threads = max(1, int(interop))
            if torch.get_num_interop_threads() != interop_threads:
                torch.set_num_interop_threads(interop_threads)
        except (ValueError, RuntimeError):
            pass

    _FLASHLOG_CPU_THREADS_CONFIGURED = True
    return torch.get_num_threads()


def load_flashlog_kernel():
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
    sources = [os.path.join(source_dir, "flashlog.cpp")]

    # Add CUDA source if available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        sources.append(os.path.join(source_dir, "flashlog.cu"))

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "--use_fast_math"]

    if cuda_available:
        extra_cflags.append("-DWITH_CUDA")

    if sys.platform == 'darwin':
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
        cxx_include = os.path.join(sdk_path, 'usr', 'include', 'c++', 'v1')
        extra_cflags += ["-stdlib=libc++", f"-isysroot{sdk_path}", f"-I{cxx_include}"]

    if torch.backends.openmp.is_available() and not sys.platform.startswith('darwin'):
        extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
    else:
        extra_cflags.append("-DAT_PARALLEL_NATIVE")

    print("Loading FlashLog kernel extension. This may take a while...")
    module = cpp_extension.load(
        name="flashlog_kernel_cpp",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags if cuda_available else None,
        verbose=False
    )
    print("FlashLog kernel loaded.")
    return module


_module = None


def _kernel_stage_timing_enabled():
    return os.environ.get("FLASHLOG_STAGE_TIMING", "") == "1"


def _kernel_cuda_reduce_enabled():
    return os.environ.get("FLASHLOG_CUDA_REDUCE", "") == "1"


def _kernel_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _kernel_log(stage, start_time):
    if not _kernel_stage_timing_enabled():
        return
    _kernel_sync()
    print("StageTiming flashlog_kernel stage:{} seconds:{:.4f}".format(
        stage,
        time.perf_counter() - float(start_time),
    ))


def sparse2d_filtered_ranks_cuda(idx_b, idx_e, values, targets, filter_keys, B, E, use_eql=False):
    module = get_module()
    return module.sparse2d_filtered_ranks_cuda(
        idx_b.long().contiguous(),
        idx_e.long().contiguous(),
        values.contiguous(),
        targets.long().contiguous(),
        filter_keys.long().contiguous(),
        int(B),
        int(E),
        bool(use_eql),
    )


def vectorized_operation_sparse3d_fixed_relation(sp_state, operators, csr_struct, r_size):
    """Advance sparse rule levels through one fixed relation per level.

    ``operators`` is shaped ``[B, L]`` and uses the standard FlashLog operator
    IDs: forward relations, inverse relations offset by ``r_size``, and the
    identity operator at ``2 * r_size``.
    """
    if not sp_state.is_cuda:
        raise ValueError("fixed-relation rule inference currently requires CUDA")
    sp_state = sp_state.coalesce()
    indices = sp_state.indices()
    values = sp_state.values()
    B, L, E = sp_state.shape
    if tuple(operators.shape) != (B, L):
        raise ValueError(
            "operators must have shape [{}, {}], got {}".format(B, L, tuple(operators.shape))
        )
    module = get_module()
    batch, level, entity, output_values = module.flashlog_forward_sparse3d_fixed_relation_cuda(
        indices[0], indices[1], indices[2], values, operators,
        csr_struct[1], csr_struct[8], csr_struct[9], csr_struct[10], csr_struct[11],
        csr_struct[5], csr_struct[12], csr_struct[13], csr_struct[14], csr_struct[15],
        int(B), int(L), int(E), int(r_size),
    )
    return _coalesce_sparse3d_sum_raw(
        ((batch, level, entity, output_values),),
        B, L, E,
        use_cuda_kernel=_kernel_cuda_reduce_enabled(),
    )

def get_module():
    global _module
    if _module is None:
        t0 = time.perf_counter()
        _module = load_flashlog_kernel()
        _kernel_log("load_module", t0)
    return _module


def score_function(x):
    return torch.sigmoid(x)


def _prepared_mask_pair(mask_values):
    if (isinstance(mask_values, (tuple, list)) and len(mask_values) == 2 and
            torch.is_tensor(mask_values[0]) and torch.is_tensor(mask_values[1])):
        return mask_values[0].contiguous().float(), mask_values[1].contiguous().float()
    return None


def _apply_mask_order(mask_values, order, weight=None):
    mask_f = (mask_values if mask_values.dtype == torch.bool else mask_values.float()).contiguous().view(-1)
    if mask_f.is_cuda:
        module = get_module()
        order_i = order.to(device=mask_f.device, dtype=torch.int32).contiguous()
        if weight is None:
            w_tensor = torch.empty(0, device=mask_f.device)
        else:
            w_tensor = weight.to(device=mask_f.device, dtype=torch.float).squeeze(-1).contiguous()
        return module.apply_mask_cuda(mask_f, order_i, w_tensor)

    order_l = order.to(device=mask_f.device, dtype=torch.long)
    if weight is not None:
        edge_weight = score_function(weight.to(device=mask_f.device, dtype=torch.float)).squeeze(-1).view(-1)
        mask_f = mask_f * edge_weight
    return mask_f[order_l].contiguous().float()


def prepare_csr_masks(csr_struct, mask_values, weight=None):
    """Prepare CSR-ordered ORI/INV masks once for a model forward."""
    return (
        _apply_mask_order(mask_values, csr_struct[3], weight),
        _apply_mask_order(mask_values, csr_struct[7], weight),
    )


def _coalesce_sparse3d_reduce(b, l, e, v, B, L, E, reduce="sum", return_inverse=False):
    if v.numel() == 0:
        empty_i = torch.empty(0, device=v.device, dtype=torch.long)
        empty_v = torch.empty(0, device=v.device, dtype=v.dtype)
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        sp = torch.sparse_coo_tensor(idx, empty_v, [B, L, E], is_coalesced=True)
        result = (sp, empty_i, empty_v)
        return result + (empty_i.int(),) if return_inverse else result

    keys = b * (L * E) + l * E + e
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = v[sort_idx]
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    out_values = torch.zeros(unique_keys.shape[0], device=v.device, dtype=v.dtype)
    if reduce == "max":
        out_values.scatter_reduce_(0, inverse, sorted_values, reduce="amax", include_self=False)
    else:
        out_values.scatter_add_(0, inverse, sorted_values)
    out_b = unique_keys // (L * E)
    rem = unique_keys % (L * E)
    out_l = rem // E
    out_e = rem % E
    sp = torch.sparse_coo_tensor(
        torch.stack([out_b, out_l, out_e], dim=0),
        out_values,
        [B, L, E],
        is_coalesced=True
    )
    result = (sp, unique_keys, out_values)
    if return_inverse:
        # Reuse the existing sort and segment IDs; no additional key search.
        dtype = torch.int32 if unique_keys.numel() < 2**31 else torch.int64
        raw_pos = torch.empty(v.numel(), device=v.device, dtype=dtype)
        raw_pos[sort_idx] = inverse.to(dtype)
        return result + (raw_pos,)
    return result


def _coalesce_sparse3d_sum_raw(raw_parts, B, L, E, use_cuda_kernel=False, use_cpu_kernel=False,
                             return_inverse=False):
    valid = [(b, l, e, v) for b, l, e, v in raw_parts if v.numel() > 0]
    if not valid:
        device = raw_parts[0][3].device
        dtype = raw_parts[0][3].dtype
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        # Keep the zero gradient path when pruning removes every edge.
        empty_v = torch.cat([part[3] for part in raw_parts])
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        out = torch.sparse_coo_tensor(idx, empty_v, [B, L, E], is_coalesced=True)
        return (out, empty_i.int()) if return_inverse else out
    if use_cuda_kernel and valid[0][3].is_cuda and len(valid) <= 3:
        device = valid[0][3].device
        dtype = valid[0][3].dtype
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        empty_v = torch.empty(0, device=device, dtype=dtype)
        padded = valid + [(empty_i, empty_i, empty_i, empty_v)] * (3 - len(valid))
        module = get_module()
        coalesce = (module.sparse3d_coalesce_sum3_cached_cuda if return_inverse
                    else module.sparse3d_coalesce_sum3_cuda)
        reduced = coalesce(
            padded[0][0].long(), padded[0][1].long(), padded[0][2].long(), padded[0][3],
            padded[1][0].long(), padded[1][1].long(), padded[1][2].long(), padded[1][3],
            padded[2][0].long(), padded[2][1].long(), padded[2][2].long(), padded[2][3],
            B, L, E
        )
        out_b, out_l, out_e, out_v = reduced[:4]
        out = torch.sparse_coo_tensor(
            torch.stack([out_b, out_l, out_e], dim=0),
            out_v.to(dtype=dtype),
            [B, L, E],
            is_coalesced=True
        )
        return (out, reduced[4]) if return_inverse else out
    if use_cpu_kernel and not return_inverse and not valid[0][3].is_cuda and len(valid) <= 3:
        device = valid[0][3].device
        dtype = valid[0][3].dtype
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        empty_v = torch.empty(0, device=device, dtype=dtype)
        padded = valid + [(empty_i, empty_i, empty_i, empty_v)] * (3 - len(valid))
        module = get_module()
        out_b, out_l, out_e, out_v = module.sparse3d_coalesce_sum3_cpu(
            padded[0][0].long(), padded[0][1].long(), padded[0][2].long(), padded[0][3],
            padded[1][0].long(), padded[1][1].long(), padded[1][2].long(), padded[1][3],
            padded[2][0].long(), padded[2][1].long(), padded[2][2].long(), padded[2][3],
            B, L, E
        )
        return torch.sparse_coo_tensor(
            torch.stack([out_b, out_l, out_e], dim=0),
            out_v.to(dtype=dtype),
            [B, L, E],
            is_coalesced=True
        )
    b = torch.cat([x[0].long() for x in valid], dim=0)
    l = torch.cat([x[1].long() for x in valid], dim=0)
    e = torch.cat([x[2].long() for x in valid], dim=0)
    v = torch.cat([x[3] for x in valid], dim=0)
    reduced = _coalesce_sparse3d_reduce(b, l, e, v, B, L, E, reduce="sum",
                                      return_inverse=return_inverse)
    return (reduced[0], reduced[3]) if return_inverse else reduced[0]


def _align_sparse_grad_to_output(grad, out_indices, L, E):
    """Align once per unique output, not once per expanded graph edge."""
    if not grad.is_sparse:
        return grad[tuple(out_indices)].contiguous()
    grad = grad.coalesce()
    if torch.equal(grad.indices(), out_indices):
        return grad.values().contiguous()
    keys = out_indices[0] * (L * E) + out_indices[1] * E + out_indices[2]
    gidx = grad.indices()
    gkeys = gidx[0] * (L * E) + gidx[1] * E + gidx[2]
    values = torch.zeros(keys.numel(), device=grad.device, dtype=grad.dtype)
    if gkeys.numel() and keys.numel():
        pos = torch.searchsorted(gkeys, keys)
        safe = pos.clamp(max=gkeys.numel() - 1)
        valid = (pos < gkeys.numel()) & (gkeys[safe] == keys)
        values[valid] = grad.values()[safe[valid]]
    return values


def _cached_raw_grad(values, positions):
    result = torch.zeros(positions.numel(), device=values.device, dtype=values.dtype)
    valid = positions >= 0
    result[valid] = values[positions[valid].long()]
    return result


def _max_sparse3d_winner_positions(win_source, win_raw, raw_sizes):
    """Reuse exact forward tie winners; losing contributions map to -1."""
    dtype = torch.int32 if win_raw.numel() <= torch.iinfo(torch.int32).max else torch.int64
    output_pos = torch.arange(win_raw.numel(), device=win_raw.device, dtype=dtype)
    positions = []
    for source, size in enumerate(raw_sizes):
        pos = torch.full((size,), -1, device=win_raw.device, dtype=dtype)
        chosen = win_source == source
        pos[win_raw[chosen].long()] = output_pos[chosen]
        positions.append(pos)
    return tuple(positions)


def _coalesce_sparse3d_max_raw_parts(raw_parts, B, L, E):
    valid = []
    for source, b, l, e, v, tie in raw_parts:
        if v.numel() == 0:
            continue
        raw = torch.arange(v.shape[0], device=v.device, dtype=torch.long)
        valid.append((source, b.long(), l.long(), e.long(), v, tie.long(), raw))

    if not valid:
        device = raw_parts[0][4].device
        dtype = raw_parts[0][4].dtype
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        empty_v = torch.empty(0, device=device, dtype=dtype)
        idx = torch.stack([empty_i, empty_i, empty_i], dim=0)
        sp = torch.sparse_coo_tensor(idx, empty_v, [B, L, E], is_coalesced=True)
        return sp, empty_i, empty_i, empty_i, empty_i

    source = torch.cat([
        torch.full((v.shape[0],), s, device=v.device, dtype=torch.long)
        for s, _, _, _, v, _, _ in valid
    ], dim=0)
    b = torch.cat([x[1] for x in valid], dim=0)
    l = torch.cat([x[2] for x in valid], dim=0)
    e = torch.cat([x[3] for x in valid], dim=0)
    values = torch.cat([x[4] for x in valid], dim=0)
    tie = torch.cat([x[5] for x in valid], dim=0)
    raw = torch.cat([x[6] for x in valid], dim=0)

    keys = b * (L * E) + l * E + e
    del b, l, e
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_values = values[sort_idx]
    sorted_source = source[sort_idx]
    sorted_tie = tie[sort_idx]
    sorted_raw = raw[sort_idx]
    del keys, sort_idx, values, source, tie, raw
    unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
    del sorted_keys

    out_values = torch.full((unique_keys.shape[0],), float("-inf"), device=sorted_values.device, dtype=sorted_values.dtype)
    out_values.scatter_reduce_(0, inverse, sorted_values, reduce="amax", include_self=True)

    long_max = torch.iinfo(torch.long).max
    matched = sorted_values == out_values[inverse]
    candidate_source = torch.where(matched, sorted_source, torch.full_like(sorted_source, long_max))
    min_source = torch.full((unique_keys.shape[0],), long_max, device=sorted_values.device, dtype=torch.long)
    min_source.scatter_reduce_(0, inverse, candidate_source, reduce="amin", include_self=True)

    source_matched = matched & (sorted_source == min_source[inverse])
    candidate_tie = torch.where(source_matched, sorted_tie, torch.full_like(sorted_tie, long_max))
    min_tie = torch.full((unique_keys.shape[0],), long_max, device=sorted_values.device, dtype=torch.long)
    min_tie.scatter_reduce_(0, inverse, candidate_tie, reduce="amin", include_self=True)

    sorted_pos = torch.arange(sorted_values.shape[0], device=sorted_values.device, dtype=torch.long)
    sentinel = sorted_values.shape[0]
    final_matched = source_matched & (sorted_tie == min_tie[inverse])
    candidate_pos = torch.where(final_matched, sorted_pos, torch.full_like(sorted_pos, sentinel))
    arg_sorted_pos = torch.full((unique_keys.shape[0],), sentinel, device=sorted_values.device, dtype=torch.long)
    arg_sorted_pos.scatter_reduce_(0, inverse, candidate_pos, reduce="amin", include_self=True)
    arg_sorted_pos = arg_sorted_pos.clamp(max=max(sorted_values.shape[0] - 1, 0))

    win_source = sorted_source[arg_sorted_pos]
    win_raw = sorted_raw[arg_sorted_pos]
    # Winner tensors own their storage. Release reduction workspace before
    # materializing three output-coordinate arrays; no numerical work changes.
    del inverse, sorted_values, sorted_source, sorted_tie, sorted_raw
    del matched, candidate_source, min_source, source_matched, candidate_tie, min_tie
    del sorted_pos, final_matched, candidate_pos, arg_sorted_pos
    out_b = unique_keys // (L * E)
    rem = unique_keys % (L * E)
    out_l = rem // E
    out_e = rem % E
    sp = torch.sparse_coo_tensor(
        torch.stack([out_b, out_l, out_e], dim=0),
        out_values,
        [B, L, E],
        is_coalesced=True
    )
    return sp, unique_keys, win_source, win_raw, out_values


def _align_sparse_grad_to_keys(grad_sp, out_keys, raw_b, raw_l, raw_e, L, E, dtype):
    if raw_b.numel() == 0:
        return torch.empty(0, device=raw_b.device, dtype=dtype)
    raw_keys = raw_b * (L * E) + raw_l * E + raw_e
    grad_sp = grad_sp.coalesce()
    grad_keys = grad_sp.indices()[0] * (L * E) + grad_sp.indices()[1] * E + grad_sp.indices()[2]
    grad_vals = grad_sp.values()
    out_grad = torch.zeros(out_keys.shape[0], device=raw_b.device, dtype=dtype)
    if grad_keys.numel() > 0 and out_keys.numel() > 0:
        pos = torch.searchsorted(out_keys, grad_keys)
        valid_pos = pos < out_keys.numel()
        safe_pos = pos.clamp(max=max(out_keys.numel() - 1, 0))
        valid = valid_pos & (out_keys[safe_pos] == grad_keys)
        out_grad[safe_pos[valid]] = grad_vals[valid].to(dtype)
    raw_pos = torch.searchsorted(out_keys, raw_keys)
    raw_grad = torch.zeros(raw_keys.shape[0], device=raw_b.device, dtype=dtype)
    valid_raw_pos = raw_pos < out_keys.numel()
    safe_raw_pos = raw_pos.clamp(max=max(out_keys.numel() - 1, 0))
    valid = valid_raw_pos & (out_keys[safe_raw_pos] == raw_keys)
    raw_grad[valid] = out_grad[safe_raw_pos[valid]]
    return raw_grad


def _gather_sparse3d_grad_at_entities(grad_sp, b, l, e, L, E):
    if b.numel() == 0:
        return torch.empty(0, device=b.device, dtype=torch.float)
    grad_sp = grad_sp.coalesce()
    grad_keys = grad_sp.indices()[0] * (L * E) + grad_sp.indices()[1] * E + grad_sp.indices()[2]
    grad_vals = grad_sp.values()
    out = torch.zeros(b.shape[0], device=b.device, dtype=grad_vals.dtype)
    if grad_keys.numel() == 0:
        return out
    chunk_size = 1_000_000
    for start in range(0, int(b.numel()), chunk_size):
        end = min(start + chunk_size, int(b.numel()))
        query = b[start:end] * (L * E) + l[start:end] * E + e[start:end]
        pos = torch.searchsorted(grad_keys, query)
        valid_pos = pos < grad_keys.numel()
        safe_pos = pos.clamp(max=max(grad_keys.numel() - 1, 0))
        valid = valid_pos & (grad_keys[safe_pos] == query)
        if bool(valid.any().item()):
            out[start:end][valid] = grad_vals[safe_pos[valid]]
    return out


def _sparse3d_grad_values_at(grad_out, b, l, e, L, E, dtype):
    if b.numel() == 0:
        return torch.empty(0, device=b.device, dtype=dtype)
    if grad_out is None:
        return torch.zeros(b.shape[0], device=b.device, dtype=dtype)
    if getattr(grad_out, "is_sparse", False):
        return _gather_sparse3d_grad_at_entities(grad_out, b, l, e, L, E).to(dtype=dtype)
    return grad_out[b, l, e].to(dtype=dtype)


class Sparse3DNormalizeFunction(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, B, L, E, eps):
        sp_batch = sp_batch.contiguous().long()
        sp_level = sp_level.contiguous().long()
        sp_entity = sp_entity.contiguous().long()
        sp_value = sp_value.contiguous()

        if sp_value.is_cuda:
            module = get_module()
            out_value, denom, row_start, row_end = module.sparse3d_normalize_forward_cuda(
                sp_batch, sp_level, sp_value, B, L, float(eps)
            )
            used_cuda = True
        else:
            denom = torch.zeros(B * L, device=sp_value.device, dtype=sp_value.dtype)
            group = sp_batch * L + sp_level
            denom.index_add_(0, group, sp_value)
            denom_2d = denom.view(B, L)
            out_value = sp_value / denom[group].clamp(min=eps)
            row_start = torch.empty(0, device=sp_value.device, dtype=torch.long)
            row_end = torch.empty(0, device=sp_value.device, dtype=torch.long)
            denom = denom_2d
            used_cuda = False

        ctx.save_for_backward(sp_batch, sp_level, sp_entity, sp_value, denom, row_start, row_end)
        ctx.B = B
        ctx.L = L
        ctx.E = E
        ctx.eps = float(eps)
        ctx.used_cuda = used_cuda
        idx = torch.stack([sp_batch, sp_level, sp_entity], dim=0)
        return torch.sparse_coo_tensor(idx, out_value, [B, L, E], is_coalesced=True)

    @staticmethod
    def backward(ctx, grad_out):
        sp_batch, sp_level, sp_entity, sp_value, denom, row_start, row_end = ctx.saved_tensors

        if ctx.used_cuda:
            grad_out_values = _sparse3d_grad_values_at(
                grad_out, sp_batch, sp_level, sp_entity, ctx.L, ctx.E, sp_value.dtype
            )
            module = get_module()
            grad_value = module.sparse3d_normalize_backward_cuda(
                grad_out_values, sp_value, denom, row_start, row_end, ctx.eps
            )
        else:
            module = get_module()
            if getattr(grad_out, "is_sparse", False):
                grad_out_c = grad_out.coalesce()
                gidx = grad_out_c.indices()
                grad_value = module.sparse3d_normalize_backward_cpu(
                    gidx[0], gidx[1], gidx[2], grad_out_c.values().to(dtype=sp_value.dtype),
                    sp_batch, sp_level, sp_entity, sp_value, denom,
                    ctx.B, ctx.L, ctx.E, ctx.eps
                )
            else:
                grad_out_values = _sparse3d_grad_values_at(
                    grad_out, sp_batch, sp_level, sp_entity, ctx.L, ctx.E, sp_value.dtype
                )
                group = sp_batch * ctx.L + sp_level
                denom_flat = denom.reshape(-1)
                denom_at_value = denom_flat[group]
                denom_clamped = denom_at_value.clamp(min=ctx.eps)
                sum_gx = torch.zeros_like(denom_flat)
                sum_gx.index_add_(0, group, grad_out_values * sp_value)
                grad_value = grad_out_values / denom_clamped
                active = denom_at_value >= ctx.eps
                correction = sum_gx[group] / (denom_clamped * denom_clamped)
                grad_value = torch.where(active, grad_value - correction, grad_value)

        return None, None, None, grad_value, None, None, None, None


def normalize_sparse3d(s, eps=1e-7):
    s = s if s.is_coalesced() else s.coalesce()
    idx = s.indices()
    B, L, E = s.shape
    return Sparse3DNormalizeFunction.apply(idx[0], idx[1], idx[2], s.values(), B, L, E, eps)


def _compute_active_nodes(A):
    """Compute sorted unique active node indices from A [batch, L, E]."""
    # A.sum(1) -> [batch, E], nonzero -> active (batch, entity) pairs
    non_zero = torch.nonzero(A.sum(1))
    if non_zero.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=A.device)
    return torch.unique(non_zero[:, 1])


def _ind2ptr_cpu(index, size):
    """Build int32 CSR offsets without an edge-sized tensor of ones."""
    counts = torch.bincount(index, minlength=size)
    # In-place prefix sum avoids several full-entity int64 temporaries.
    np.cumsum(counts.numpy(), out=counts.numpy())
    ptr = torch.empty(size + 1, dtype=torch.int32)
    ptr[0] = 0
    ptr[1:].copy_(counts)
    return ptr


def _group_starts_and_counts(sorted_values, sorted_relations=None):
    """Compact group boundaries, with bounded int64 indexing temporaries."""
    n = sorted_values.numel()
    if not n:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)
    flags = torch.empty(n, dtype=torch.bool)
    flags[0] = True
    for lo in range(1, n, 1 << 20):
        hi = min(lo + (1 << 20), n)
        flags[lo:hi] = sorted_values[lo:hi] != sorted_values[lo - 1:hi - 1]
        if sorted_relations is not None:
            flags[lo:hi] |= sorted_relations[lo:hi] != sorted_relations[lo - 1:hi - 1]
    starts = torch.empty(int(flags.sum()), dtype=torch.int32)
    offset = 0
    for lo in range(0, n, 1 << 20):
        positions = flags[lo:lo + (1 << 20)].nonzero().flatten()
        count = positions.numel()
        starts[offset:offset + count] = positions.add_(lo)
        offset += count
    del flags
    counts = torch.empty_like(starts)
    np.subtract(starts.numpy()[1:], starts.numpy()[:-1], out=counts.numpy()[:-1])
    counts[-1] = n - starts[-1]
    return starts, counts


def _build_group_metadata(sorted_rows, sorted_rels, num_nodes, row_ptr=None):
    starts, counts = _group_starts_and_counts(sorted_rows, sorted_rels)
    relations = torch.empty(starts.numel(), dtype=torch.int16)
    for lo in range(0, starts.numel(), 1 << 20):
        hi = min(lo + (1 << 20), starts.numel())
        relations[lo:hi] = sorted_rels[starts[lo:hi].long()]
    if row_ptr is None:
        row_ptr = _ind2ptr_cpu(sorted_rows, num_nodes)
    group_ptr = torch.searchsorted(starts, row_ptr, out_int32=True)
    return group_ptr, relations, starts, counts


def build_csr_structure(row_indices, col_indices, r_indices, num_nodes):
    """Build the same ordered CSR using compact CPU buffers, one side at a time."""
    device = row_indices.device
    row_cpu = row_indices.detach().cpu()
    col_cpu = col_indices.detach().cpu()
    r_cpu = r_indices.detach().cpu()
    rel_base = int(r_cpu.max()) + 1 if r_cpu.numel() else 1

    def side(source, target):
        # Keep the exact original int64 sort keys and torch.argsort tie order.
        keys = source.to(dtype=torch.long, copy=True)
        keys.mul_(rel_base).add_(r_cpu)
        order = keys.argsort()
        del keys
        sorted_rows = source[order].to(torch.int32)
        columns = target[order].to(torch.int32)
        relations = r_cpu[order].to(torch.int16)
        compact_order = order.to(torch.int32)
        del order
        ptr = _ind2ptr_cpu(sorted_rows, num_nodes)
        groups = _build_group_metadata(sorted_rows, relations, num_nodes, ptr)
        return tuple(t.to(device) for t in (ptr, columns, relations, compact_order, *groups))

    ori = side(row_cpu, col_cpu)
    inv = side(col_cpu, row_cpu)
    return (*ori[:4], *inv[:4], *ori[4:], *inv[4:])


class FlashLogFunction(autograd.Function):
    @staticmethod
    def forward(ctx, A, w, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges):
        module = get_module()

        B, L, E = A.shape
        A_flat = A.reshape(B * L, E)
        w_flat = w.reshape(B * L, -1)

        A_f = A_flat if A_flat.is_floating_point() else A_flat.float()
        w_f = w_flat if w_flat.is_floating_point() else w_flat.float()

        if A.is_cuda:
            out_ind, out_ori, out_inv = module.flashlog_forward_cuda(
                A_f, w_f, active_nodes,
                torch.empty(0, device=A.device, dtype=torch.int32),
                ori_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                ori_mask,
                torch.empty(0, device=A.device, dtype=torch.int32),
                inv_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                inv_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges > 0, topk_edges, 0
            )
        else:
            out_ind, out_ori, out_inv = module.flashlog_forward_cpu(
                A_f, w_f, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                r_size, wot_i, topk_edges
            )

        out_ind = out_ind.reshape(B, L, E)
        out_ori = out_ori.reshape(B, L, E)
        out_inv = out_inv.reshape(B, L, E)

        ctx.save_for_backward(A, w, active_nodes,
                              ori_col_ind, ori_mask,
                              ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                              inv_col_ind, inv_mask,
                              inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count)
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.topk_edges = topk_edges
        return out_ind, out_ori, out_inv

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes,
         ori_col_ind, ori_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_col_ind, inv_mask,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L

        if A.is_cuda:
            A_f = A.reshape(BL, E).contiguous().float()
            w_f = w.reshape(BL, -1).contiguous().float()
            grad_ind_flat = grad_ind.reshape(BL, E).contiguous().float()
            grad_ori_flat = grad_ori.reshape(BL, E).contiguous().float()
            grad_inv_flat = grad_inv.reshape(BL, E).contiguous().float()
            grad_A, grad_w = module.flashlog_backward_cuda(
                grad_ind_flat, grad_ori_flat, grad_inv_flat,
                torch.zeros(BL, E, device=A.device, dtype=A_f.dtype),
                torch.zeros(BL, E, device=A.device, dtype=A_f.dtype),
                A_f, w_f, active_nodes,
                torch.empty(0, device=A.device, dtype=torch.int32),
                ori_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                ori_mask.float(),
                torch.empty(0, device=A.device, dtype=torch.int32),
                inv_col_ind, torch.empty(0, device=A.device, dtype=torch.int16),
                inv_mask.float(),
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                ctx.r_size, ctx.wot_i, ctx.topk_edges > 0, ctx.topk_edges, 0
            )
        else:
            A_f = A.reshape(BL, E).contiguous()
            w_f = w.reshape(BL, -1).contiguous()
            grad_ind_flat = grad_ind.reshape(BL, E).contiguous().to(dtype=A_f.dtype)
            grad_ori_flat = grad_ori.reshape(BL, E).contiguous().to(dtype=A_f.dtype)
            grad_inv_flat = grad_inv.reshape(BL, E).contiguous().to(dtype=A_f.dtype)
            grad_A, grad_w = module.flashlog_backward_cpu(
                grad_ind_flat, grad_ori_flat, grad_inv_flat,
                A_f, w_f, active_nodes,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                ctx.r_size, ctx.wot_i, ctx.topk_edges
            )

        grad_A = grad_A.reshape(B, L, E).to(dtype=A.dtype)
        grad_w = grad_w.reshape(B, L, -1).to(dtype=w.dtype)

        return (grad_A, grad_w, None,
                None, None,
                None, None, None, None,
                None, None,
                None, None, None, None,
                None, None, None)


def _dense_topc_state(A, topk_nodes):
    """Prune all propagation sources, including identity, per (item, rule).

    The Boolean selection is detached; multiplying outside no_grad preserves
    gradients only through admitted graph sources. Entity order breaks ties.
    """
    if topk_nodes is None:
        return A
    if topk_nodes < 0:
        raise ValueError("topk_nodes/c must be nonnegative")
    if topk_nodes >= A.shape[-1]:
        return A
    with torch.no_grad():
        nonzero = A != 0
        over_limit = nonzero.sum(-1, keepdim=True) > topk_nodes
        if not bool(over_limit.any()):
            return A
        if topk_nodes == 0:
            keep = ~over_limit
        else:
            # Zero entries do not compete with negative nonzero source states.
            scores = A.masked_fill(~nonzero, -float('inf'))
            cutoff = scores.topk(topk_nodes, dim=-1).values.min(-1, keepdim=True).values
            above = nonzero & (A > cutoff)
            tied = nonzero & (A == cutoff)
            remaining = topk_nodes - above.sum(-1, keepdim=True)
            chosen = above | (tied & (tied.cumsum(-1) <= remaining))
            keep = ~over_limit | chosen
    return A * keep


def vectorized_operation_csr(A, w, csr_struct, mask_values, r_size,
                              weight=None, wot_i=False,
                              use_topk=False, topk_nodes=100000, topk_edges=0,
                              prepared_masks=None):
    """
    CPU-capable entry point with topk support, with the same arguments as the CUDA implementation.
    Uses group-based iteration aligned with CUDA kernel.

    csr_struct: output of build_csr_structure() (16-tuple)
    mask_values: raw mask tensor (weight applied here in Python)
    """
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    prepared = _prepared_mask_pair(prepared_masks) if prepared_masks is not None else _prepared_mask_pair(mask_values)
    if prepared is not None:
        ori_mask, inv_mask = prepared
    else:
        ori_mask, inv_mask = prepare_csr_masks(csr_struct, mask_values, weight)

    A = _dense_topc_state(A, topk_nodes if use_topk else None)
    # The union only schedules work; admission was decided per item/rule.
    active_nodes = _compute_active_nodes(A)

    # Level 2: effective topk_edges (0 means no pruning → pass huge value)
    effective_topk = topk_edges if (use_topk and topk_edges > 0) else (1 << 60)

    outputs = FlashLogFunction.apply(
        A, w, active_nodes,
        ori_col_ind, ori_mask,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_col_ind, inv_mask,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        r_size, wot_i, effective_topk
    )
    return outputs


class FlashLogDenseSumFunction(autograd.Function):
    """
    Active nodes and masks are computed on GPU in C++.
    Eliminates Python-side nonzero/unique sync and temporary tensors.
    Supports optional top-k pruning (Level 1: nodes, Level 2: edges per node).
    """
    @staticmethod
    def forward(ctx, A, w, ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, r_size, wot_i,
                use_topk, topk_nodes, topk_edges, agg_mode):
        module = get_module()
        ori_row_ptr = ori_row_ptr.int()
        ori_col_ind = ori_col_ind.int()
        ori_r_ind = ori_r_ind.short()
        ori_order = ori_order.int()
        inv_row_ptr = inv_row_ptr.int()
        inv_col_ind = inv_col_ind.int()
        inv_r_ind = inv_r_ind.short()
        inv_order = inv_order.int()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        # Active nodes on GPU (no CPU sync)
        active_nodes = module.compute_active_nodes_cuda(A_flat)

        # The entry point already applied per-item/per-rule source admission.

        prepared = _prepared_mask_pair(mask_values)
        if prepared is not None:
            ori_mask, inv_mask = prepared
        else:
            mask_f = (mask_values if mask_values.dtype == torch.bool else mask_values.float()).contiguous()
            w_tensor = weight.to(device=A.device, dtype=torch.float).squeeze(-1) if weight is not None else torch.empty(0, device=A.device)
            ori_mask = module.apply_mask_cuda(mask_f, ori_order, w_tensor)
            inv_mask = module.apply_mask_cuda(mask_f, inv_order, w_tensor)

        # CSR forward pass
        out_ind, out_ori, out_inv = module.flashlog_forward_cuda(
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            r_size, wot_i, use_topk, topk_edges, agg_mode
        )

        ctx.save_for_backward(A, w, active_nodes, out_ori.reshape(B, L, E), out_inv.reshape(B, L, E),
                              ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
                              inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
                              ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                              inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count)
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.use_topk = use_topk
        ctx.topk_edges = topk_edges
        ctx.agg_mode = agg_mode

        return (out_ind.reshape(B, L, E),
                out_ori.reshape(B, L, E),
                out_inv.reshape(B, L, E))

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes, out_ori, out_inv,
         ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
         inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        grad_A, grad_w = module.flashlog_backward_cuda(
            grad_ind.reshape(BL, E).contiguous(),
            grad_ori.reshape(BL, E).contiguous(),
            grad_inv.reshape(BL, E).contiguous(),
            out_ori.reshape(BL, E).contiguous(),
            out_inv.reshape(BL, E).contiguous(),
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            ctx.r_size, ctx.wot_i, ctx.use_topk, ctx.topk_edges, ctx.agg_mode
        )

        grad_A = grad_A.reshape(B, L, E)
        grad_w = grad_w.reshape(B, L, -1)

        return (grad_A, grad_w,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None)


class FlashLogDenseMaxFunction(autograd.Function):
    @staticmethod
    def forward(ctx, A, w, ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, r_size, wot_i,
                use_topk, topk_nodes, topk_edges):
        module = get_module()
        ori_row_ptr = ori_row_ptr.int()
        ori_col_ind = ori_col_ind.int()
        ori_r_ind = ori_r_ind.short()
        ori_order = ori_order.int()
        inv_row_ptr = inv_row_ptr.int()
        inv_col_ind = inv_col_ind.int()
        inv_r_ind = inv_r_ind.short()
        inv_order = inv_order.int()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        active_nodes = module.compute_active_nodes_cuda(A_flat)
        # The entry point already applied per-item/per-rule source admission.

        prepared = _prepared_mask_pair(mask_values)
        if prepared is not None:
            ori_mask, inv_mask = prepared
        else:
            mask_f = (mask_values if mask_values.dtype == torch.bool else mask_values.float()).contiguous()
            w_tensor = weight.to(device=A.device, dtype=torch.float).squeeze(-1) if weight is not None else torch.empty(0, device=A.device)
            ori_mask = module.apply_mask_cuda(mask_f, ori_order, w_tensor)
            inv_mask = module.apply_mask_cuda(mask_f, inv_order, w_tensor)

        out_ind, out_ori, out_inv, ori_arg, inv_arg = module.flashlog_forward_max_cuda(
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            r_size, wot_i, use_topk, topk_edges
        )

        ctx.save_for_backward(
            A, w, active_nodes, ori_arg, inv_arg,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count
        )
        ctx.r_size = r_size
        ctx.wot_i = wot_i
        ctx.use_topk = use_topk
        ctx.topk_edges = topk_edges

        return (out_ind.reshape(B, L, E),
                out_ori.reshape(B, L, E),
                out_inv.reshape(B, L, E))

    @staticmethod
    def backward(ctx, grad_ind, grad_ori, grad_inv):
        (A, w, active_nodes, ori_arg, inv_arg,
         ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
         inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
         ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
         inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = ctx.saved_tensors
        module = get_module()

        B, L, E = A.shape
        BL = B * L
        A_flat = A.reshape(BL, E).contiguous().float()
        w_flat = w.reshape(BL, -1).contiguous().float()

        grad_A, grad_w = module.flashlog_backward_max_cuda(
            grad_ind.reshape(BL, E).contiguous(),
            grad_ori.reshape(BL, E).contiguous(),
            grad_inv.reshape(BL, E).contiguous(),
            ori_arg.contiguous(),
            inv_arg.contiguous(),
            A_flat, w_flat, active_nodes,
            ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
            inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            ctx.r_size, ctx.wot_i, ctx.use_topk, ctx.topk_edges
        )

        grad_A = grad_A.reshape(B, L, E)
        grad_w = grad_w.reshape(B, L, -1)

        return (grad_A, grad_w,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None, None,
                None, None, None)


@torch.no_grad()
def _sparse_topc_entries(sp_batch, sp_level, sp_entity, sp_value, topk_nodes, num_rules):
    """Limit nodes independently in each (batch item, rule) of a coalesced state.

    Each coordinate occurs once and is ordered by (batch, rule, entity).
    Count nonzero nodes first; rank only groups whose count exceeds c.
    Never allocate a dense [B, L, E] state or an E-sized selection mask.
    """
    if topk_nodes is None:
        return None
    if topk_nodes < 0:
        raise ValueError("topk_nodes/c must be nonnegative")
    # Total nnz is an upper bound on every group's active-node count.
    if sp_entity.numel() <= topk_nodes:
        return None
    groups, inverse = torch.unique_consecutive(
        sp_batch * num_rules + sp_level, return_inverse=True)
    nonzero = sp_value != 0
    counts = torch.zeros(groups.numel(), dtype=torch.long, device=sp_entity.device)
    counts.scatter_add_(0, inverse, nonzero.long())
    over_limit = counts > topk_nodes
    if not bool(over_limit.any()):
        return None
    # Leave all entries (including explicit zeros) of small groups untouched.
    keep = ~over_limit[inverse]
    if topk_nodes == 0:
        return torch.nonzero(keep, as_tuple=True)[0]

    candidates = torch.nonzero(over_limit[inverse] & nonzero, as_tuple=True)[0]
    # Segmented top-c: sort values, then stably group them by (item, rule).
    # Scores are never summed or compared for admission across different groups.
    # Stable ties retain the smaller entity ID within the coalesced group.
    ranked = candidates[torch.argsort(sp_value[candidates], descending=True, stable=True)]
    ranked = ranked[torch.argsort(inverse[ranked], stable=True)]
    _, sizes = torch.unique_consecutive(inverse[ranked], return_counts=True)
    starts = sizes.cumsum(0) - sizes
    ranks = torch.arange(ranked.numel(), device=ranked.device) - torch.repeat_interleave(starts, sizes)
    keep[ranked[ranks < topk_nodes]] = True
    return torch.nonzero(keep, as_tuple=True)[0]


def _prune_sparse_sources(ctx, batch, level, entity, value, topk_nodes, num_rules):
    """Select one common input for fact and identity propagation."""
    entries = _sparse_topc_entries(batch, level, entity, value, topk_nodes, num_rules)
    ctx.pruned_entries = entries
    ctx.original_input_size = value.numel()
    if entries is None:
        return batch, level, entity, value
    return batch[entries], level[entries], entity[entries], value[entries]


def _restore_pruned_gradient(ctx, gradient):
    if ctx.pruned_entries is None:
        return gradient
    restored = gradient.new_zeros(ctx.original_input_size)
    restored[ctx.pruned_entries] = gradient
    return restored


class FlashLogFunctionSparse3DTopKSum(autograd.Function):
    """Sparse sum with forward-local expansion-to-output index caching.

    FLASHLOG_CACHE_BACKWARD_INDEX=0 retains the search-based reference path.
    The cache never crosses batches or reasoning steps. Inference callers
    pass cache_backward=False. Expansion is processed in one pass; backward
    reuses its saved provenance without rematerializing graph contributions.
    """
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i, topk_edges,
                edge_weight_is_score=False, topk_nodes=None, cache_backward=True):
        ctx.cache_index = bool(
            cache_backward and any(ctx.needs_input_grad)
            and os.environ.get("FLASHLOG_CACHE_BACKWARD_INDEX", "1") != "0")
        sp_batch, sp_level, sp_entity, sp_value = _prune_sparse_sources(
            ctx, sp_batch, sp_level, sp_entity, sp_value, topk_nodes, w.size(1))
        t_stage = time.perf_counter()
        module = get_module()
        _kernel_log("sparse3d_topk_sum.get_module", t_stage)
        t_stage = time.perf_counter()
        ori_col_ind = ori_col_ind.int()
        ori_order = ori_order.int()
        inv_col_ind = inv_col_ind.int()
        inv_order = inv_order.int()
        mask_f = mask_values
        edge_weight = weight if weight is not None else torch.empty(0, device=sp_value.device)
        edge_weight_is_score = bool(edge_weight_is_score)
        _kernel_log("sparse3d_topk_sum.prepare_inputs", t_stage)

        t_stage = time.perf_counter()
        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets,
         ori_meta_edge, inv_meta_edge) = module.flashlog_forward_sparse3d_topk_cuda(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_order,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_order,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_f, edge_weight,
            B, L, E, r_size, wot_i, topk_edges, edge_weight_is_score
        )
        _kernel_log("sparse3d_topk_sum.cuda_forward", t_stage)

        t_stage = time.perf_counter()
        # A hard-masked edge is identically zero in both value and all
        # trainable derivatives. Do not materialize a new zero coordinate
        # for it: that coordinate would otherwise expand again next step.
        # Keep unmasked zero values and identity, whose derivatives can be
        # nonzero. Apply this AFTER top-k selection so edge quotas stay exact.
        visible_masks = []
        visible_sizes = []
        def visible_part(b, l, e, v, edges):
            keep = mask_f.view(-1)[edges.long()] != 0
            if bool(keep.all()):
                visible_masks.append(None)
                visible_sizes.append(v.numel())
                return b, l, e, v
            part = b[keep], l[keep], e[keep], v[keep]
            visible_masks.append(keep)
            visible_sizes.append(part[3].numel())
            return part

        out = _coalesce_sparse3d_sum_raw(
            (visible_part(ori_b, ori_l, ori_e, ori_v, ori_meta_edge),
             visible_part(inv_b, inv_l, inv_e, inv_v, inv_meta_edge),
             (ind_b, ind_l, ind_e, ind_v)),
            B, L, E,
            use_cuda_kernel=_kernel_cuda_reduce_enabled(), return_inverse=ctx.cache_index
        )
        cache = ()
        if ctx.cache_index:
            out, positions = out
            raw_maps = []
            offset = 0
            for keep, size in zip(visible_masks, visible_sizes):
                raw = positions[offset:offset + size]
                if keep is not None:
                    raw = torch.full((keep.numel(),), -1, device=positions.device, dtype=positions.dtype)
                    raw[keep] = positions[offset:offset + size]
                raw_maps.append(raw)
                offset += size
            cache = (out.indices(), *raw_maps, positions[offset:])
        _kernel_log("sparse3d_topk_sum.reduce", t_stage)

        t_stage = time.perf_counter()
        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value, w,
            None if ctx.cache_index else ori_b, None if ctx.cache_index else ori_l,
            None if ctx.cache_index else ori_e, None if ctx.cache_index else ori_v,
            ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
            None if ctx.cache_index else inv_b, None if ctx.cache_index else inv_l,
            None if ctx.cache_index else inv_e, None if ctx.cache_index else inv_v,
            inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
            edge_weight if weight is not None and bool(getattr(weight, "requires_grad", False)) else torch.empty(0, device=sp_value.device),
            *cache
        )
        ctx.has_edge_weight = weight is not None and bool(getattr(weight, "requires_grad", False))
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i = B, L, E, r_size, wot_i
        _kernel_log("sparse3d_topk_sum.save_ctx", t_stage)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
         inv_b, inv_l, inv_e, inv_v,
         inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
         edge_weight) = ctx.saved_tensors[:24]
        module = get_module()

        if ctx.cache_index:
            out_indices, ori_pos, inv_pos, ind_pos = ctx.saved_tensors[24:]
            aligned = _align_sparse_grad_to_output(grad_out, out_indices, ctx.L, ctx.E)
            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_cached_cuda(
                aligned, ori_pos, inv_pos, ind_pos,
                sp_batch, sp_level, sp_value, w,
                ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.r_size, ctx.wot_i
            )
        else:
            grad_out = grad_out.coalesce()
            gval = grad_out.values()
            gidx = grad_out.indices()
            gb, gl, ge = gidx[0], gidx[1], gidx[2]

            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_topk_cuda(
                gval, gb, gl, ge,
                gval, gb, gl, ge,
                gval, gb, gl, ge,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e,
                ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_b, inv_l, inv_e,
                inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
        grad_edge_weight = None
        if ctx.has_edge_weight:
            edge_weight_flat = edge_weight.view(-1)
            grad_edge_flat = torch.zeros_like(edge_weight_flat)
            if ctx.cache_index:
                grad_ori_raw = _cached_raw_grad(aligned, ori_pos)
                grad_inv_raw = _cached_raw_grad(aligned, inv_pos)
            else:
                grad_ori_raw = _gather_sparse3d_grad_at_entities(grad_out, ori_b, ori_l, ori_e, ctx.L, ctx.E)
                grad_inv_raw = _gather_sparse3d_grad_at_entities(grad_out, inv_b, inv_l, inv_e, ctx.L, ctx.E)

            def add_edge_weight_grad(raw_grad, meta_entry, meta_rel, meta_mask, meta_edge, rel_offset):
                if meta_edge.numel() == 0:
                    return
                chunk_size = 1_000_000
                for start in range(0, int(meta_edge.numel()), chunk_size):
                    end = min(start + chunk_size, int(meta_edge.numel()))
                    entry = meta_entry[start:end].long()
                    rel = meta_rel[start:end].long()
                    edge = meta_edge[start:end].long()
                    b = sp_batch[entry]
                    l = sp_level[entry]
                    rule_weight = w[b, l, rel_offset + rel]
                    edge_score = score_function(edge_weight_flat[edge])
                    mask_score = meta_mask[start:end].to(dtype=raw_grad.dtype)
                    contrib = raw_grad[start:end] * sp_value[entry] * rule_weight * mask_score * (1.0 - edge_score)
                    grad_edge_flat.index_add_(0, edge, contrib)

            add_edge_weight_grad(grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_meta_edge, 0)
            add_edge_weight_grad(grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_meta_edge, ctx.r_size)
            grad_edge_weight = grad_edge_flat.view_as(edge_weight)

        grad_sp_value = _restore_pruned_gradient(ctx, grad_sp_value)
        grads = (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, grad_edge_weight,
            None, None, None, None, None, None, None
        )
        return (grads + (None, None))[:len(ctx.needs_input_grad)]


class FlashLogFunctionSparse3DAllEdgesSum(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                B, L, E, r_size, wot_i, cache_backward=True):
        ctx.cache_index = bool(cache_backward and any(ctx.needs_input_grad)
                               and os.environ.get("FLASHLOG_CACHE_BACKWARD_INDEX", "1") != "0")
        module = get_module()

        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets) = module.flashlog_forward_sparse3d_alledges_cuda(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind.int(), ori_mask.float(),
            ori_row_group_ptr.int(), ori_group_rel.short(),
            ori_group_edge_start.int(), ori_group_edge_count.int(),
            inv_col_ind.int(), inv_mask.float(),
            inv_row_group_ptr.int(), inv_group_rel.short(),
            inv_group_edge_start.int(), inv_group_edge_count.int(),
            B, L, E, r_size, wot_i
        )

        out = _coalesce_sparse3d_sum_raw(
            ((ori_b, ori_l, ori_e, ori_v),
             (inv_b, inv_l, inv_e, inv_v),
             (ind_b, ind_l, ind_e, ind_v)),
            B, L, E,
            use_cuda_kernel=_kernel_cuda_reduce_enabled(), return_inverse=ctx.cache_index
        )
        cache = ()
        if ctx.cache_index:
            out, positions = out
            no, ni = ori_v.numel(), inv_v.numel()
            cache = (out.indices(), positions[:no], positions[no:no + ni], positions[no + ni:])

        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value, w,
            None if ctx.cache_index else ori_b, None if ctx.cache_index else ori_l, None if ctx.cache_index else ori_e,
            ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
            None if ctx.cache_index else inv_b, None if ctx.cache_index else inv_l, None if ctx.cache_index else inv_e,
            inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets,
            *cache
        )
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i = B, L, E, r_size, wot_i
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e,
         ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
         inv_b, inv_l, inv_e,
         inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets) = ctx.saved_tensors[:19]
        module = get_module()

        if ctx.cache_index:
            out_indices, ori_pos, inv_pos, ind_pos = ctx.saved_tensors[19:]
            aligned = _align_sparse_grad_to_output(grad_out, out_indices, ctx.L, ctx.E)
            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_cached_cuda(
                aligned, ori_pos, inv_pos, ind_pos,
                sp_batch, sp_level, sp_value, w,
                ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.r_size, ctx.wot_i)
        else:
            grad_out = grad_out.coalesce()
            gval = grad_out.values()
            gidx = grad_out.indices()
            gb, gl, ge = gidx[0], gidx[1], gidx[2]

            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_topk_cuda(
                gval, gb, gl, ge,
                gval, gb, gl, ge,
                gval, gb, gl, ge,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e,
                ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets,
                inv_b, inv_l, inv_e,
                inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )
        grads = (
            None, None, None, grad_sp_value, grad_w,
            None, None,
            None, None, None, None,
            None, None,
            None, None, None, None,
            None, None, None, None, None
        )

        return (grads + (None,))[:len(ctx.needs_input_grad)]


class FlashLogFunctionSparse3DTopKMax(autograd.Function):
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_row_ptr, ori_col_ind, ori_r_ind, ori_order,
                inv_row_ptr, inv_col_ind, inv_r_ind, inv_order,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                mask_values, weight, B, L, E, r_size, wot_i, topk_edges,
                edge_weight_is_score=False, topk_nodes=None, cache_backward=True):
        ctx.cache_index = bool(cache_backward and any(ctx.needs_input_grad)
                               and os.environ.get("FLASHLOG_CACHE_BACKWARD_INDEX", "1") != "0")
        sp_batch, sp_level, sp_entity, sp_value = _prune_sparse_sources(
            ctx, sp_batch, sp_level, sp_entity, sp_value, topk_nodes, w.size(1))
        t_stage = time.perf_counter()
        module = get_module()
        _kernel_log("sparse3d_topk_max.get_module", t_stage)
        t_stage = time.perf_counter()
        ori_col_ind = ori_col_ind.int()
        ori_order = ori_order.int()
        inv_col_ind = inv_col_ind.int()
        inv_order = inv_order.int()
        mask_f = mask_values
        edge_weight = weight if weight is not None else torch.empty(0, device=sp_value.device)
        edge_weight_is_score = bool(edge_weight_is_score)
        _kernel_log("sparse3d_topk_max.prepare_inputs", t_stage)

        t_stage = time.perf_counter()
        (ori_b, ori_l, ori_e, ori_v,
         inv_b, inv_l, inv_e, inv_v,
         ind_b, ind_l, ind_e, ind_v,
         ori_meta_entry, ori_meta_rel, ori_meta_mask,
         inv_meta_entry, inv_meta_rel, inv_meta_mask,
         ori_offsets, inv_offsets,
         ori_meta_edge, inv_meta_edge) = module.flashlog_forward_sparse3d_topk_cuda(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_order,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_order,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_f, edge_weight,
            B, L, E, r_size, wot_i, topk_edges, edge_weight_is_score
        )
        _kernel_log("sparse3d_topk_max.cuda_forward", t_stage)

        t_stage = time.perf_counter()
        ind_tie = torch.zeros(ind_v.shape[0], device=ind_v.device, dtype=torch.long)
        out, out_keys, win_source, win_raw, _ = _coalesce_sparse3d_max_raw_parts(
            ((0, ori_b, ori_l, ori_e, ori_v, ori_meta_edge),
             (1, inv_b, inv_l, inv_e, inv_v, inv_meta_edge),
             (2, ind_b, ind_l, ind_e, ind_v, ind_tie)),
            B, L, E
        )
        _kernel_log("sparse3d_topk_max.reduce", t_stage)

        t_stage = time.perf_counter()
        cache = ()
        if ctx.cache_index:
            positions = _max_sparse3d_winner_positions(
                win_source, win_raw, (ori_meta_entry.numel(), inv_meta_entry.numel(), ind_v.numel()))
            cache = (out.indices(), *positions)
        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value, w,
            None if ctx.cache_index else ori_b, None if ctx.cache_index else ori_l, None if ctx.cache_index else ori_e,
            ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
            None if ctx.cache_index else inv_b, None if ctx.cache_index else inv_l, None if ctx.cache_index else inv_e,
            inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
            None if ctx.cache_index else ind_b, None if ctx.cache_index else ind_l, None if ctx.cache_index else ind_e,
            None if ctx.cache_index else out_keys, None if ctx.cache_index else win_source, None if ctx.cache_index else win_raw,
            edge_weight if weight is not None and bool(getattr(weight, "requires_grad", False)) else torch.empty(0, device=sp_value.device),
            *cache
        )
        ctx.has_edge_weight = weight is not None and bool(getattr(weight, "requires_grad", False))
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i = B, L, E, r_size, wot_i
        _kernel_log("sparse3d_topk_max.save_ctx", t_stage)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e,
         ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_offsets, ori_meta_edge,
         inv_b, inv_l, inv_e,
         inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_offsets, inv_meta_edge,
         ind_b, ind_l, ind_e,
         out_keys, win_source, win_raw,
         edge_weight) = ctx.saved_tensors[:28]
        module = get_module()

        if ctx.cache_index:
            out_indices, ori_pos, inv_pos, ind_pos = ctx.saved_tensors[28:]
            aligned = _align_sparse_grad_to_output(grad_out, out_indices, ctx.L, ctx.E)
            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_max_cached_cuda(
                aligned, ori_pos, inv_pos, ind_pos,
                sp_batch, sp_level, sp_value, w,
                ori_meta_entry, ori_meta_rel, ori_meta_mask,
                inv_meta_entry, inv_meta_rel, inv_meta_mask,
                ctx.B, ctx.L, ctx.r_size, ctx.wot_i)
            if ctx.has_edge_weight:
                grad_ori_raw = _cached_raw_grad(aligned, ori_pos)
                grad_inv_raw = _cached_raw_grad(aligned, inv_pos)
        else:
            out_b = out_keys // (ctx.L * ctx.E)
            rem = out_keys % (ctx.L * ctx.E)
            out_l = rem // ctx.E
            out_e = rem % ctx.E
            out_grad = _align_sparse_grad_to_keys(
                grad_out, out_keys, out_b, out_l, out_e, ctx.L, ctx.E, sp_value.dtype
            )

            grad_ori_raw = torch.zeros(ori_meta_entry.shape[0], device=sp_value.device, dtype=sp_value.dtype)
            grad_inv_raw = torch.zeros(inv_meta_entry.shape[0], device=sp_value.device, dtype=sp_value.dtype)
            ori_win = win_source == 0
            inv_win = win_source == 1
            ind_win = win_source == 2
            if ori_win.any():
                grad_ori_raw[win_raw[ori_win].long()] = out_grad[ori_win]
            if inv_win.any():
                grad_inv_raw[win_raw[inv_win].long()] = out_grad[inv_win]

            if ind_win.any():
                ind_raw = win_raw[ind_win].long()
                gindv = out_grad[ind_win]
                gindb = ind_b[ind_raw].long()
                gindl = ind_l[ind_raw].long()
                ginde = ind_e[ind_raw].long()
            else:
                gindv = torch.empty(0, device=sp_value.device, dtype=sp_value.dtype)
                gindb = torch.empty(0, device=sp_value.device, dtype=torch.long)
                gindl = torch.empty(0, device=sp_value.device, dtype=torch.long)
                ginde = torch.empty(0, device=sp_value.device, dtype=torch.long)

            grad_sp_value, grad_w = module.flashlog_backward_sparse3d_topk_aligned_cuda(
                grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask,
                grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask,
                gindv, gindb, gindl, ginde,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
            )

        grad_edge_weight = None
        if ctx.has_edge_weight:
            edge_weight_flat = edge_weight.view(-1)
            grad_edge_flat = torch.zeros_like(edge_weight_flat)

            def add_edge_weight_grad(raw_grad, meta_entry, meta_rel, meta_mask, meta_edge, rel_offset):
                if meta_edge.numel() == 0:
                    return
                chunk_size = 1_000_000
                for start in range(0, int(meta_edge.numel()), chunk_size):
                    end = min(start + chunk_size, int(meta_edge.numel()))
                    entry = meta_entry[start:end].long()
                    rel = meta_rel[start:end].long()
                    edge = meta_edge[start:end].long()
                    b = sp_batch[entry]
                    l = sp_level[entry]
                    rule_weight = w[b, l, rel_offset + rel]
                    edge_score = score_function(edge_weight_flat[edge])
                    mask_score = meta_mask[start:end].to(dtype=raw_grad.dtype)
                    contrib = raw_grad[start:end] * sp_value[entry] * rule_weight * mask_score * (1.0 - edge_score)
                    grad_edge_flat.index_add_(0, edge, contrib)

            add_edge_weight_grad(grad_ori_raw, ori_meta_entry, ori_meta_rel, ori_meta_mask, ori_meta_edge, 0)
            add_edge_weight_grad(grad_inv_raw, inv_meta_entry, inv_meta_rel, inv_meta_mask, inv_meta_edge, ctx.r_size)
            grad_edge_weight = grad_edge_flat.view_as(edge_weight)

        grad_sp_value = _restore_pruned_gradient(ctx, grad_sp_value)
        grads = (
            None, None, None, grad_sp_value, grad_w,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, grad_edge_weight,
            None, None, None, None, None, None, None
        )
        return (grads + (None, None))[:len(ctx.needs_input_grad)]


def dense_csr_sum_cuda(A, w, csr_struct, mask_values, r_size,
                                 weight=None, wot_i=False,
                                 use_topk=False, topk_nodes=100000, topk_edges=0,
                                 agg_mode="sum", prepared_masks=None):
    """
    CUDA entry point for dense sum aggregation.
    - active_nodes computed on GPU (no CPU sync)
    - mask can be preordered once per model forward via prepared_masks
    - Optional top-k pruning (use_topk=True):
        Level 1: limit nonzero graph sources independently per (item, rule)
        Level 2: limit edges per node to topk_edges

    csr_struct: output of build_csr_structure() (includes sort orders)
    mask_values: raw mask tensor, used when prepared_masks is not supplied
    """
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    agg_id = 1 if agg_mode == "max" else 0
    A = _dense_topc_state(A, topk_nodes if use_topk else None)
    mask_input = prepared_masks if prepared_masks is not None else mask_values
    weight_input = None if prepared_masks is not None else weight
    outputs = FlashLogDenseSumFunction.apply(
        A, w,
        ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
        inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        mask_input, weight_input,
        r_size, wot_i,
        use_topk, topk_nodes, topk_edges, agg_id
    )
    return outputs


def dense_csr_max(A, w, csr_struct, mask_values, r_size,
                                          weight=None, wot_i=False,
                                          use_topk=False, topk_nodes=100000, topk_edges=0,
                                          prepared_masks=None):
    (ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
     inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
     ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
     inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count) = csr_struct

    A = _dense_topc_state(A, topk_nodes if use_topk else None)
    mask_input = prepared_masks if prepared_masks is not None else mask_values
    weight_input = None if prepared_masks is not None else weight
    outputs = FlashLogDenseMaxFunction.apply(
        A, w,
        ori_row_ptr, ori_col_ind, ori_r_ind, order_ori,
        inv_row_ptr, inv_col_ind, inv_r_ind, order_inv,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        mask_input, weight_input,
        r_size, wot_i,
        use_topk, topk_nodes, topk_edges
    )
    return outputs


class FlashLogFunctionSparse3DCPUSum(autograd.Function):
    """CPU sparse 3D sum path with C++ key-sort reduce, matching CUDA sum dispatch."""
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                B, L, E, r_size, wot_i, topk_edges):
        module = get_module()
        (out_b, out_l, out_e, out_v,
         ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m,
         inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m) = module.flashlog_forward_sparse3d_sum_cpu(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            B, L, E, r_size, wot_i, topk_edges
        )
        out = torch.sparse_coo_tensor(
            torch.stack([out_b, out_l, out_e], dim=0),
            out_v.to(dtype=sp_value.dtype),
            [B, L, E],
            is_coalesced=True
        )

        ctx.save_for_backward(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_entry.long(), ori_rel, ori_m,
            inv_b, inv_l, inv_e, inv_entry.long(), inv_rel, inv_m)
        ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i = B, L, E, r_size, wot_i
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m,
         inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m) = ctx.saved_tensors
        module = get_module()
        grad_out = grad_out.coalesce()
        gidx = grad_out.indices()
        grad_sp, grad_w = module.flashlog_backward_sparse3d_sum_cpu(
            gidx[0], gidx[1], gidx[2], grad_out.values().to(dtype=sp_value.dtype),
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m,
            inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m,
            ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i
        )

        return (None, None, None, grad_sp, grad_w,
                None, None, None, None, None, None,
                None, None, None, None, None, None,
                None, None, None, None, None, None)


class FlashLogFunctionSparse3DCPUSumNorm(autograd.Function):
    """CPU sparse 3D sum+normalize with C++ forward and backward."""
    @staticmethod
    def forward(ctx, sp_batch, sp_level, sp_entity, sp_value, w,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                B, L, E, r_size, wot_i, topk_edges, eps):
        module = get_module()
        save_backward = bool(ctx.needs_input_grad[3] or ctx.needs_input_grad[4])
        (out_b, out_l, out_e, out_v, out_keys, out_values, denom,
         ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m, ori_pos,
         inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m, inv_pos, ind_pos) = module.flashlog_forward_sparse3d_sum_norm_cpu(
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_col_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_col_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            B, L, E, r_size, wot_i, topk_edges, float(eps), save_backward
        )
        out = torch.sparse_coo_tensor(
            torch.stack([out_b, out_l, out_e], dim=0),
            out_v.to(dtype=sp_value.dtype),
            [B, L, E],
            is_coalesced=True
        )
        ctx.has_backward = save_backward
        if save_backward:
            ctx.save_for_backward(
                out_keys, out_values, denom,
                sp_batch, sp_level, sp_entity, sp_value, w,
                ori_b, ori_l, ori_e, ori_entry, ori_rel,
                ori_m.to(dtype=sp_value.dtype), ori_pos,
                inv_b, inv_l, inv_e, inv_entry, inv_rel,
                inv_m.to(dtype=sp_value.dtype), inv_pos, ind_pos
            )
        ctx.B, ctx.L, ctx.E, ctx.r_size = B, L, E, r_size
        ctx.wot_i = wot_i
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not getattr(ctx, "has_backward", True):
            raise RuntimeError("CPU fused sparse sum+norm backward metadata was not saved")
        (out_keys, out_values, denom,
         sp_batch, sp_level, sp_entity, sp_value, w,
         ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m, ori_pos,
         inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m, inv_pos, ind_pos) = ctx.saved_tensors
        module = get_module()
        grad_out = grad_out.coalesce()
        gidx = grad_out.indices()
        grad_sp, grad_w = module.flashlog_backward_sparse3d_sum_norm_cpu(
            gidx[0], gidx[1], gidx[2], grad_out.values().to(dtype=sp_value.dtype),
            out_keys, out_values, denom,
            sp_batch, sp_level, sp_entity, sp_value, w,
            ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m, ori_pos,
            inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m, inv_pos, ind_pos,
            ctx.B, ctx.L, ctx.E, ctx.r_size, ctx.wot_i, ctx.eps
        )
        return (None, None, None, grad_sp, grad_w,
                None, None, None, None, None, None,
                None, None, None, None, None, None,
                None, None, None, None, None, None, None)


def vectorized_operation_sparse3d_csr_sum(sp_state, w, csr_struct, mask_values,
                                          r_size, weight=None, wot_i=False,
                                          use_topk=False, topk_edges=0,
                                          prepared_masks=None, topk_nodes=None):
    sp_state = sp_state.coalesce()
    idx = sp_state.indices()
    val = sp_state.values()
    B, L, E = sp_state.shape

    if use_topk and topk_nodes is not None and not sp_state.is_cuda:
        entries = _sparse_topc_entries(idx[0], idx[1], idx[2], val, topk_nodes, L)
        if entries is not None:
            return _sparse_sum_selected_nodes(
                sp_state, entries, w, csr_struct, mask_values, r_size,
                weight, wot_i, topk_edges, prepared_masks)

    ori_col_ind = csr_struct[1]
    order_ori = csr_struct[3]
    inv_col_ind = csr_struct[5]
    order_inv = csr_struct[7]
    ori_row_group_ptr = csr_struct[8]
    ori_group_rel = csr_struct[9]
    ori_group_edge_start = csr_struct[10]
    ori_group_edge_count = csr_struct[11]
    inv_row_group_ptr = csr_struct[12]
    inv_group_rel = csr_struct[13]
    inv_group_edge_start = csr_struct[14]
    inv_group_edge_count = csr_struct[15]
    effective_topk = topk_edges if (use_topk and topk_edges > 0) else (1 << 60)

    if sp_state.is_cuda:
        can_use_all_edges = not (use_topk and (topk_edges > 0 or topk_nodes is not None))
        can_use_all_edges = can_use_all_edges and not (
            weight is not None and bool(getattr(weight, "requires_grad", False))
        )
        if can_use_all_edges:
            prepared = _prepared_mask_pair(prepared_masks) if prepared_masks is not None else _prepared_mask_pair(mask_values)
            if prepared is not None:
                ori_mask, inv_mask = prepared
            else:
                ori_mask, inv_mask = prepare_csr_masks(csr_struct, mask_values, weight)
            return FlashLogFunctionSparse3DAllEdgesSum.apply(
                idx[0], idx[1], idx[2], val, w,
                ori_col_ind, ori_mask,
                ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
                inv_col_ind, inv_mask,
                inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
                B, L, E, r_size, wot_i, torch.is_grad_enabled()
            )

        return FlashLogFunctionSparse3DTopKSum.apply(
            idx[0], idx[1], idx[2], val, w,
            csr_struct[0], ori_col_ind, csr_struct[2], order_ori,
            csr_struct[4], inv_col_ind, csr_struct[6], order_inv,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
            mask_values, weight, B, L, E, r_size, wot_i, effective_topk,
            False, topk_nodes if use_topk else None, torch.is_grad_enabled()
        )

    prepared = _prepared_mask_pair(prepared_masks) if prepared_masks is not None else None
    if prepared is not None:
        ori_mask, inv_mask = prepared
    else:
        ori_mask, inv_mask = prepare_csr_masks(csr_struct, mask_values, weight)
    return FlashLogFunctionSparse3DCPUSum.apply(
        idx[0], idx[1], idx[2], val, w,
        ori_col_ind, ori_mask,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_col_ind, inv_mask,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        B, L, E, r_size, wot_i, effective_topk)


def _sparse_sum_selected_nodes(sp_state, entries, w, csr_struct, mask_values,
                               r_size, weight, wot_i, topk_edges, prepared_masks):
    idx, val = sp_state.indices(), sp_state.values()
    selected = torch.sparse_coo_tensor(
        idx[:, entries], val[entries], sp_state.shape, is_coalesced=True)
    return vectorized_operation_sparse3d_csr_sum(
        selected, w, csr_struct, mask_values, r_size, weight=weight,
        wot_i=wot_i, use_topk=True, topk_edges=topk_edges, prepared_masks=prepared_masks)


def vectorized_operation_sparse3d_csr_sum_norm(sp_state, w, csr_struct, mask_values,
                                               r_size, weight=None, wot_i=False,
                                               use_topk=False, topk_edges=0,
                                               prepared_masks=None, eps=1e-7, topk_nodes=None):
    if sp_state.is_cuda:
        return normalize_sparse3d(
            vectorized_operation_sparse3d_csr_sum(
                sp_state, w, csr_struct, mask_values,
                r_size, weight=weight, wot_i=wot_i,
                use_topk=use_topk, topk_edges=topk_edges,
                prepared_masks=prepared_masks, topk_nodes=topk_nodes),
            eps=eps
        )

    sp_state = sp_state.coalesce()
    idx = sp_state.indices()
    val = sp_state.values()
    B, L, E = sp_state.shape

    if use_topk and topk_nodes is not None:
        entries = _sparse_topc_entries(idx[0], idx[1], idx[2], val, topk_nodes, L)
        if entries is not None:
            return normalize_sparse3d(_sparse_sum_selected_nodes(
                sp_state, entries, w, csr_struct, mask_values, r_size,
                weight, wot_i, topk_edges, prepared_masks), eps=eps)

    ori_col_ind = csr_struct[1]
    inv_col_ind = csr_struct[5]
    ori_row_group_ptr = csr_struct[8]
    ori_group_rel = csr_struct[9]
    ori_group_edge_start = csr_struct[10]
    ori_group_edge_count = csr_struct[11]
    inv_row_group_ptr = csr_struct[12]
    inv_group_rel = csr_struct[13]
    inv_group_edge_start = csr_struct[14]
    inv_group_edge_count = csr_struct[15]
    effective_topk = topk_edges if (use_topk and topk_edges > 0) else (1 << 60)

    prepared = _prepared_mask_pair(prepared_masks) if prepared_masks is not None else None
    if prepared is not None:
        ori_mask, inv_mask = prepared
    else:
        ori_mask, inv_mask = prepare_csr_masks(csr_struct, mask_values, weight)

    return FlashLogFunctionSparse3DCPUSumNorm.apply(
        idx[0], idx[1], idx[2], val, w,
        ori_col_ind, ori_mask,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        inv_col_ind, inv_mask,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        B, L, E, r_size, wot_i, effective_topk, eps)
