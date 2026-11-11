#include "flashlog.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <cstdint>
#include <type_traits>

namespace flashlog {

constexpr int AGG_MAX = 1;
// Exact nonnegative floating-point keys; no quantization of priorities.
constexpr int RADIX_SIZE = 1 << 8;
template <typename scalar_t>
__device__ __forceinline__ int64_t exact_weight_key(scalar_t value) {
    if constexpr (sizeof(scalar_t) == sizeof(float))
        return static_cast<int64_t>(__float_as_uint(static_cast<float>(value)));
    else
        return __double_as_longlong(static_cast<double>(value));
}

// Weighted order statistic over relation groups, with group sizes as multiplicities.
// Four passes for float32 / eight for float64; all block threads participate.
template <typename scalar_t>
__device__ void exact_group_threshold(
    const scalar_t* weights, const int16_t* relations, const int32_t* sizes,
    int64_t begin, int64_t end, int64_t budget, int* digits,
    int64_t* threshold, int64_t* remaining, int64_t* total
) {
    __shared__ int64_t prefix;
    if (threadIdx.x == 0) { *threshold = -1; *remaining = budget; *total = 0; prefix = 0; }
    __syncthreads();
    int64_t local = 0;
    for (int64_t g = begin + threadIdx.x; g < end; g += blockDim.x)
        if (weights[relations[g]] > scalar_t(0)) local += sizes[g];
    atomicAdd(reinterpret_cast<unsigned long long*>(total), static_cast<unsigned long long>(local));
    __syncthreads();
    if (*total <= budget) return;
    constexpr int first_shift = (sizeof(scalar_t) - 1) * 8;
    for (int shift = first_shift; shift >= 0; shift -= 8) {
        for (int d = threadIdx.x; d < RADIX_SIZE; d += blockDim.x) digits[d] = 0;
        __syncthreads();
        uint64_t prefix_mask = shift == first_shift ? 0 : (~uint64_t(0) << (shift + 8));
        for (int64_t g = begin + threadIdx.x; g < end; g += blockDim.x) {
            scalar_t score = weights[relations[g]];
            uint64_t key = static_cast<uint64_t>(exact_weight_key(score));
            if (score > scalar_t(0) && (key & prefix_mask) == static_cast<uint64_t>(prefix))
                atomicAdd(&digits[(key >> shift) & (RADIX_SIZE - 1)], sizes[g]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int d = RADIX_SIZE - 1; d >= 0; --d) {
                if (*remaining <= digits[d]) { prefix |= static_cast<int64_t>(d) << shift; break; }
                *remaining -= digits[d];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *threshold = prefix;
    __syncthreads();
}


__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* address_as_i = reinterpret_cast<int*>(addr);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ double atomicMaxDouble(double* addr, double value) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(static_cast<long long>(assumed)) >= value) break;
        old = atomicCAS(address_as_ull, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(value)));
    } while (assumed != old);
    return __longlong_as_double(static_cast<long long>(old));
}

template <typename scalar_t>
__device__ __forceinline__ void atomic_reduce(scalar_t* addr, scalar_t value, int64_t agg_mode) {
    if (agg_mode == AGG_MAX) {
        if constexpr (std::is_same<scalar_t, float>::value) {
            atomicMaxFloat(addr, value);
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            atomicMaxDouble(addr, value);
        } else {
            atomicAdd(addr, value);
        }
    } else {
        atomicAdd(addr, value);
    }
}

__device__ __forceinline__ unsigned long long pack_max_pair(float value, int edge_idx) {
    return (static_cast<unsigned long long>(__float_as_uint(value)) << 32) |
           static_cast<unsigned int>(0xFFFFFFFFu - static_cast<unsigned int>(edge_idx));
}

__device__ __forceinline__ int unpack_max_edge(unsigned long long packed) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<unsigned int>(packed & 0xFFFFFFFFu));
}

__device__ __forceinline__ void atomicMaxPair(unsigned long long* packed_ptr, float value, int edge_idx) {
    unsigned long long old = *packed_ptr, assumed;
    unsigned long long candidate = pack_max_pair(value, edge_idx);
    while (candidate > old) {
        assumed = old;
        old = atomicCAS(packed_ptr, assumed, candidate);
        if (old == assumed) break;
    }
}

__global__ void unpack_max_pairs_kernel(
    const unsigned long long* __restrict__ packed,
    float* __restrict__ out,
    int64_t* __restrict__ arg,
    int64_t total
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long v = packed[idx];
    out[idx] = __uint_as_float(static_cast<unsigned int>(v >> 32));
    int edge_idx = unpack_max_edge(v);
    arg[idx] = (v == 0ULL) ? -1 : static_cast<int64_t>(edge_idx);
}

template <typename scalar_t>
__global__ void fill_sparse3d_identity_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t nnz,
    int64_t L,
    int64_t n
);

// Identity contribution for every entity.
template <typename scalar_t>
__global__ void flashlog_identity_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    int64_t BL,
    int64_t E,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BL * E;
    if (idx >= total) return;

    int64_t bl = idx / E;
    out[idx] = A[idx] * w[bl * n + (n - 1)];
}

// One block per (batch, rule): coalesced entity traversal and fixed-tree reduction.
// All callers launch 256 threads. Identity remains independent of graph pruning.
template <typename scalar_t>
__global__ void flashlog_identity_backward_kernel(
    const scalar_t* __restrict__ grad_ind,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t BL,
    int64_t E,
    int64_t n
) {
    int64_t bl = blockIdx.x;
    if (bl >= BL) return;

    scalar_t wid = w[bl * n + (n - 1)];
    scalar_t gw_acc = 0;

    for (int64_t e = threadIdx.x; e < E; e += blockDim.x) {
        scalar_t gi = grad_ind[bl * E + e];
        gA[bl * E + e] = gi * wid;
        gw_acc += gi * A[bl * E + e];
    }
    __shared__ scalar_t partial[256];
    partial[threadIdx.x] = gw_acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gw[bl * n + (n - 1)] = partial[0];
    }
}

// Every thread uses the same quota; only thread 0 advances the threshold count.
template <typename scalar_t>
__device__ __forceinline__ int32_t dense_group_take(
    int32_t count, scalar_t weight, int64_t threshold_key,
    int64_t keep_in_threshold, int64_t* threshold_seen
) {
    __shared__ int32_t selected_take;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t take = weight > scalar_t(0) ? count : 0;
        if (take > 0 && threshold_key >= 0) {
            int64_t key = exact_weight_key(weight);
            if (key < threshold_key) {
                take = 0;
            } else if (key == threshold_key) {
                int64_t left = keep_in_threshold - *threshold_seen;
                take = left > 0 ? static_cast<int32_t>(left < count ? left : count) : 0;
                *threshold_seen += take;
            }
        }
        selected_take = take;
    }
    __syncthreads();
    return selected_take;
}

constexpr int TOPK_THREADS_PER_BLOCK = 128;

template <typename scalar_t, bool IsInv>
__device__ void forward_topk_row(
    int64_t src,
    int64_t bl,
    scalar_t a_val,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const int16_t* __restrict__ r_ind,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ out,
    int radix_counts[RADIX_SIZE],
    int64_t* threshold_key_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    __shared__ scalar_t group_product;
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    exact_group_threshold(w + bl * n + rel_offset, group_rel, group_edge_count,
        group_start, group_end, topk_edges, radix_counts, threshold_key_shared,
        keep_in_threshold_shared, active_edges_shared);
    if (threadIdx.x == 0) *threshold_seen_shared = 0;
    __syncthreads();
    int64_t threshold_key = *threshold_key_shared;
    int64_t keep_in_threshold = *keep_in_threshold_shared;
    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = dense_group_take(
            group_edge_count[g], w_val, threshold_key,
            keep_in_threshold, threshold_seen_shared);
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        // One source/attention product per relation group, shared by its edges.
        if (threadIdx.x == 0) group_product = a_val * w_val;
        __syncthreads();
        int64_t edge_start = group_edge_start[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_start + take; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            atomic_reduce(&out[bl * E + dst], group_product * m, agg_mode);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__device__ void forward_notopk_row(
    int64_t src,
    int64_t bl,
    scalar_t a_val,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ col_ind,
    const scalar_t* __restrict__ mask,
    scalar_t* __restrict__ out,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    __shared__ scalar_t group_product;
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        // One source/attention product per relation group, shared by its edges.
        if (threadIdx.x == 0) group_product = a_val * w_val;
        __syncthreads();
        int64_t edge_start = group_edge_start[g];
        int64_t edge_end = edge_start + group_edge_count[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_end; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            atomic_reduce(&out[bl * E + dst], group_product * m, agg_mode);
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void flashlog_forward_notopk_fused_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ out_ori,
    scalar_t* __restrict__ out_inv,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    scalar_t a_val = A[bl * E + src];
    if (a_val == 0) return;

    forward_notopk_row<scalar_t>(
        src, bl, a_val, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_col_ind, ori_mask, out_ori,
        E, n, 0, agg_mode
    );
    forward_notopk_row<scalar_t>(
        src, bl, a_val, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_col_ind, inv_mask, out_inv,
        E, n, r_size, agg_mode
    );
}

template <typename scalar_t>
__global__ void flashlog_forward_topk_fused_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ out_ori,
    scalar_t* __restrict__ out_inv,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    scalar_t a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int radix_counts[RADIX_SIZE];
    __shared__ int64_t threshold_key_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    forward_topk_row<scalar_t, false>(
        src, bl, a_val, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask,
        out_ori, radix_counts,
        &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0, agg_mode
    );
    __syncthreads();
    forward_topk_row<scalar_t, true>(
        src, bl, a_val, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask,
        out_inv, radix_counts,
        &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size, agg_mode
    );
}

template <typename scalar_t>
__device__ void backward_topk_row(
    int64_t src,
    int64_t bl,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const int16_t* __restrict__ r_ind,
    const scalar_t* __restrict__ mask,
    const scalar_t* __restrict__ out,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int radix_counts[RADIX_SIZE],
    int64_t* threshold_key_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    exact_group_threshold(w + bl * n + rel_offset, group_rel, group_edge_count,
        group_start, group_end, topk_edges, radix_counts, threshold_key_shared,
        keep_in_threshold_shared, active_edges_shared);
    if (threadIdx.x == 0) *threshold_seen_shared = 0;
    __syncthreads();
    int64_t threshold_key = *threshold_key_shared;
    int64_t keep_in_threshold = *keep_in_threshold_shared;
    scalar_t a_val = A[bl * E + src];
    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = dense_group_take(
            group_edge_count[g], w_val, threshold_key,
            keep_in_threshold, threshold_seen_shared);
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        int64_t edge_start = group_edge_start[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_start + take; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            scalar_t go = grad_out[bl * E + dst];
            if (go == 0) continue;
            scalar_t contrib = a_val * w_val * m;
            if (agg_mode == AGG_MAX) {
                scalar_t out_val = out[bl * E + dst];
                scalar_t diff = contrib - out_val;
                if (diff < static_cast<scalar_t>(-1e-6) || diff > static_cast<scalar_t>(1e-6)) continue;
            }
            atomicAdd(&gA[bl * E + src], go * w_val * m);
            atomicAdd(&gw[bl * n + rel_offset + rel], go * a_val * m);
        }
        __syncthreads();
    }
}

__device__ void forward_max_topk_row(
    int64_t src,
    int64_t bl,
    float a_val,
    const float* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ col_ind,
    const float* __restrict__ mask,
    unsigned long long* __restrict__ packed_out,
    int radix_counts[RADIX_SIZE],
    int64_t* threshold_key_shared,
    int64_t* keep_in_threshold_shared,
    int64_t* threshold_seen_shared,
    int64_t* active_edges_shared,
    int64_t topk_edges,
    int64_t E,
    int64_t n,
    int64_t rel_offset
) {
    __shared__ float group_product;
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    exact_group_threshold(w + bl * n + rel_offset, group_rel, group_edge_count,
        group_start, group_end, topk_edges, radix_counts, threshold_key_shared,
        keep_in_threshold_shared, active_edges_shared);
    if (threadIdx.x == 0) *threshold_seen_shared = 0;
    __syncthreads();
    int64_t threshold_key = *threshold_key_shared;
    int64_t keep_in_threshold = *keep_in_threshold_shared;
    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        float w_val = w[bl * n + rel_offset + rel];
        int32_t take = dense_group_take(
            group_edge_count[g], w_val, threshold_key,
            keep_in_threshold, threshold_seen_shared);
        if (take <= 0) {
            __syncthreads();
            continue;
        }

        // One source/attention product per relation group, shared by its edges.
        if (threadIdx.x == 0) group_product = a_val * w_val;
        __syncthreads();
        int64_t edge_start = group_edge_start[g];
        for (int64_t e = threadIdx.x; e < take; e += blockDim.x) {
            int64_t p = edge_start + e;
            float m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            float val = group_product * m;
            atomicMaxPair(&packed_out[bl * E + dst], val, static_cast<int>(p));
        }
        __syncthreads();
    }
}

__global__ void flashlog_forward_max_topk_fused_kernel(
    const float* __restrict__ A,
    const float* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const float* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const float* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    unsigned long long* __restrict__ out_ori_packed,
    unsigned long long* __restrict__ out_inv_packed,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;
    int64_t src = active_nodes[node_idx];
    float a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int radix_counts[RADIX_SIZE];
    __shared__ int64_t threshold_key_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    forward_max_topk_row(
        src, bl, a_val, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_col_ind, ori_mask, out_ori_packed,
        radix_counts, &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0
    );
    __syncthreads();
    forward_max_topk_row(
        src, bl, a_val, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_col_ind, inv_mask, out_inv_packed,
        radix_counts, &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size
    );
}

__global__ void flashlog_backward_max_topk_fused_kernel(
    const float* __restrict__ grad_ori,
    const float* __restrict__ grad_inv,
    const int64_t* __restrict__ ori_arg,
    const int64_t* __restrict__ inv_arg,
    const float* __restrict__ A,
    const float* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const float* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const float* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    float* __restrict__ gA,
    float* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;
    int64_t src = active_nodes[node_idx];
    float a_val = A[bl * E + src];
    if (a_val == 0) return;

    __shared__ int radix_counts[RADIX_SIZE];
    __shared__ int64_t threshold_key_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;

    auto process = [&](const float* grad_out, const int64_t* arg_out,
                       const int32_t* col_ind, const int16_t* r_ind, const float* mask,
                       const int32_t* row_group_ptr, const int16_t* group_rel,
                       const int32_t* group_edge_start, const int32_t* group_edge_count,
                       int64_t rel_offset) {
        int64_t group_start = row_group_ptr[src];
        int64_t group_end = row_group_ptr[src + 1];
        exact_group_threshold(w + bl * n + rel_offset, group_rel, group_edge_count,
            group_start, group_end, topk_edges, radix_counts, &threshold_key_shared,
            &keep_in_threshold_shared, &active_edges_shared);
        if (threadIdx.x == 0) threshold_seen_shared = 0;
        __syncthreads();
        for (int64_t g = group_start; g < group_end; ++g) {
            int64_t rel = group_rel[g];
            float w_val = w[bl * n + rel_offset + rel];
            int32_t take = dense_group_take(
                group_edge_count[g], w_val, threshold_key_shared,
                keep_in_threshold_shared, &threshold_seen_shared);
            if (take <= 0) {
                __syncthreads();
                continue;
            }
            int64_t edge_start = group_edge_start[g];
            for (int64_t e = threadIdx.x; e < take; e += blockDim.x) {
                int64_t p = edge_start + e;
                float m = mask[p];
                if (m == 0) continue;
                int64_t dst = col_ind[p];
                if (arg_out[bl * E + dst] != p) continue;
                float go = grad_out[bl * E + dst];
                if (go == 0) continue;
                atomicAdd(&gA[bl * E + src], go * w_val * m);
                atomicAdd(&gw[bl * n + rel_offset + rel], go * a_val * m);
            }
            __syncthreads();
        }
    };

    process(grad_ori, ori_arg, ori_col_ind, ori_r_ind, ori_mask,
            ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count, 0);
    __syncthreads();
    process(grad_inv, inv_arg, inv_col_ind, inv_r_ind, inv_mask,
            inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count, r_size);
}

template <typename scalar_t>
__device__ void backward_notopk_row(
    int64_t src,
    int64_t bl,
    scalar_t a_val,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int32_t* __restrict__ col_ind,
    const scalar_t* __restrict__ mask,
    const scalar_t* __restrict__ out,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t agg_mode
) {
    int64_t group_start = row_group_ptr[src];
    int64_t group_end = row_group_ptr[src + 1];

    for (int64_t g = group_start; g < group_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int64_t edge_start = group_edge_start[g];
        int64_t edge_end = edge_start + group_edge_count[g];
        for (int64_t p = edge_start + threadIdx.x; p < edge_end; p += blockDim.x) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            scalar_t go = grad_out[bl * E + dst];
            if (go == 0) continue;
            scalar_t contrib = a_val * w_val * m;
            if (agg_mode == AGG_MAX) {
                scalar_t out_val = out[bl * E + dst];
                scalar_t diff = contrib - out_val;
                if (diff < static_cast<scalar_t>(-1e-6) || diff > static_cast<scalar_t>(1e-6)) continue;
            }
            atomicAdd(&gA[bl * E + src], go * w_val * m);
            atomicAdd(&gw[bl * n + rel_offset + rel], go * a_val * m);
        }
    }
}

template <typename scalar_t>
__global__ void flashlog_backward_topk_fused_kernel(
    const scalar_t* __restrict__ grad_ori,
    const scalar_t* __restrict__ grad_inv,
    const scalar_t* __restrict__ out_ori,
    const scalar_t* __restrict__ out_inv,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_row_ptr,
    const int32_t* __restrict__ ori_col_ind,
    const int16_t* __restrict__ ori_r_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_row_ptr,
    const int32_t* __restrict__ inv_col_ind,
    const int16_t* __restrict__ inv_r_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t topk_edges,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    if (A[bl * E + src] == 0) return;

    __shared__ int radix_counts[RADIX_SIZE];
    __shared__ int64_t threshold_key_shared;
    __shared__ int64_t keep_in_threshold_shared;
    __shared__ int64_t threshold_seen_shared;
    __shared__ int64_t active_edges_shared;
    backward_topk_row(
        src, bl, grad_ori, A, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_row_ptr, ori_col_ind, ori_r_ind, ori_mask, out_ori,
        gA, gw, radix_counts,
        &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, 0, agg_mode
    );
    __syncthreads();
    backward_topk_row(
        src, bl, grad_inv, A, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_row_ptr, inv_col_ind, inv_r_ind, inv_mask, out_inv,
        gA, gw, radix_counts,
        &threshold_key_shared, &keep_in_threshold_shared,
        &threshold_seen_shared, &active_edges_shared,
        topk_edges, E, n, r_size, agg_mode
    );
}

template <typename scalar_t>
__global__ void flashlog_backward_notopk_fused_kernel(
    const scalar_t* __restrict__ grad_ori,
    const scalar_t* __restrict__ grad_inv,
    const scalar_t* __restrict__ out_ori,
    const scalar_t* __restrict__ out_inv,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ active_nodes,
    const int32_t* __restrict__ ori_col_ind,
    const scalar_t* __restrict__ ori_mask,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_col_ind,
    const scalar_t* __restrict__ inv_mask,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int32_t* __restrict__ inv_group_edge_count,
    scalar_t* __restrict__ gA,
    scalar_t* __restrict__ gw,
    int64_t num_active,
    int64_t BL,
    int64_t E,
    int64_t n,
    int64_t r_size,
    int64_t agg_mode
) {
    int64_t node_idx = blockIdx.x;
    int64_t bl = blockIdx.y;
    if (node_idx >= num_active || bl >= BL) return;

    int64_t src = active_nodes[node_idx];
    scalar_t a_val = A[bl * E + src];
    if (a_val == 0) return;

    backward_notopk_row<scalar_t>(
        src, bl, a_val, grad_ori, w,
        ori_row_group_ptr, ori_group_rel, ori_group_edge_start, ori_group_edge_count,
        ori_col_ind, ori_mask, out_ori,
        gA, gw, E, n, 0, agg_mode
    );
    backward_notopk_row<scalar_t>(
        src, bl, a_val, grad_inv, w,
        inv_row_group_ptr, inv_group_rel, inv_group_edge_start, inv_group_edge_count,
        inv_col_ind, inv_mask, out_inv,
        gA, gw, E, n, r_size, agg_mode
    );
}

// =========================================================================
// Forward API — 2 launches instead of 3 (identity + fused ORI/INV)
// =========================================================================
std::tuple<Tensor, Tensor, Tensor> flashlog_forward_cuda(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
) {
    TORCH_CHECK(A_.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(A_.dim() == 2, "A must be 2D [BL, E]");

    const Tensor A = A_.contiguous();
    const Tensor w = w_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor orp = ori_row_ptr_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.to(A.dtype()).contiguous();
    const Tensor irp = inv_row_ptr_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);

    Tensor out_ind = at::zeros({BL, E}, A.options());
    Tensor out_ori = at::zeros({BL, E}, A.options());
    Tensor out_inv = at::zeros({BL, E}, A.options());

    // Launch 1: Identity (unchanged)
    if (!wot_i) {
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_identity_cuda", [&] {
            flashlog_identity_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                out_ind.data_ptr<scalar_t>(), BL, E, n
            );
        });
    }

    // Launch 2: Group-based ORI + INV path. No-topk skips histogram/threshold
    // bookkeeping and directly expands all relation groups.
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        bool do_topk = use_topk && topk_edges > 0 && topk_edges < (static_cast<int64_t>(1) << 60);
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_forward_group_fused_cuda", [&] {
            if (do_topk) {
                flashlog_forward_topk_fused_kernel<scalar_t><<<grid, block>>>(
                    A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                    act.data_ptr<int64_t>(),
                    orp.data_ptr<int32_t>(), oci.data_ptr<int32_t>(),
                    ori.data_ptr<int16_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    irp.data_ptr<int32_t>(), ici.data_ptr<int32_t>(),
                    iri.data_ptr<int16_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    out_ori.data_ptr<scalar_t>(),
                    out_inv.data_ptr<scalar_t>(),
                    num_active, BL, E, n, r_size, topk_edges, agg_mode
                );
            } else {
                flashlog_forward_notopk_fused_kernel<scalar_t><<<grid, block>>>(
                    A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                    act.data_ptr<int64_t>(),
                    oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    out_ori.data_ptr<scalar_t>(),
                    out_inv.data_ptr<scalar_t>(),
                    num_active, BL, E, n, r_size, agg_mode
                );
            }
        });
    }

    return std::make_tuple(out_ind, out_ori, out_inv);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> flashlog_forward_max_cuda(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor act = active_nodes_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);
    Tensor out_ind = at::zeros({BL, E}, A.options());
    Tensor out_ori = at::zeros({BL, E}, A.options());
    Tensor out_inv = at::zeros({BL, E}, A.options());
    Tensor ori_arg = at::full({BL, E}, -1, act.options());
    Tensor inv_arg = at::full({BL, E}, -1, act.options());
    Tensor out_ori_packed = at::zeros({BL, E}, act.options().dtype(at::kLong));
    Tensor out_inv_packed = at::zeros({BL, E}, act.options().dtype(at::kLong));

    if (!wot_i) {
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        flashlog_identity_kernel<float><<<blocks, threads>>>(
            A.data_ptr<float>(), w.data_ptr<float>(),
            out_ind.data_ptr<float>(), BL, E, n
        );
    }
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        flashlog_forward_max_topk_fused_kernel<<<grid, block>>>(
            A.data_ptr<float>(), w.data_ptr<float>(), act.data_ptr<int64_t>(),
            oci.data_ptr<int32_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            reinterpret_cast<unsigned long long*>(out_ori_packed.data_ptr<int64_t>()),
            reinterpret_cast<unsigned long long*>(out_inv_packed.data_ptr<int64_t>()),
            num_active, BL, E, n, r_size, effective_topk
        );
        int threads = 256;
        int blocks = (BL * E + threads - 1) / threads;
        unpack_max_pairs_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned long long*>(out_ori_packed.data_ptr<int64_t>()),
            out_ori.data_ptr<float>(), ori_arg.data_ptr<int64_t>(), BL * E
        );
        unpack_max_pairs_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned long long*>(out_inv_packed.data_ptr<int64_t>()),
            out_inv.data_ptr<float>(), inv_arg.data_ptr<int64_t>(), BL * E
        );
    }
    return std::make_tuple(out_ind, out_ori, out_inv, ori_arg, inv_arg);
}

// =========================================================================
// Backward API — 2 launches instead of 3 (identity_bwd + fused ORI/INV bwd)
// =========================================================================
std::tuple<Tensor, Tensor> flashlog_backward_cuda(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &out_ori_, const Tensor &out_inv_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges, int64_t agg_mode
) {
    const Tensor A = A_.contiguous();
    const Tensor w = w_.contiguous();
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous();
    const Tensor grad_ori = grad_ori_.contiguous();
    const Tensor grad_inv = grad_inv_.contiguous();
    const Tensor out_ori = out_ori_.contiguous();
    const Tensor out_inv = out_inv_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor orp = ori_row_ptr_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.to(A.dtype()).contiguous();
    const Tensor irp = inv_row_ptr_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);

    Tensor gA = at::zeros({BL, E}, A.options());
    Tensor gw = at::zeros({BL, n}, w.options());

    // Launch 1: Identity backward, one block per (batch, rule)
    if (!wot_i) {
        int threads = 256;
        int blocks = BL;
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_identity_backward_cuda", [&] {
            flashlog_identity_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad_ind.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                BL, E, n
            );
        });
    }

    // Launch 2: Group-based ORI + INV backward. No-topk mirrors the forward
    // fast path and skips histogram/threshold bookkeeping.
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        bool do_topk = use_topk && topk_edges > 0 && topk_edges < (static_cast<int64_t>(1) << 60);
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_backward_group_fused_cuda", [&] {
            if (do_topk) {
                flashlog_backward_topk_fused_kernel<scalar_t><<<grid, block>>>(
                    grad_ori.data_ptr<scalar_t>(),
                    grad_inv.data_ptr<scalar_t>(),
                    out_ori.data_ptr<scalar_t>(),
                    out_inv.data_ptr<scalar_t>(),
                    A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                    act.data_ptr<int64_t>(),
                    orp.data_ptr<int32_t>(), oci.data_ptr<int32_t>(),
                    ori.data_ptr<int16_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    irp.data_ptr<int32_t>(), ici.data_ptr<int32_t>(),
                    iri.data_ptr<int16_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                    num_active, BL, E, n, r_size, topk_edges, agg_mode
                );
            } else {
                flashlog_backward_notopk_fused_kernel<scalar_t><<<grid, block>>>(
                    grad_ori.data_ptr<scalar_t>(),
                    grad_inv.data_ptr<scalar_t>(),
                    out_ori.data_ptr<scalar_t>(),
                    out_inv.data_ptr<scalar_t>(),
                    A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                    act.data_ptr<int64_t>(),
                    oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
                    num_active, BL, E, n, r_size, agg_mode
                );
            }
        });
    }

    return std::make_tuple(gA, gw);
}

std::tuple<Tensor, Tensor> flashlog_backward_max_cuda(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &ori_arg_, const Tensor &inv_arg_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_row_ptr_, const Tensor &ori_col_ind_,
    const Tensor &ori_r_ind_, const Tensor &ori_mask_,
    const Tensor &inv_row_ptr_, const Tensor &inv_col_ind_,
    const Tensor &inv_r_ind_, const Tensor &inv_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, bool use_topk, int64_t topk_edges
) {
    const Tensor A = A_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous().to(at::kFloat);
    const Tensor grad_ori = grad_ori_.contiguous().to(at::kFloat);
    const Tensor grad_inv = grad_inv_.contiguous().to(at::kFloat);
    const Tensor ori_arg = ori_arg_.contiguous();
    const Tensor inv_arg = inv_arg_.contiguous();
    const Tensor act = active_nodes_.contiguous();
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor ori = ori_r_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor iri = inv_r_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    int64_t num_active = act.size(0);
    Tensor gA = at::zeros({BL, E}, A.options());
    Tensor gw = at::zeros({BL, n}, w.options());

    if (!wot_i) {
        int threads = 256;
        int blocks = BL;
        flashlog_identity_backward_kernel<float><<<blocks, threads>>>(
            grad_ind.data_ptr<float>(), A.data_ptr<float>(), w.data_ptr<float>(),
            gA.data_ptr<float>(), gw.data_ptr<float>(), BL, E, n
        );
    }
    if (num_active > 0) {
        dim3 block(TOPK_THREADS_PER_BLOCK);
        dim3 grid(num_active, BL);
        int64_t effective_topk = (use_topk && topk_edges > 0) ? topk_edges : (static_cast<int64_t>(1) << 60);
        flashlog_backward_max_topk_fused_kernel<<<grid, block>>>(
            grad_ori.data_ptr<float>(), grad_inv.data_ptr<float>(),
            ori_arg.data_ptr<int64_t>(), inv_arg.data_ptr<int64_t>(),
            A.data_ptr<float>(), w.data_ptr<float>(), act.data_ptr<int64_t>(),
            oci.data_ptr<int32_t>(), ori.data_ptr<int16_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), iri.data_ptr<int16_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            gA.data_ptr<float>(), gw.data_ptr<float>(),
            num_active, BL, E, n, r_size, effective_topk
        );
    }
    return std::make_tuple(gA, gw);
}

__global__ void mark_active_kernel(
    const float* __restrict__ A,
    int32_t* __restrict__ flags,
    int64_t BL, int64_t E
) {
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    for (int64_t bl = 0; bl < BL; bl++) {
        if (A[bl * E + e] != 0.0f) {
            flags[e] = 1;
            return;
        }
    }
}

Tensor compute_active_nodes_cuda(const Tensor &A_flat) {
    int64_t BL = A_flat.size(0), E = A_flat.size(1);
    auto flags = at::zeros({E}, A_flat.options().dtype(at::kInt));

    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    mark_active_kernel<<<blocks, threads>>>(
        A_flat.data_ptr<float>(), flags.data_ptr<int32_t>(), BL, E
    );

    return flags.nonzero().reshape(-1).to(at::kLong);
}

// =========================================================================
// Apply mask on GPU — fuses indexing + score_function + multiply
// mask_out[i] = mask_values[order[i]] * score_func(weight[order[i]])
// score_func(x) = sigmoid(x)
// =========================================================================
template <typename mask_t>
__global__ void apply_mask_kernel(
    const mask_t* __restrict__ mask_values,
    const int32_t* __restrict__ order,
    const float* __restrict__ weight,  // nullptr if no weight
    float* __restrict__ out,
    int64_t nnz,
    bool has_weight
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int64_t idx = order[i];
    float m = mask_values[idx];
    if (has_weight) {
        float w = weight[idx];
        float score = 1.0f / (1.0f + expf(-w));
        m *= score;
    }
    out[i] = m;
}

Tensor apply_mask_cuda(
    const Tensor &mask_values,
    const Tensor &order_,
    const Tensor &weight
) {
    const Tensor order = order_.contiguous();
    int64_t nnz = order.size(0);
    auto out = at::empty({nnz}, mask_values.options().dtype(at::kFloat));
    bool has_weight = weight.defined() && weight.numel() > 0;

    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    if (nnz == 0) return out;
    TORCH_CHECK(mask_values.scalar_type() == at::kBool || mask_values.scalar_type() == at::kFloat,
                "edge visibility mask must be bool or float32");
    if (mask_values.scalar_type() == at::kBool) {
        apply_mask_kernel<bool><<<blocks, threads>>>(
            mask_values.data_ptr<bool>(), order.data_ptr<int32_t>(),
            has_weight ? weight.data_ptr<float>() : nullptr,
            out.data_ptr<float>(), nnz, has_weight);
    } else {
        apply_mask_kernel<float><<<blocks, threads>>>(
            mask_values.data_ptr<float>(), order.data_ptr<int32_t>(),
            has_weight ? weight.data_ptr<float>() : nullptr,
            out.data_ptr<float>(), nnz, has_weight);
    }
    return out;
}

__device__ __forceinline__ void select_group_take_1d(
    int rel,
    int cnt,
    float score,
    int64_t threshold_key,
    int64_t* remain,
    int* take_out
) {
    if (!(score > 0.0f)) {
        *take_out = 0;
        return;
    }
    int take = cnt;
    if (threshold_key >= 0) {
        int64_t key = exact_weight_key(score);
        if (key < threshold_key) {
            take = 0;
        } else if (key == threshold_key) {
            int64_t left = *remain;
            take = left > 0 ? static_cast<int>(left < cnt ? left : cnt) : 0;
            *remain -= take;
        }
    }
    *take_out = take;
}

__global__ void count_sparse3d_topk_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const float* __restrict__ w,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_count,
    int64_t* __restrict__ out_count,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];

    __shared__ int64_t total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    int64_t local = 0;
    for (int64_t g = g0 + threadIdx.x; g < g1; g += blockDim.x)
        if (w[b * L * n + l * n + rel_offset + group_rel[g]] > 0.0f)
            local += group_edge_count[g];
    atomicAdd(reinterpret_cast<unsigned long long*>(&total), static_cast<unsigned long long>(local));
    __syncthreads();
    if (threadIdx.x == 0) out_count[i] = min(total, topk_edges);
}

template <typename scalar_t, bool IsInv>
__global__ void fill_sparse3d_topk_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ col_ind,
    const bool* __restrict__ mask_values,
    const int32_t* __restrict__ order,
    const scalar_t* __restrict__ edge_weight,
    bool has_edge_weight,
    bool edge_weight_is_score,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int64_t* __restrict__ entry_offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t* __restrict__ meta_entry,
    int16_t* __restrict__ meta_rel,
    scalar_t* __restrict__ meta_mask,
    int64_t* __restrict__ meta_edge,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges, int64_t sampling_seed
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    scalar_t sv = sp_value[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];

    __shared__ int counts[RADIX_SIZE];
    __shared__ int64_t threshold_key;
    __shared__ int64_t remain;
    __shared__ int64_t out_base;
    __shared__ int current_take;
    __shared__ int current_rel;
    __shared__ scalar_t group_product;
    __shared__ int32_t current_edge_start;

    __shared__ int64_t total_shared;
    // Training: one weight-proportional integer allocation, then capacity-proportional
    // residual allocation. Two setup scans plus one prefix scan: O(groups + edges).
    // No per-edge random numbers or reordered CSR entries are needed.
    __shared__ double quota_weight_sum, quota_phase;
    __shared__ int64_t quota_target, quota_residual, quota_spare, quota_prefix;
    __shared__ int64_t quota_left, capacity_left;
    if (sampling_seed < 0) {
        exact_group_threshold(w + b * L * n + l * n + rel_offset, group_rel, group_edge_count,
            g0, g1, topk_edges, counts, &threshold_key, &remain, &total_shared);
    } else if (threadIdx.x == 0) {
        int64_t capacity = 0;
        double weight_sum = 0.0;
        for (int64_t g = g0; g < g1; ++g) {
            double weight = w[b * L * n + l * n + rel_offset + group_rel[g]];
            if (weight > 0 && group_edge_count[g] > 0) {
                capacity += group_edge_count[g]; weight_sum += weight;
            }
        }
        quota_target = min(topk_edges, capacity);
        quota_weight_sum = weight_sum;
        int64_t allocated = capacity;
        if (quota_target < capacity) {
            allocated = 0;
            for (int64_t g = g0; g < g1; ++g) {
                double weight = w[b * L * n + l * n + rel_offset + group_rel[g]];
                int64_t count = group_edge_count[g];
                if (weight > 0 && count > 0) {
                    allocated += int64_t(fmin(double(count),
                        floor(double(quota_target) * weight / weight_sum)));
                }
            }
        }
        quota_residual = quota_target - allocated;
        quota_spare = capacity - allocated;
        quota_left = quota_residual;
        capacity_left = quota_spare;
        // SplitMix64: independent phase for each (item, channel, source, direction).
        uint64_t z = uint64_t(sampling_seed) ^ (uint64_t(b) * 0x9e3779b97f4a7c15ULL)
            ^ (uint64_t(l) * 0xbf58476d1ce4e5b9ULL) ^ (uint64_t(src) * 0x94d049bb133111ebULL)
            ^ (IsInv ? 0xd1b54a32d192ed03ULL : 0x8cb92baa3f3d8dd7ULL);
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        quota_phase = double(z >> 11) * 0x1.0p-53;
        quota_prefix = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out_base = entry_offsets[i]; current_take = 0; current_rel = 0; current_edge_start = 0;
    }
    __syncthreads();

    for (int64_t g = g0; g < g1; ++g) {
        if (threadIdx.x == 0) {
            current_rel = group_rel[g];
            group_product = sv * w[b * L * n + l * n + rel_offset + current_rel];
            int cnt = group_edge_count[g];
            scalar_t score = w[b * L * n + l * n + rel_offset + current_rel];
            current_take = score > scalar_t(0) ? cnt : 0;
            if (sampling_seed >= 0 && current_take > 0 && quota_spare > 0) {
                int64_t base = int64_t(fmin(double(cnt),
                    floor(double(quota_target) * double(score) / quota_weight_sum)));
                int64_t spare = int64_t(cnt) - base;
                quota_prefix += spare;
                capacity_left -= spare;
                // Systematic rounding of the residual-capacity prefix. Computing
                // prefixes from integer capacities avoids accumulated quota error.
                int64_t cumulative = quota_prefix == quota_spare ? quota_residual
                    : int64_t(floor(double(quota_residual) * double(quota_prefix)
                                    / double(quota_spare) + quota_phase));
                int64_t extra = cumulative - (quota_residual - quota_left);
                // Preserve exact budget/capacity even at floating-point boundaries.
                extra = max(int64_t(0), min(spare, min(extra, quota_left)));
                extra = max(extra, quota_left - capacity_left);
                quota_left -= extra;
                current_take = int(base + extra);
            } else if (sampling_seed < 0 && current_take > 0 && threshold_key >= 0) {
                select_group_take_1d(current_rel, cnt, score, threshold_key, &remain, &current_take);
            }
            current_edge_start = group_edge_start[g];
        }
        __syncthreads();

        for (int e = threadIdx.x; e < current_take; e += blockDim.x) {
            int64_t p = current_edge_start + e;
            int64_t out = out_base + e;
            int64_t edge_idx = static_cast<int64_t>(order[p]);
            scalar_t m = mask_values[edge_idx] ? scalar_t(1) : scalar_t(0);
            if (has_edge_weight) {
                scalar_t ew = edge_weight[edge_idx];
                scalar_t score = edge_weight_is_score ? ew : scalar_t(1) / (scalar_t(1) + expf(-ew));
                m *= score;
            }
            out_batch[out] = b;
            out_level[out] = l;
            out_entity[out] = col_ind[p];
            out_value[out] = group_product * m;
            meta_entry[out] = i;
            meta_rel[out] = static_cast<int16_t>(current_rel);
            meta_mask[out] = m;
            meta_edge[out] = edge_idx;
        }
        __syncthreads();
        if (threadIdx.x == 0) out_base += current_take;
        __syncthreads();
    }
}

__device__ __forceinline__ int64_t find_relation_group(
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    int64_t source,
    int relation
) {
    int64_t left = row_group_ptr[source];
    int64_t right = row_group_ptr[source + 1];
    while (left < right) {
        int64_t middle = left + (right - left) / 2;
        int value = static_cast<int>(group_rel[middle]);
        if (value < relation) left = middle + 1;
        else right = middle;
    }
    if (left < row_group_ptr[source + 1] &&
        static_cast<int>(group_rel[left]) == relation) {
        return left;
    }
    return -1;
}

__global__ void count_sparse3d_fixed_relation_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const int64_t* __restrict__ operators,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_count,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_count,
    int64_t* __restrict__ counts,
    int64_t nnz,
    int64_t L,
    int64_t r_size
) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nnz) return;
    int64_t batch = sp_batch[index];
    int64_t level = sp_level[index];
    int64_t source = sp_entity[index];
    int64_t operation = operators[batch * L + level];
    if (operation == 2 * r_size) {
        counts[index] = 1;
        return;
    }
    bool inverse = operation >= r_size;
    int relation = static_cast<int>(inverse ? operation - r_size : operation);
    int64_t group = inverse
        ? find_relation_group(inv_row_group_ptr, inv_group_rel, source, relation)
        : find_relation_group(ori_row_group_ptr, ori_group_rel, source, relation);
    if (group < 0) {
        counts[index] = 0;
    } else {
        counts[index] = inverse ? inv_group_edge_count[group] : ori_group_edge_count[group];
    }
}

__global__ void fill_sparse3d_fixed_relation_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const float* __restrict__ sp_value,
    const int64_t* __restrict__ operators,
    const int32_t* __restrict__ ori_col_ind,
    const int32_t* __restrict__ ori_row_group_ptr,
    const int16_t* __restrict__ ori_group_rel,
    const int32_t* __restrict__ ori_group_edge_start,
    const int32_t* __restrict__ inv_col_ind,
    const int32_t* __restrict__ inv_row_group_ptr,
    const int16_t* __restrict__ inv_group_rel,
    const int32_t* __restrict__ inv_group_edge_start,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    float* __restrict__ out_value,
    int64_t nnz,
    int64_t L,
    int64_t r_size
) {
    int64_t index = blockIdx.x;
    if (index >= nnz) return;
    int64_t batch = sp_batch[index];
    int64_t level = sp_level[index];
    int64_t source = sp_entity[index];
    int64_t operation = operators[batch * L + level];
    int64_t output_start = offsets[index];
    int64_t output_count = offsets[index + 1] - output_start;
    if (operation == 2 * r_size) {
        if (threadIdx.x == 0 && output_count == 1) {
            out_batch[output_start] = batch;
            out_level[output_start] = level;
            out_entity[output_start] = source;
            out_value[output_start] = sp_value[index];
        }
        return;
    }
    bool inverse = operation >= r_size;
    int relation = static_cast<int>(inverse ? operation - r_size : operation);
    int64_t group = inverse
        ? find_relation_group(inv_row_group_ptr, inv_group_rel, source, relation)
        : find_relation_group(ori_row_group_ptr, ori_group_rel, source, relation);
    if (group < 0) return;
    int64_t edge_start = inverse ? inv_group_edge_start[group] : ori_group_edge_start[group];
    const int32_t* columns = inverse ? inv_col_ind : ori_col_ind;
    for (int64_t offset = threadIdx.x; offset < output_count; offset += blockDim.x) {
        int64_t output = output_start + offset;
        out_batch[output] = batch;
        out_level[output] = level;
        out_entity[output] = columns[edge_start + offset];
        out_value[output] = sp_value[index];
    }
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
flashlog_forward_sparse3d_fixed_relation_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_,
    const Tensor &sp_value_, const Tensor &operators_,
    const Tensor &ori_col_ind_, const Tensor &ori_row_group_ptr_,
    const Tensor &ori_group_rel_, const Tensor &ori_group_edge_start_,
    const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_row_group_ptr_,
    const Tensor &inv_group_rel_, const Tensor &inv_group_edge_start_,
    const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size
) {
    const Tensor sp_batch = sp_batch_.contiguous().to(at::kLong);
    const Tensor sp_level = sp_level_.contiguous().to(at::kLong);
    const Tensor sp_entity = sp_entity_.contiguous().to(at::kLong);
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor operators = operators_.contiguous().to(at::kLong);
    const Tensor ori_col_ind = ori_col_ind_.contiguous().to(at::kInt);
    const Tensor ori_row_group_ptr = ori_row_group_ptr_.contiguous().to(at::kInt);
    const Tensor ori_group_rel = ori_group_rel_.contiguous().to(at::kShort);
    const Tensor ori_group_edge_start = ori_group_edge_start_.contiguous().to(at::kInt);
    const Tensor ori_group_edge_count = ori_group_edge_count_.contiguous().to(at::kInt);
    const Tensor inv_col_ind = inv_col_ind_.contiguous().to(at::kInt);
    const Tensor inv_row_group_ptr = inv_row_group_ptr_.contiguous().to(at::kInt);
    const Tensor inv_group_rel = inv_group_rel_.contiguous().to(at::kShort);
    const Tensor inv_group_edge_start = inv_group_edge_start_.contiguous().to(at::kInt);
    const Tensor inv_group_edge_count = inv_group_edge_count_.contiguous().to(at::kInt);
    int64_t nnz = sp_batch.numel();
    auto counts = at::zeros({nnz}, sp_batch.options());
    if (nnz > 0) {
        int threads = 256;
        int blocks = static_cast<int>((nnz + threads - 1) / threads);
        count_sparse3d_fixed_relation_kernel<<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(),
            sp_entity.data_ptr<int64_t>(), operators.data_ptr<int64_t>(),
            ori_row_group_ptr.data_ptr<int32_t>(), ori_group_rel.data_ptr<int16_t>(),
            ori_group_edge_count.data_ptr<int32_t>(),
            inv_row_group_ptr.data_ptr<int32_t>(), inv_group_rel.data_ptr<int16_t>(),
            inv_group_edge_count.data_ptr<int32_t>(), counts.data_ptr<int64_t>(),
            nnz, L, r_size
        );
    }
    auto offsets = at::zeros({nnz + 1}, sp_batch.options());
    if (nnz > 0) offsets.slice(0, 1, nnz + 1).copy_(counts.cumsum(0));
    int64_t total = nnz > 0 ? offsets[nnz].item<int64_t>() : 0;
    auto out_batch = at::empty({total}, sp_batch.options());
    auto out_level = at::empty({total}, sp_batch.options());
    auto out_entity = at::empty({total}, sp_batch.options());
    auto out_value = at::empty({total}, sp_value.options());
    if (nnz > 0 && total > 0) {
        fill_sparse3d_fixed_relation_kernel<<<nnz, 128>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(),
            sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            operators.data_ptr<int64_t>(),
            ori_col_ind.data_ptr<int32_t>(), ori_row_group_ptr.data_ptr<int32_t>(),
            ori_group_rel.data_ptr<int16_t>(), ori_group_edge_start.data_ptr<int32_t>(),
            inv_col_ind.data_ptr<int32_t>(), inv_row_group_ptr.data_ptr<int32_t>(),
            inv_group_rel.data_ptr<int16_t>(), inv_group_edge_start.data_ptr<int32_t>(),
            offsets.data_ptr<int64_t>(), out_batch.data_ptr<int64_t>(),
            out_level.data_ptr<int64_t>(), out_entity.data_ptr<int64_t>(),
            out_value.data_ptr<float>(), nnz, L, r_size
        );
    }
    return std::make_tuple(out_batch, out_level, out_entity, out_value);
}

__global__ void count_sparse3d_all_edges_kernel(
    const int64_t* __restrict__ sp_entity,
    const int32_t* __restrict__ row_group_ptr,
    const int32_t* __restrict__ group_edge_count,
    int64_t* __restrict__ out_count,
    int64_t nnz
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t src = sp_entity[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];
    int64_t total = 0;
    for (int64_t g = g0; g < g1; ++g) total += group_edge_count[g];
    out_count[i] = total;
}

template <typename scalar_t, bool IsInv>
__global__ void fill_sparse3d_all_edges_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    const int32_t* __restrict__ col_ind,
    const scalar_t* __restrict__ mask,
    const int32_t* __restrict__ row_group_ptr,
    const int16_t* __restrict__ group_rel,
    const int32_t* __restrict__ group_edge_start,
    const int32_t* __restrict__ group_edge_count,
    const int64_t* __restrict__ entry_offsets,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t* __restrict__ meta_entry,
    int16_t* __restrict__ meta_rel,
    scalar_t* __restrict__ meta_mask,
    int64_t nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset
) {
    int64_t i = blockIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t src = sp_entity[i];
    scalar_t sv = sp_value[i];
    int64_t g0 = row_group_ptr[src], g1 = row_group_ptr[src + 1];
    __shared__ int current_rel;
    __shared__ scalar_t group_product;
    __shared__ int32_t current_edge_start;
    __shared__ int current_take;
    __shared__ int64_t out_base;
    if (threadIdx.x == 0) {
        current_rel = 0;
        current_edge_start = 0;
        current_take = 0;
        out_base = entry_offsets[i];
    }
    __syncthreads();

    for (int64_t g = g0; g < g1; ++g) {
        if (threadIdx.x == 0) {
            current_rel = group_rel[g];
            group_product = sv * w[b * L * n + l * n + rel_offset + current_rel];
            current_edge_start = group_edge_start[g];
            current_take = group_edge_count[g];
        }
        __syncthreads();
        for (int e = threadIdx.x; e < current_take; e += blockDim.x) {
            int64_t p = current_edge_start + e;
            int64_t out = out_base + e;
            scalar_t m = mask[p];
            out_batch[out] = b;
            out_level[out] = l;
            out_entity[out] = col_ind[p];
            out_value[out] = group_product * m;
            meta_entry[out] = i;
            meta_rel[out] = static_cast<int16_t>(current_rel);
            meta_mask[out] = m;
        }
        __syncthreads();
        if (threadIdx.x == 0) out_base += current_take;
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void fill_sparse3d_identity_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    int64_t* __restrict__ out_batch,
    int64_t* __restrict__ out_level,
    int64_t* __restrict__ out_entity,
    scalar_t* __restrict__ out_value,
    int64_t nnz,
    int64_t L,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    out_batch[i] = b;
    out_level[i] = l;
    out_entity[i] = sp_entity[i];
    out_value[i] = sp_value[i] * w[b * L * n + l * n + (n - 1)];
}

template <typename scalar_t, bool CACHED = false, typename index_t = int64_t>
__global__ void backward_sparse3d_topk_entry_kernel(
    const scalar_t* __restrict__ grad_values,
    const index_t* __restrict__ grad_keys,
    const int64_t* __restrict__ meta_entity,
    const int64_t* __restrict__ entry_offsets,
    const int16_t* __restrict__ meta_rel,
    const scalar_t* __restrict__ meta_mask,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t nnz,
    int64_t grad_nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    int64_t E
) {
    int64_t entry = blockIdx.x;
    if (entry >= nnz) return;

    int64_t b = sp_batch[entry];
    int64_t l = sp_level[entry];
    scalar_t sv = sp_value[entry];

    int64_t start = entry_offsets[entry];
    int64_t end = entry_offsets[entry + 1];
    scalar_t local_grad_sp = 0;

    for (int64_t k = start + threadIdx.x; k < end; k += blockDim.x) {
        int64_t idx = -1;
        if constexpr (CACHED) {
            idx = grad_keys[k];
        } else {
            int64_t key = b * (L * E) + l * E + meta_entity[k];
            int64_t lo = 0, hi = grad_nnz - 1;
            while (lo <= hi) {
                int64_t mid = (lo + hi) / 2;
                int64_t mid_key = grad_keys[mid];
                if (mid_key == key) { idx = mid; break; }
                if (mid_key < key) lo = mid + 1;
                else hi = mid - 1;
            }
        }
        if (idx < 0) continue;

        int64_t rel = meta_rel[k];
        scalar_t m = meta_mask[k];
        scalar_t gv = grad_values[idx];
        scalar_t wv = w[b * L * n + l * n + rel_offset + rel];
        local_grad_sp += gv * m * wv;
        atomicAdd(&grad_w[b * L * n + l * n + rel_offset + rel], gv * m * sv);
    }

    if (local_grad_sp != 0) {
        atomicAdd(&grad_sp_value[entry], local_grad_sp);
    }
}

template <typename scalar_t, bool CACHED = false, typename index_t = int64_t>
__global__ void backward_sparse3d_identity_kernel(
    const scalar_t* __restrict__ grad_values,
    const index_t* __restrict__ grad_keys,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const int64_t* __restrict__ sp_entity,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t nnz,
    int64_t grad_nnz,
    int64_t E,
    int64_t n,
    int64_t L
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = sp_batch[i];
    int64_t l = sp_level[i];
    int64_t e = CACHED ? 0 : sp_entity[i];
    int64_t idx = -1;
    if constexpr (CACHED) {
        idx = grad_keys[i];
    } else {
        int64_t key = b * (L * E) + l * E + e;
        int64_t lo = 0, hi = grad_nnz - 1;
        while (lo <= hi) {
            int64_t mid = (lo + hi) / 2;
            int64_t mid_key = grad_keys[mid];
            if (mid_key == key) { idx = mid; break; }
            if (mid_key < key) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    if (idx < 0) return;
    scalar_t gv = grad_values[idx];
    scalar_t wv = w[b * L * n + l * n + (n - 1)];
    atomicAdd(&grad_w[b * L * n + l * n + (n - 1)], gv * sp_value[i]);
    atomicAdd(&grad_sp_value[i], gv * wv);
}

template <typename scalar_t, bool CACHED = false, typename index_t = int64_t>
__global__ void backward_sparse3d_topk_aligned_kernel(
    const scalar_t* __restrict__ grad_raw_values,
    const int64_t* __restrict__ meta_entry,
    const int16_t* __restrict__ meta_rel,
    const scalar_t* __restrict__ meta_mask,
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    const scalar_t* __restrict__ sp_value,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_sp_value,
    scalar_t* __restrict__ grad_w,
    int64_t raw_nnz,
    int64_t L,
    int64_t n,
    int64_t rel_offset,
    const index_t* __restrict__ cached_pos = nullptr
) {
    int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= raw_nnz) return;
    int64_t pos = k;
    if constexpr (CACHED) {
        pos = cached_pos[k];
        if (pos < 0) return;
    }
    scalar_t gv = grad_raw_values[pos];
    if (gv == 0) return;
    int64_t entry = meta_entry[k];
    int64_t b = sp_batch[entry];
    int64_t l = sp_level[entry];
    int64_t rel = meta_rel[k];
    scalar_t m = meta_mask[k];
    scalar_t sv = sp_value[entry];
    scalar_t wv = w[b * L * n + l * n + rel_offset + rel];
    atomicAdd(&grad_sp_value[entry], gv * m * wv);
    atomicAdd(&grad_w[b * L * n + l * n + rel_offset + rel], gv * m * sv);
}

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor, Tensor,Tensor>
flashlog_forward_sparse3d_topk_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_order_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_order_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    const Tensor &mask_values_, const Tensor &edge_weight_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges,
    bool edge_weight_is_score, int64_t sampling_seed
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    TORCH_CHECK(at::isfinite(w).all().item<bool>(),
                "FlashLog sparse top-k received non-finite attention weights; "
                "aborting before edge count/fill to prevent uninitialized indices");
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor oo = ori_order_.contiguous();
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor io = inv_order_.contiguous();
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    const Tensor mask_values = mask_values_.contiguous().to(at::kBool).view({-1});
    const bool has_edge_weight = edge_weight_.defined() && edge_weight_.numel() > 0;
    const Tensor edge_weight = has_edge_weight
        ? edge_weight_.contiguous().to(at::kFloat).view({-1})
        : Tensor();
    int64_t nnz = sp_batch.size(0);
    auto ori_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    auto inv_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    int threads_per_entry = 128;
    if (nnz > 0) {
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogc.data_ptr<int32_t>(),
            ori_count.data_ptr<int64_t>(), nnz, L, w.size(2), 0, topk_edges
        );
        count_sparse3d_topk_kernel<<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            w.data_ptr<float>(), irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igc.data_ptr<int32_t>(),
            inv_count.data_ptr<int64_t>(), nnz, L, w.size(2), r_size, topk_edges
        );
    }
    auto ori_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    auto inv_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    if (nnz > 0) {
        ori_offsets.slice(0, 1, nnz + 1).copy_(ori_count.cumsum(0));
        inv_offsets.slice(0, 1, nnz + 1).copy_(inv_count.cumsum(0));
    }
    int64_t ori_total = ori_offsets[nnz].item<int64_t>();
    int64_t inv_total = inv_offsets[nnz].item<int64_t>();
    auto ori_b = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_l = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_e = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_v = at::empty({ori_total}, sp_value.options());
    auto ori_meta_entry = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_meta_rel = at::empty({ori_total}, sp_batch.options().dtype(at::kShort));
    auto ori_meta_mask = at::empty({ori_total}, sp_value.options());
    auto ori_meta_edge = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto inv_b = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_l = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_e = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_v = at::empty({inv_total}, sp_value.options());
    auto inv_meta_entry = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_meta_rel = at::empty({inv_total}, sp_batch.options().dtype(at::kShort));
    auto inv_meta_mask = at::empty({inv_total}, sp_value.options());
    auto inv_meta_edge = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    if (ori_total > 0) {
        fill_sparse3d_topk_kernel<float, false><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), oci.data_ptr<int32_t>(), mask_values.data_ptr<bool>(), oo.data_ptr<int32_t>(),
            has_edge_weight ? edge_weight.data_ptr<float>() : nullptr, has_edge_weight, edge_weight_is_score,
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_offsets.data_ptr<int64_t>(),
            ori_b.data_ptr<int64_t>(), ori_l.data_ptr<int64_t>(), ori_e.data_ptr<int64_t>(), ori_v.data_ptr<float>(),
            ori_meta_entry.data_ptr<int64_t>(), ori_meta_rel.data_ptr<int16_t>(), ori_meta_mask.data_ptr<float>(),
            ori_meta_edge.data_ptr<int64_t>(),
            nnz, L, w.size(2), 0, topk_edges, sampling_seed
        );
    }
    if (inv_total > 0) {
        fill_sparse3d_topk_kernel<float, true><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), ici.data_ptr<int32_t>(), mask_values.data_ptr<bool>(), io.data_ptr<int32_t>(),
            has_edge_weight ? edge_weight.data_ptr<float>() : nullptr, has_edge_weight, edge_weight_is_score,
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_offsets.data_ptr<int64_t>(),
            inv_b.data_ptr<int64_t>(), inv_l.data_ptr<int64_t>(), inv_e.data_ptr<int64_t>(), inv_v.data_ptr<float>(),
            inv_meta_entry.data_ptr<int64_t>(), inv_meta_rel.data_ptr<int16_t>(), inv_meta_mask.data_ptr<float>(),
            inv_meta_edge.data_ptr<int64_t>(),
            nnz, L, w.size(2), r_size, topk_edges, sampling_seed
        );
    }
    int64_t ind_nnz = wot_i ? 0 : nnz;
    auto ind_b = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_l = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_e = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_v = at::zeros({ind_nnz}, sp_value.options());
    if (!wot_i && nnz > 0) {
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        fill_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            ind_b.data_ptr<int64_t>(), ind_l.data_ptr<int64_t>(), ind_e.data_ptr<int64_t>(), ind_v.data_ptr<float>(),
            nnz, L, w.size(2)
        );
    }
    return std::make_tuple(
        ori_b, ori_l, ori_e, ori_v,
        inv_b, inv_l, inv_e, inv_v,
        ind_b, ind_l, ind_e, ind_v,
        ori_meta_entry, ori_meta_rel, ori_meta_mask,
        inv_meta_entry, inv_meta_rel, inv_meta_mask,
        ori_offsets, inv_offsets,
        ori_meta_edge, inv_meta_edge
    );
}

std::tuple<Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor, Tensor,Tensor,Tensor, Tensor,Tensor>
flashlog_forward_sparse3d_alledges_cuda(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    const Tensor oci = ori_col_ind_.contiguous();
    const Tensor om = ori_mask_.contiguous().to(at::kFloat);
    const Tensor orgp = ori_row_group_ptr_.contiguous();
    const Tensor ogr = ori_group_rel_.contiguous();
    const Tensor ogs = ori_group_edge_start_.contiguous();
    const Tensor ogc = ori_group_edge_count_.contiguous();
    const Tensor ici = inv_col_ind_.contiguous();
    const Tensor im = inv_mask_.contiguous().to(at::kFloat);
    const Tensor irgp = inv_row_group_ptr_.contiguous();
    const Tensor igr = inv_group_rel_.contiguous();
    const Tensor igs = inv_group_edge_start_.contiguous();
    const Tensor igc = inv_group_edge_count_.contiguous();
    int64_t nnz = sp_batch.size(0);
    auto ori_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    auto inv_count = at::zeros({nnz}, sp_batch.options().dtype(at::kLong));
    int threads_per_entry = 128;
    if (nnz > 0) {
        int blocks = static_cast<int>((nnz + 255) / 256);
        count_sparse3d_all_edges_kernel<<<blocks, 256>>>(
            sp_entity.data_ptr<int64_t>(), orgp.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_count.data_ptr<int64_t>(), nnz
        );
        count_sparse3d_all_edges_kernel<<<blocks, 256>>>(
            sp_entity.data_ptr<int64_t>(), irgp.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_count.data_ptr<int64_t>(), nnz
        );
    }
    auto ori_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    auto inv_offsets = at::zeros({nnz + 1}, sp_batch.options().dtype(at::kLong));
    if (nnz > 0) {
        ori_offsets.slice(0, 1, nnz + 1).copy_(ori_count.cumsum(0));
        inv_offsets.slice(0, 1, nnz + 1).copy_(inv_count.cumsum(0));
    }
    int64_t ori_total = ori_offsets[nnz].item<int64_t>();
    int64_t inv_total = inv_offsets[nnz].item<int64_t>();
    auto ori_b = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_l = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_e = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_v = at::empty({ori_total}, sp_value.options());
    auto ori_meta_entry = at::empty({ori_total}, sp_batch.options().dtype(at::kLong));
    auto ori_meta_rel = at::empty({ori_total}, sp_batch.options().dtype(at::kShort));
    auto ori_meta_mask = at::empty({ori_total}, sp_value.options());
    auto inv_b = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_l = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_e = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_v = at::empty({inv_total}, sp_value.options());
    auto inv_meta_entry = at::empty({inv_total}, sp_batch.options().dtype(at::kLong));
    auto inv_meta_rel = at::empty({inv_total}, sp_batch.options().dtype(at::kShort));
    auto inv_meta_mask = at::empty({inv_total}, sp_value.options());
    if (ori_total > 0) {
        fill_sparse3d_all_edges_kernel<float, false><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), oci.data_ptr<int32_t>(), om.data_ptr<float>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(), ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ori_offsets.data_ptr<int64_t>(),
            ori_b.data_ptr<int64_t>(), ori_l.data_ptr<int64_t>(), ori_e.data_ptr<int64_t>(), ori_v.data_ptr<float>(),
            ori_meta_entry.data_ptr<int64_t>(), ori_meta_rel.data_ptr<int16_t>(), ori_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), 0
        );
    }
    if (inv_total > 0) {
        fill_sparse3d_all_edges_kernel<float, true><<<nnz, threads_per_entry>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(), sp_value.data_ptr<float>(),
            w.data_ptr<float>(), ici.data_ptr<int32_t>(), im.data_ptr<float>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(), igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            inv_offsets.data_ptr<int64_t>(),
            inv_b.data_ptr<int64_t>(), inv_l.data_ptr<int64_t>(), inv_e.data_ptr<int64_t>(), inv_v.data_ptr<float>(),
            inv_meta_entry.data_ptr<int64_t>(), inv_meta_rel.data_ptr<int16_t>(), inv_meta_mask.data_ptr<float>(),
            nnz, L, w.size(2), r_size
        );
    }
    int64_t ind_nnz = wot_i ? 0 : nnz;
    auto ind_b = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_l = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_e = at::empty({ind_nnz}, sp_batch.options().dtype(at::kLong));
    auto ind_v = at::zeros({ind_nnz}, sp_value.options());
    if (!wot_i && nnz > 0) {
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        fill_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            ind_b.data_ptr<int64_t>(), ind_l.data_ptr<int64_t>(), ind_e.data_ptr<int64_t>(), ind_v.data_ptr<float>(),
            nnz, L, w.size(2)
        );
    }
    return std::make_tuple(
        ori_b, ori_l, ori_e, ori_v,
        inv_b, inv_l, inv_e, inv_v,
        ind_b, ind_l, ind_e, ind_v,
        ori_meta_entry, ori_meta_rel, ori_meta_mask,
        inv_meta_entry, inv_meta_rel, inv_meta_mask,
        ori_offsets, inv_offsets
    );
}

std::tuple<Tensor, Tensor>
flashlog_backward_sparse3d_topk_cuda(
    const Tensor &grad_ori_values_, const Tensor &grad_ori_batch_, const Tensor &grad_ori_level_, const Tensor &grad_ori_entity_,
    const Tensor &grad_inv_values_, const Tensor &grad_inv_batch_, const Tensor &grad_inv_level_, const Tensor &grad_inv_entity_,
    const Tensor &grad_ind_values_, const Tensor &grad_ind_batch_, const Tensor &grad_ind_level_, const Tensor &grad_ind_entity_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_, const Tensor &w_,
    const Tensor &ori_meta_batch_, const Tensor &ori_meta_level_, const Tensor &ori_meta_entity_,
    const Tensor &ori_meta_entry_, const Tensor &ori_meta_rel_, const Tensor &ori_meta_mask_, const Tensor &ori_offsets_,
    const Tensor &inv_meta_batch_, const Tensor &inv_meta_level_, const Tensor &inv_meta_entity_,
    const Tensor &inv_meta_entry_, const Tensor &inv_meta_rel_, const Tensor &inv_meta_mask_, const Tensor &inv_offsets_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    int64_t nnz = sp_batch.size(0);
    int64_t n = w.size(2);
    auto grad_sp_value = at::zeros({nnz}, sp_value.options());
    auto grad_w = at::zeros({B, L, n}, w.options());
    const Tensor ori_meta_entity = ori_meta_entity_.contiguous();
    const Tensor ori_offsets = ori_offsets_.contiguous();
    if (grad_ori_batch_.size(0) > 0 && ori_meta_entry_.size(0) > 0) {
        auto gb = grad_ori_batch_.contiguous();
        auto gl = grad_ori_level_.contiguous();
        auto ge = grad_ori_entity_.contiguous();
        auto gv = grad_ori_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 128, blocks = static_cast<int>(nnz);
        backward_sparse3d_topk_entry_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            ori_meta_entity.data_ptr<int64_t>(),
            ori_offsets.data_ptr<int64_t>(), ori_meta_rel_.data_ptr<int16_t>(), ori_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(),
            nnz, gb.size(0), L, n, 0, E
        );
    }
    const Tensor inv_meta_entity = inv_meta_entity_.contiguous();
    const Tensor inv_offsets = inv_offsets_.contiguous();
    if (grad_inv_batch_.size(0) > 0 && inv_meta_entry_.size(0) > 0) {
        auto gb = grad_inv_batch_.contiguous();
        auto gl = grad_inv_level_.contiguous();
        auto ge = grad_inv_entity_.contiguous();
        auto gv = grad_inv_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 128, blocks = static_cast<int>(nnz);
        backward_sparse3d_topk_entry_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            inv_meta_entity.data_ptr<int64_t>(),
            inv_offsets.data_ptr<int64_t>(), inv_meta_rel_.data_ptr<int16_t>(), inv_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(),
            nnz, gb.size(0), L, n, r_size, E
        );
    }
    if (!wot_i && grad_ind_batch_.size(0) > 0) {
        auto gb = grad_ind_batch_.contiguous();
        auto gl = grad_ind_level_.contiguous();
        auto ge = grad_ind_entity_.contiguous();
        auto gv = grad_ind_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int threads = 256, blocks = (nnz + threads - 1) / threads;
        backward_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(), nnz, gv.size(0), E, n, L
        );
    }
    return std::make_tuple(grad_sp_value, grad_w);
}

std::tuple<Tensor, Tensor>
flashlog_backward_sparse3d_cached_cuda(
    const Tensor &grad_, const Tensor &ori_pos_, const Tensor &inv_pos_, const Tensor &ind_pos_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_value_, const Tensor &w_,
    const Tensor &ori_rel_, const Tensor &ori_mask_, const Tensor &ori_offsets_,
    const Tensor &inv_rel_, const Tensor &inv_mask_, const Tensor &inv_offsets_,
    int64_t B, int64_t L, int64_t r_size, bool wot_i
) {
    auto grad = grad_.contiguous().to(at::kFloat);
    auto sb = sp_batch_.contiguous();
    auto sl = sp_level_.contiguous();
    auto sv = sp_value_.contiguous().to(at::kFloat);
    auto w = w_.contiguous().to(at::kFloat);
    auto op = ori_pos_.contiguous(), ip = inv_pos_.contiguous(), dp = ind_pos_.contiguous();
    TORCH_CHECK(grad.is_cuda() && op.is_cuda() && ip.is_cuda() && dp.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(op.scalar_type() == ip.scalar_type() && op.scalar_type() == dp.scalar_type(), "cache index dtypes must match");
    TORCH_CHECK(op.numel() == ori_rel_.numel() && ip.numel() == inv_rel_.numel(), "cache size mismatch");
    int64_t nnz = sv.numel(), n = w.size(2);
    TORCH_CHECK(wot_i || dp.numel() == nnz, "identity cache size mismatch");
    auto gs = at::zeros_like(sv), gw = at::zeros({B, L, n}, w.options());
    if (nnz == 0 || grad.numel() == 0) return std::make_tuple(gs, gw);
    // Same source-entry blocks, arithmetic, and atomics as the old search kernel.
    AT_DISPATCH_INDEX_TYPES(op.scalar_type(), "sparse3d_cached_backward", [&] {
        if (op.numel()) {
            backward_sparse3d_topk_entry_kernel<float, true, index_t><<<nnz, 128>>>(
                grad.data_ptr<float>(), op.data_ptr<index_t>(), nullptr,
                ori_offsets_.data_ptr<int64_t>(), ori_rel_.data_ptr<int16_t>(), ori_mask_.data_ptr<float>(),
                sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), sv.data_ptr<float>(), w.data_ptr<float>(),
                gs.data_ptr<float>(), gw.data_ptr<float>(), nnz, grad.numel(), L, n, 0, 0);
        }
        if (ip.numel()) {
            backward_sparse3d_topk_entry_kernel<float, true, index_t><<<nnz, 128>>>(
                grad.data_ptr<float>(), ip.data_ptr<index_t>(), nullptr,
                inv_offsets_.data_ptr<int64_t>(), inv_rel_.data_ptr<int16_t>(), inv_mask_.data_ptr<float>(),
                sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), sv.data_ptr<float>(), w.data_ptr<float>(),
                gs.data_ptr<float>(), gw.data_ptr<float>(), nnz, grad.numel(), L, n, r_size, 0);
        }
        if (!wot_i) {
            backward_sparse3d_identity_kernel<float, true, index_t><<<(nnz + 255) / 256, 256>>>(
                grad.data_ptr<float>(), dp.data_ptr<index_t>(), sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), nullptr,
                sv.data_ptr<float>(), w.data_ptr<float>(), gs.data_ptr<float>(), gw.data_ptr<float>(), nnz, grad.numel(), 0, n, L);
        }
    });
    return std::make_tuple(gs, gw);
}

std::tuple<Tensor, Tensor>
flashlog_backward_sparse3d_max_cached_cuda(
    const Tensor &grad_, const Tensor &ori_pos_, const Tensor &inv_pos_, const Tensor &ind_pos_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_value_, const Tensor &w_,
    const Tensor &ori_entry_, const Tensor &ori_rel_, const Tensor &ori_mask_,
    const Tensor &inv_entry_, const Tensor &inv_rel_, const Tensor &inv_mask_,
    int64_t B, int64_t L, int64_t r_size, bool wot_i
) {
    auto grad = grad_.contiguous().to(at::kFloat);
    auto sb = sp_batch_.contiguous(), sl = sp_level_.contiguous();
    auto sv = sp_value_.contiguous().to(at::kFloat), w = w_.contiguous().to(at::kFloat);
    auto op = ori_pos_.contiguous(), ip = inv_pos_.contiguous(), dp = ind_pos_.contiguous();
    TORCH_CHECK(grad.is_cuda() && op.is_cuda() && ip.is_cuda() && dp.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(op.scalar_type() == ip.scalar_type() && op.scalar_type() == dp.scalar_type(), "cache index dtypes must match");
    TORCH_CHECK(op.numel() == ori_entry_.numel() && ip.numel() == inv_entry_.numel(), "cache size mismatch");
    int64_t nnz = sv.numel(), n = w.size(2);
    TORCH_CHECK(wot_i || dp.numel() == nnz, "identity cache size mismatch");
    auto gs = at::zeros_like(sv), gw = at::zeros({B, L, n}, w.options());
    if (nnz == 0 || grad.numel() == 0) return std::make_tuple(gs, gw);
    // Preserve the original max backward's per-edge threads and arithmetic.
    // Only replace gradient lookup with the forward winner's cached position.
    AT_DISPATCH_INDEX_TYPES(op.scalar_type(), "sparse3d_max_cached_backward", [&] {
        if (op.numel()) {
            backward_sparse3d_topk_aligned_kernel<float, true, index_t><<<(op.numel() + 255) / 256, 256>>>(
                grad.data_ptr<float>(), ori_entry_.data_ptr<int64_t>(),
                ori_rel_.data_ptr<int16_t>(), ori_mask_.data_ptr<float>(),
                sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), sv.data_ptr<float>(), w.data_ptr<float>(),
                gs.data_ptr<float>(), gw.data_ptr<float>(), op.numel(), L, n, 0, op.data_ptr<index_t>());
        }
        if (ip.numel()) {
            backward_sparse3d_topk_aligned_kernel<float, true, index_t><<<(ip.numel() + 255) / 256, 256>>>(
                grad.data_ptr<float>(), inv_entry_.data_ptr<int64_t>(),
                inv_rel_.data_ptr<int16_t>(), inv_mask_.data_ptr<float>(),
                sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), sv.data_ptr<float>(), w.data_ptr<float>(),
                gs.data_ptr<float>(), gw.data_ptr<float>(), ip.numel(), L, n, r_size, ip.data_ptr<index_t>());
        }
        if (!wot_i) {
            backward_sparse3d_identity_kernel<float, true, index_t><<<(nnz + 255) / 256, 256>>>(
                grad.data_ptr<float>(), dp.data_ptr<index_t>(), sb.data_ptr<int64_t>(), sl.data_ptr<int64_t>(), nullptr,
                sv.data_ptr<float>(), w.data_ptr<float>(), gs.data_ptr<float>(), gw.data_ptr<float>(), nnz, grad.numel(), 0, n, L);
        }
    });
    return std::make_tuple(gs, gw);
}

std::tuple<Tensor, Tensor>
flashlog_backward_sparse3d_topk_aligned_cuda(
    const Tensor &grad_ori_raw_values_,
    const Tensor &ori_meta_entry_, const Tensor &ori_meta_rel_, const Tensor &ori_meta_mask_,
    const Tensor &grad_inv_raw_values_,
    const Tensor &inv_meta_entry_, const Tensor &inv_meta_rel_, const Tensor &inv_meta_mask_,
    const Tensor &grad_ind_values_, const Tensor &grad_ind_batch_, const Tensor &grad_ind_level_, const Tensor &grad_ind_entity_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_, const Tensor &w_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor sp_entity = sp_entity_.contiguous();
    const Tensor sp_value = sp_value_.contiguous().to(at::kFloat);
    const Tensor w = w_.contiguous().to(at::kFloat);
    int64_t nnz = sp_batch.size(0);
    int64_t n = w.size(2);
    auto grad_sp_value = at::zeros({nnz}, sp_value.options());
    auto grad_w = at::zeros({B, L, n}, w.options());

    int threads = 256;
    if (grad_ori_raw_values_.numel() > 0) {
        auto gv = grad_ori_raw_values_.contiguous().to(at::kFloat);
        int blocks = (gv.size(0) + threads - 1) / threads;
        backward_sparse3d_topk_aligned_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(),
            ori_meta_entry_.data_ptr<int64_t>(),
            ori_meta_rel_.data_ptr<int16_t>(),
            ori_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(),
            sp_level.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            gv.size(0), L, n, 0
        );
    }
    if (grad_inv_raw_values_.numel() > 0) {
        auto gv = grad_inv_raw_values_.contiguous().to(at::kFloat);
        int blocks = (gv.size(0) + threads - 1) / threads;
        backward_sparse3d_topk_aligned_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(),
            inv_meta_entry_.data_ptr<int64_t>(),
            inv_meta_rel_.data_ptr<int16_t>(),
            inv_meta_mask_.data_ptr<float>(),
            sp_batch.data_ptr<int64_t>(),
            sp_level.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(),
            w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(),
            grad_w.data_ptr<float>(),
            gv.size(0), L, n, r_size
        );
    }
    if (!wot_i && grad_ind_batch_.size(0) > 0) {
        auto gb = grad_ind_batch_.contiguous();
        auto gl = grad_ind_level_.contiguous();
        auto ge = grad_ind_entity_.contiguous();
        auto gv = grad_ind_values_.contiguous().to(at::kFloat);
        auto gkeys = gb * (L * E) + gl * E + ge;
        int blocks = (nnz + threads - 1) / threads;
        backward_sparse3d_identity_kernel<float><<<blocks, threads>>>(
            gv.data_ptr<float>(), gkeys.data_ptr<int64_t>(),
            sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
            sp_value.data_ptr<float>(), w.data_ptr<float>(),
            grad_sp_value.data_ptr<float>(), grad_w.data_ptr<float>(), nnz, gv.size(0), E, n, L
        );
    }
    return std::make_tuple(grad_sp_value, grad_w);
}

template <typename scalar_t>
__global__ void sparse3d_mark_bl_segments_kernel(
    const int64_t* __restrict__ sp_batch,
    const int64_t* __restrict__ sp_level,
    int64_t* __restrict__ row_start,
    int64_t* __restrict__ row_end,
    int64_t nnz,
    int64_t L
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int64_t key = sp_batch[i] * L + sp_level[i];
    if (i == 0) {
        row_start[key] = 0;
    } else {
        int64_t prev_key = sp_batch[i - 1] * L + sp_level[i - 1];
        if (prev_key != key) row_start[key] = i;
    }

    if (i == nnz - 1) {
        row_end[key] = nnz;
    } else {
        int64_t next_key = sp_batch[i + 1] * L + sp_level[i + 1];
        if (next_key != key) row_end[key] = i + 1;
    }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void sparse3d_normalize_forward_segments_kernel(
    const scalar_t* __restrict__ values,
    const int64_t* __restrict__ row_start,
    const int64_t* __restrict__ row_end,
    scalar_t* __restrict__ out_values,
    scalar_t* __restrict__ denom,
    int64_t groups,
    scalar_t eps
) {
    int64_t g = blockIdx.x;
    if (g >= groups) return;

    int64_t start = row_start[g];
    int64_t end = row_end[g];
    if (end <= start) return;

    __shared__ scalar_t shared[BLOCK_SIZE];
    scalar_t local_sum = static_cast<scalar_t>(0);
    for (int64_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        local_sum += values[i];
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    scalar_t raw = shared[0];
    if (threadIdx.x == 0) denom[g] = raw;
    scalar_t d = raw >= eps ? raw : eps;
    __syncthreads();

    for (int64_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        out_values[i] = values[i] / d;
    }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void sparse3d_normalize_backward_segments_kernel(
    const scalar_t* __restrict__ grad_out_values,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ denom,
    const int64_t* __restrict__ row_start,
    const int64_t* __restrict__ row_end,
    scalar_t* __restrict__ grad_values,
    int64_t groups,
    scalar_t eps
) {
    int64_t g = blockIdx.x;
    if (g >= groups) return;

    int64_t start = row_start[g];
    int64_t end = row_end[g];
    if (end <= start) return;

    __shared__ scalar_t shared[BLOCK_SIZE];
    scalar_t local_sum = static_cast<scalar_t>(0);
    for (int64_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        local_sum += grad_out_values[i] * values[i];
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    scalar_t raw = denom[g];
    bool active = raw >= eps;
    scalar_t d = active ? raw : eps;
    scalar_t correction = active ? (shared[0] / (d * d)) : static_cast<scalar_t>(0);

    for (int64_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        grad_values[i] = grad_out_values[i] / d - correction;
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
sparse3d_normalize_forward_cuda(
    const Tensor &sp_batch_,
    const Tensor &sp_level_,
    const Tensor &values_,
    int64_t B,
    int64_t L,
    double eps
) {
    TORCH_CHECK(sp_batch_.is_cuda(), "sp_batch must be a CUDA tensor");
    TORCH_CHECK(sp_level_.is_cuda(), "sp_level must be a CUDA tensor");
    TORCH_CHECK(values_.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(sp_batch_.scalar_type() == at::kLong, "sp_batch must be int64");
    TORCH_CHECK(sp_level_.scalar_type() == at::kLong, "sp_level must be int64");

    const Tensor sp_batch = sp_batch_.contiguous();
    const Tensor sp_level = sp_level_.contiguous();
    const Tensor values = values_.contiguous();
    int64_t nnz = values.size(0);
    int64_t groups = B * L;

    auto out_values = at::empty_like(values);
    auto denom = at::zeros({B, L}, values.options());
    auto row_start = at::zeros({groups}, sp_batch.options());
    auto row_end = at::zeros({groups}, sp_batch.options());

    if (nnz == 0 || groups == 0) {
        return std::make_tuple(out_values, denom, row_start, row_end);
    }

    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    sparse3d_mark_bl_segments_kernel<float><<<blocks, threads>>>(
        sp_batch.data_ptr<int64_t>(),
        sp_level.data_ptr<int64_t>(),
        row_start.data_ptr<int64_t>(),
        row_end.data_ptr<int64_t>(),
        nnz,
        L
    );

    constexpr int BLOCK_SIZE = 256;
    int group_blocks = static_cast<int>(groups);
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse3d_normalize_forward_cuda", [&] {
        sparse3d_normalize_forward_segments_kernel<scalar_t, BLOCK_SIZE><<<group_blocks, BLOCK_SIZE>>>(
            values.data_ptr<scalar_t>(),
            row_start.data_ptr<int64_t>(),
            row_end.data_ptr<int64_t>(),
            out_values.data_ptr<scalar_t>(),
            denom.data_ptr<scalar_t>(),
            groups,
            static_cast<scalar_t>(eps)
        );
    });

    return std::make_tuple(out_values, denom, row_start, row_end);
}

Tensor sparse3d_normalize_backward_cuda(
    const Tensor &grad_out_values_,
    const Tensor &values_,
    const Tensor &denom_,
    const Tensor &row_start_,
    const Tensor &row_end_,
    double eps
) {
    TORCH_CHECK(grad_out_values_.is_cuda(), "grad_out_values must be a CUDA tensor");
    TORCH_CHECK(values_.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(denom_.is_cuda(), "denom must be a CUDA tensor");

    const Tensor grad_out_values = grad_out_values_.contiguous();
    const Tensor values = values_.contiguous();
    const Tensor denom = denom_.contiguous().view({-1});
    const Tensor row_start = row_start_.contiguous();
    const Tensor row_end = row_end_.contiguous();

    auto grad_values = at::empty_like(values);
    int64_t groups = denom.size(0);
    if (values.size(0) == 0 || groups == 0) {
        return grad_values;
    }

    constexpr int BLOCK_SIZE = 256;
    int group_blocks = static_cast<int>(groups);
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse3d_normalize_backward_cuda", [&] {
        sparse3d_normalize_backward_segments_kernel<scalar_t, BLOCK_SIZE><<<group_blocks, BLOCK_SIZE>>>(
            grad_out_values.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            denom.data_ptr<scalar_t>(),
            row_start.data_ptr<int64_t>(),
            row_end.data_ptr<int64_t>(),
            grad_values.data_ptr<scalar_t>(),
            groups,
            static_cast<scalar_t>(eps)
        );
    });

    return grad_values;
}

template <typename scalar_t>
__global__ void append_sparse3d_sum_keys_kernel(
    const int64_t* __restrict__ b,
    const int64_t* __restrict__ l,
    const int64_t* __restrict__ e,
    const scalar_t* __restrict__ v,
    int64_t* __restrict__ keys,
    scalar_t* __restrict__ values,
    int64_t offset,
    int64_t n,
    int64_t L,
    int64_t E
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t out = offset + i;
    keys[out] = b[i] * (L * E) + l[i] * E + e[i];
    values[out] = v[i];
}

template <typename scalar_t>
__global__ void unpack_sparse3d_sum_keys_kernel(
    const int64_t* __restrict__ keys,
    const scalar_t* __restrict__ values,
    int64_t* __restrict__ out_b,
    int64_t* __restrict__ out_l,
    int64_t* __restrict__ out_e,
    scalar_t* __restrict__ out_v,
    int64_t n,
    int64_t L,
    int64_t E
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t key = keys[i];
    out_b[i] = key / (L * E);
    int64_t rem = key % (L * E);
    out_l[i] = rem / E;
    out_e[i] = rem % E;
    out_v[i] = values[i];
}

__device__ __forceinline__ bool binary_search_i64(
    const int64_t* __restrict__ keys,
    int64_t n,
    int64_t key
) {
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        int64_t mid = lo + ((hi - lo) >> 1);
        int64_t v = keys[mid];
        if (v < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < n && keys[lo] == key;
}

template <typename scalar_t>
__global__ void sparse2d_truth_scores_kernel(
    const int64_t* __restrict__ idx_b,
    const int64_t* __restrict__ idx_e,
    const scalar_t* __restrict__ values,
    const int64_t* __restrict__ targets,
    scalar_t* __restrict__ truth,
    int64_t nnz
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = idx_b[i];
    if (idx_e[i] == targets[b]) {
        truth[b] = values[i];
    }
}

__global__ void sparse2d_filter_counts_kernel(
    const int64_t* __restrict__ filter_keys,
    float* __restrict__ filter_counts,
    int64_t n_filter,
    int64_t E
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_filter) return;
    int64_t b = filter_keys[i] / E;
    atomicAdd(filter_counts + b, 1.0f);
}

template <typename scalar_t>
__global__ void sparse2d_rank_counts_kernel(
    const int64_t* __restrict__ idx_b,
    const int64_t* __restrict__ idx_e,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ truth,
    const int64_t* __restrict__ filter_keys,
    float* __restrict__ greater,
    float* __restrict__ equal,
    float* __restrict__ explicit_unfiltered,
    int64_t nnz,
    int64_t n_filter,
    int64_t E
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int64_t b = idx_b[i];
    int64_t e = idx_e[i];
    int64_t key = b * E + e;
    if (binary_search_i64(filter_keys, n_filter, key)) {
        return;
    }
    scalar_t v = values[i];
    scalar_t t = truth[b];
    atomicAdd(explicit_unfiltered + b, 1.0f);
    if (v > t) {
        atomicAdd(greater + b, 1.0f);
    } else if (v == t) {
        atomicAdd(equal + b, 1.0f);
    }
}

template <typename scalar_t>
__global__ void sparse2d_finalize_ranks_kernel(
    const scalar_t* __restrict__ truth,
    const float* __restrict__ greater,
    const float* __restrict__ equal,
    const float* __restrict__ explicit_unfiltered,
    const float* __restrict__ filter_counts,
    float* __restrict__ ranks,
    int64_t B,
    int64_t E,
    bool use_eql
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    if (use_eql) {
        float missing_unfiltered = static_cast<float>(E) - filter_counts[b] - explicit_unfiltered[b];
        float eq = equal[b] + (truth[b] == static_cast<scalar_t>(0) ? missing_unfiltered : 0.0f);
        eq += 1.0f;
        ranks[b] = greater[b] + (eq + 1.0f) * 0.5f;
    } else {
        ranks[b] = greater[b] + 1.0f;
    }
}

Tensor sparse2d_filtered_ranks_cuda(
    const Tensor &idx_b_,
    const Tensor &idx_e_,
    const Tensor &values_,
    const Tensor &targets_,
    const Tensor &filter_keys_,
    int64_t B,
    int64_t E,
    bool use_eql
) {
    TORCH_CHECK(idx_b_.is_cuda(), "idx_b must be a CUDA tensor");
    TORCH_CHECK(idx_e_.is_cuda(), "idx_e must be a CUDA tensor");
    TORCH_CHECK(values_.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(targets_.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(filter_keys_.is_cuda(), "filter_keys must be a CUDA tensor");
    TORCH_CHECK(idx_b_.scalar_type() == at::kLong, "idx_b must be int64");
    TORCH_CHECK(idx_e_.scalar_type() == at::kLong, "idx_e must be int64");
    TORCH_CHECK(targets_.scalar_type() == at::kLong, "targets must be int64");
    TORCH_CHECK(filter_keys_.scalar_type() == at::kLong, "filter_keys must be int64");

    const Tensor idx_b = idx_b_.contiguous();
    const Tensor idx_e = idx_e_.contiguous();
    const Tensor values = values_.contiguous();
    const Tensor targets = targets_.contiguous();
    const Tensor filter_keys = filter_keys_.contiguous();
    int64_t nnz = values.size(0);
    int64_t n_filter = filter_keys.size(0);

    Tensor truth = at::zeros({B}, values.options());
    Tensor greater = at::zeros({B}, values.options().dtype(at::kFloat));
    Tensor equal = at::zeros({B}, values.options().dtype(at::kFloat));
    Tensor explicit_unfiltered = at::zeros({B}, values.options().dtype(at::kFloat));
    Tensor filter_counts = at::zeros({B}, values.options().dtype(at::kFloat));
    Tensor ranks = at::empty({B}, values.options().dtype(at::kFloat));

    const int threads = 256;
    if (nnz > 0) {
        int blocks = static_cast<int>((nnz + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse2d_truth_scores_cuda", [&] {
            sparse2d_truth_scores_kernel<scalar_t><<<blocks, threads>>>(
                idx_b.data_ptr<int64_t>(), idx_e.data_ptr<int64_t>(), values.data_ptr<scalar_t>(),
                targets.data_ptr<int64_t>(), truth.data_ptr<scalar_t>(), nnz
            );
        });
    }
    if (n_filter > 0) {
        int blocks = static_cast<int>((n_filter + threads - 1) / threads);
        sparse2d_filter_counts_kernel<<<blocks, threads>>>(
            filter_keys.data_ptr<int64_t>(), filter_counts.data_ptr<float>(), n_filter, E
        );
    }
    if (nnz > 0) {
        int blocks = static_cast<int>((nnz + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse2d_rank_counts_cuda", [&] {
            sparse2d_rank_counts_kernel<scalar_t><<<blocks, threads>>>(
                idx_b.data_ptr<int64_t>(), idx_e.data_ptr<int64_t>(), values.data_ptr<scalar_t>(),
                truth.data_ptr<scalar_t>(), filter_keys.data_ptr<int64_t>(),
                greater.data_ptr<float>(), equal.data_ptr<float>(), explicit_unfiltered.data_ptr<float>(),
                nnz, n_filter, E
            );
        });
    }
    int blocks = static_cast<int>((B + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse2d_finalize_ranks_cuda", [&] {
        sparse2d_finalize_ranks_kernel<scalar_t><<<blocks, threads>>>(
            truth.data_ptr<scalar_t>(), greater.data_ptr<float>(), equal.data_ptr<float>(),
            explicit_unfiltered.data_ptr<float>(), filter_counts.data_ptr<float>(),
            ranks.data_ptr<float>(), B, E, use_eql
        );
    });
    return ranks;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sparse3d_coalesce_sum3_impl(
    const Tensor &b0_, const Tensor &l0_, const Tensor &e0_, const Tensor &v0_,
    const Tensor &b1_, const Tensor &l1_, const Tensor &e1_, const Tensor &v1_,
    const Tensor &b2_, const Tensor &l2_, const Tensor &e2_, const Tensor &v2_,
    int64_t B, int64_t L, int64_t E, bool cache
) {
    const Tensor b0 = b0_.contiguous();
    const Tensor l0 = l0_.contiguous();
    const Tensor e0 = e0_.contiguous();
    const Tensor v0 = v0_.contiguous().to(at::kFloat);
    const Tensor b1 = b1_.contiguous();
    const Tensor l1 = l1_.contiguous();
    const Tensor e1 = e1_.contiguous();
    const Tensor v1 = v1_.contiguous().to(at::kFloat);
    const Tensor b2 = b2_.contiguous();
    const Tensor l2 = l2_.contiguous();
    const Tensor e2 = e2_.contiguous();
    const Tensor v2 = v2_.contiguous().to(at::kFloat);

    TORCH_CHECK(v0.is_cuda() || v1.is_cuda() || v2.is_cuda(), "at least one input must be CUDA");
    int64_t n0 = v0.size(0);
    int64_t n1 = v1.size(0);
    int64_t n2 = v2.size(0);
    int64_t total = n0 + n1 + n2;

    auto opts_long = (n0 > 0 ? b0.options() : (n1 > 0 ? b1.options() : b2.options())).dtype(at::kLong);
    auto opts_val = (n0 > 0 ? v0.options() : (n1 > 0 ? v1.options() : v2.options())).dtype(at::kFloat);
    if (total == 0) {
        auto empty_i = at::empty({0}, opts_long);
        auto empty_v = at::empty({0}, opts_val);
        return std::make_tuple(empty_i, empty_i, empty_i, empty_v, empty_i.to(at::kInt));
    }

    auto keys = at::empty({total}, opts_long);
    auto values = at::empty({total}, opts_val);
    int threads = 256;
    auto append = [&](const Tensor &b, const Tensor &l, const Tensor &e, const Tensor &v, int64_t offset, int64_t n) {
        if (n <= 0) return;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        append_sparse3d_sum_keys_kernel<float><<<blocks, threads>>>(
            b.data_ptr<int64_t>(), l.data_ptr<int64_t>(), e.data_ptr<int64_t>(), v.data_ptr<float>(),
            keys.data_ptr<int64_t>(), values.data_ptr<float>(), offset, n, L, E
        );
    };
    append(b0, l0, e0, v0, 0, n0);
    append(b1, l1, e1, v1, n0, n1);
    append(b2, l2, e2, v2, n0 + n1, n2);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<int64_t> key_ptr(keys.data_ptr<int64_t>());
    thrust::device_ptr<float> val_ptr(values.data_ptr<float>());
    Tensor permutation;
    if (cache) {
        permutation = at::arange(total, opts_long);
        thrust::device_ptr<int64_t> perm_ptr(permutation.data_ptr<int64_t>());
        thrust::sort_by_key(policy, key_ptr, key_ptr + total, perm_ptr);
        values = values.index_select(0, permutation);
        val_ptr = thrust::device_ptr<float>(values.data_ptr<float>());
    } else {
        thrust::sort_by_key(policy, key_ptr, key_ptr + total, val_ptr);
    }

    auto red_keys = at::empty({total}, opts_long);
    auto red_values = at::empty({total}, opts_val);
    thrust::device_ptr<int64_t> red_key_ptr(red_keys.data_ptr<int64_t>());
    thrust::device_ptr<float> red_val_ptr(red_values.data_ptr<float>());
    auto new_end = thrust::reduce_by_key(policy, key_ptr, key_ptr + total, val_ptr, red_key_ptr, red_val_ptr);
    int64_t out_n = new_end.first - red_key_ptr;

    auto out_b = at::empty({out_n}, opts_long);
    auto out_l = at::empty({out_n}, opts_long);
    auto out_e = at::empty({out_n}, opts_long);
    auto out_v = at::empty({out_n}, opts_val);
    if (out_n > 0) {
        int blocks = static_cast<int>((out_n + threads - 1) / threads);
        unpack_sparse3d_sum_keys_kernel<float><<<blocks, threads>>>(
            red_keys.data_ptr<int64_t>(), red_values.data_ptr<float>(),
            out_b.data_ptr<int64_t>(), out_l.data_ptr<int64_t>(), out_e.data_ptr<int64_t>(), out_v.data_ptr<float>(),
            out_n, L, E
        );
    }
    auto positions = at::empty({cache ? total : 0}, opts_long.dtype(out_n < INT32_MAX ? at::kInt : at::kLong));
    if (cache) {
        auto flags = at::ones({total}, opts_long);
        if (total > 1) flags.slice(0, 1).copy_(keys.slice(0, 1).ne(keys.slice(0, 0, total - 1)));
        auto segments = flags.cumsum(0) - 1;
        positions.index_copy_(0, permutation, segments.to(positions.scalar_type()));
    }
    return std::make_tuple(out_b, out_l, out_e, out_v, positions);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
sparse3d_coalesce_sum3_cuda(
    const Tensor &b0, const Tensor &l0, const Tensor &e0, const Tensor &v0,
    const Tensor &b1, const Tensor &l1, const Tensor &e1, const Tensor &v1,
    const Tensor &b2, const Tensor &l2, const Tensor &e2, const Tensor &v2,
    int64_t B, int64_t L, int64_t E
) {
    auto result = sparse3d_coalesce_sum3_impl(b0,l0,e0,v0,b1,l1,e1,v1,b2,l2,e2,v2,B,L,E,false);
    return std::make_tuple(std::get<0>(result),std::get<1>(result),std::get<2>(result),std::get<3>(result));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sparse3d_coalesce_sum3_cached_cuda(
    const Tensor &b0, const Tensor &l0, const Tensor &e0, const Tensor &v0,
    const Tensor &b1, const Tensor &l1, const Tensor &e1, const Tensor &v1,
    const Tensor &b2, const Tensor &l2, const Tensor &e2, const Tensor &v2,
    int64_t B, int64_t L, int64_t E
) {
    return sparse3d_coalesce_sum3_impl(b0,l0,e0,v0,b1,l1,e1,v1,b2,l2,e2,v2,B,L,E,true);
}

} // namespace flashlog
