#include "flashlog.h"
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace flashlog {

// =========================================================================
// Bucket-based top-k helpers (aligned with CUDA forward_topk_row)
// =========================================================================
constexpr int NUM_BUCKETS = 256;

template <class scalar_t>
static inline int bucketize_weight_cpu(scalar_t score) {
    int bucket = static_cast<int>(score * static_cast<scalar_t>(NUM_BUCKETS - 1));
    return std::min(std::max(bucket, 0), NUM_BUCKETS - 1);
}

// Process one direction for a single (active_node, bl) pair.
// Mirrors CUDA forward_topk_row exactly: histogram → threshold → prune.
template <class scalar_t>
static void forward_topk_row_cpu(
    int64_t src, int64_t bl, scalar_t a_val,
    const scalar_t *w,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    const int32_t *col_ind,
    const scalar_t *mask,
    scalar_t *out,
    int64_t topk_edges,
    int64_t E, int64_t n, int64_t rel_offset
) {
    int64_t g_start = row_group_ptr[src];
    int64_t g_end   = row_group_ptr[src + 1];

    // Phase 1: Build histogram of edge counts per weight bucket
    int counts[NUM_BUCKETS] = {};
    int64_t total_edges = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight_cpu(score);
        int32_t cnt = group_edge_count[g];
        counts[bucket] += cnt;
        total_edges += cnt;
    }

    // Phase 2: Find threshold bucket (greedy from high to low)
    int threshold_bucket = -1;
    int64_t keep_in_threshold = 0;
    if (total_edges > topk_edges) {
        int64_t cumsum = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
            if (cumsum + counts[b] >= topk_edges) {
                threshold_bucket = b;
                keep_in_threshold = topk_edges - cumsum;
                break;
            }
            cumsum += counts[b];
        }
    }

    // Phase 3: Iterate groups with pruning
    int64_t threshold_seen = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];

        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight_cpu(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t left = keep_in_threshold - threshold_seen;
                take = left > 0
                    ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(group_edge_count[g])))
                    : 0;
                threshold_seen += take;
            }
        }
        if (take <= 0) continue;

        int64_t edge_s = group_edge_start[g];
        for (int64_t p = edge_s; p < edge_s + take; ++p) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            out[bl * E + dst] += a_val * w_val * m;
        }
    }
}

// Backward helper: same bucket-based pruning to identify kept edges.
template <class scalar_t>
static void backward_topk_row_cpu(
    int64_t src, int64_t bl,
    const scalar_t *grad_out,
    const scalar_t *A, const scalar_t *w,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    const int32_t *col_ind,
    const scalar_t *mask,
    scalar_t *gA, scalar_t *gw,
    int64_t topk_edges,
    int64_t E, int64_t n, int64_t rel_offset
) {
    int64_t g_start = row_group_ptr[src];
    int64_t g_end   = row_group_ptr[src + 1];

    // Phase 1 + 2: identical bucketing / threshold as forward
    int counts[NUM_BUCKETS] = {};
    int64_t total_edges = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t score = w[bl * n + rel_offset + rel];
        int bucket = bucketize_weight_cpu(score);
        counts[bucket] += group_edge_count[g];
        total_edges += group_edge_count[g];
    }

    int threshold_bucket = -1;
    int64_t keep_in_threshold = 0;
    if (total_edges > topk_edges) {
        int64_t cumsum = 0;
        for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
            if (cumsum + counts[b] >= topk_edges) {
                threshold_bucket = b;
                keep_in_threshold = topk_edges - cumsum;
                break;
            }
            cumsum += counts[b];
        }
    }

    // Phase 3: gradient accumulation on kept edges only
    scalar_t a_val = A[bl * E + src];
    int64_t threshold_seen = 0;
    for (int64_t g = g_start; g < g_end; ++g) {
        int64_t rel = group_rel[g];
        scalar_t w_val = w[bl * n + rel_offset + rel];
        int32_t take = group_edge_count[g];

        if (threshold_bucket >= 0) {
            int bucket = bucketize_weight_cpu(w_val);
            if (bucket < threshold_bucket) {
                take = 0;
            } else if (bucket == threshold_bucket) {
                int64_t left = keep_in_threshold - threshold_seen;
                take = left > 0
                    ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(group_edge_count[g])))
                    : 0;
                threshold_seen += take;
            }
        }
        if (take <= 0) continue;

        int64_t edge_s = group_edge_start[g];
        for (int64_t p = edge_s; p < edge_s + take; ++p) {
            scalar_t m = mask[p];
            if (m == 0) continue;
            int64_t dst = col_ind[p];
            scalar_t go = grad_out[bl * E + dst];
            if (go == 0) continue;
            gA[bl * E + src] += go * w_val * m;
            gw[bl * n + rel_offset + rel] += go * a_val * m;
        }
    }
}

// =========================================================================
// Forward: group-based iteration with top-k pruning
// =========================================================================
template <class scalar_t>
void flashlog_forward_out_cpu(
    const scalar_t *A,              // [BL, E]
    const scalar_t *w,              // [BL, n]
    const int64_t  *active,         // [num_active]
    int64_t         num_active,
    const int32_t  *ori_col_ind,
    const scalar_t *ori_mask,
    const int32_t  *ori_row_group_ptr,
    const int16_t  *ori_group_rel,
    const int32_t  *ori_group_edge_start,
    const int32_t  *ori_group_edge_count,
    const int32_t  *inv_col_ind,
    const scalar_t *inv_mask,
    const int32_t  *inv_row_group_ptr,
    const int16_t  *inv_group_rel,
    const int32_t  *inv_group_edge_start,
    const int32_t  *inv_group_edge_count,
    scalar_t *out_ind,
    scalar_t *out_ori,
    scalar_t *out_inv,
    int64_t BL, int64_t E, int64_t n,
    int64_t r_size, bool wot_i,
    int64_t topk_edges
) {
    int64_t BLE = BL * E;
    std::memset(out_ori, 0, BLE * sizeof(scalar_t));
    std::memset(out_inv, 0, BLE * sizeof(scalar_t));

    // Identity: out_ind[bl,e] = A[bl,e] * w[bl,-1]
    if (!wot_i) {
        for (int64_t bl = 0; bl < BL; bl++) {
            scalar_t wid = w[bl * n + (n - 1)];
            const scalar_t *Abl = A + bl * E;
            scalar_t *obl = out_ind + bl * E;
            for (int64_t e = 0; e < E; e++)
                obl[e] = Abl[e] * wid;
        }
    } else {
        std::memset(out_ind, 0, BLE * sizeof(scalar_t));
    }

    // ORI + INV: group-based iteration with top-k pruning
    for (int64_t ai = 0; ai < num_active; ++ai) {
        int64_t src = active[ai];
        for (int64_t bl = 0; bl < BL; ++bl) {
            scalar_t a_val = A[bl * E + src];
            if (a_val == 0) continue;
            forward_topk_row_cpu(src, bl, a_val, w,
                ori_row_group_ptr, ori_group_rel,
                ori_group_edge_start, ori_group_edge_count,
                ori_col_ind, ori_mask, out_ori,
                topk_edges, E, n, 0);
            forward_topk_row_cpu(src, bl, a_val, w,
                inv_row_group_ptr, inv_group_rel,
                inv_group_edge_start, inv_group_edge_count,
                inv_col_ind, inv_mask, out_inv,
                topk_edges, E, n, r_size);
        }
    }
}

// =========================================================================
// Backward: group-based iteration with top-k pruning
// =========================================================================
template <class scalar_t>
void flashlog_backward_out_cpu(
    const scalar_t *grad_ind,
    const scalar_t *grad_ori,
    const scalar_t *grad_inv,
    const scalar_t *A,
    const scalar_t *w,
    const int64_t  *active,
    int64_t         num_active,
    const int32_t  *ori_col_ind,
    const scalar_t *ori_mask,
    const int32_t  *ori_row_group_ptr,
    const int16_t  *ori_group_rel,
    const int32_t  *ori_group_edge_start,
    const int32_t  *ori_group_edge_count,
    const int32_t  *inv_col_ind,
    const scalar_t *inv_mask,
    const int32_t  *inv_row_group_ptr,
    const int16_t  *inv_group_rel,
    const int32_t  *inv_group_edge_start,
    const int32_t  *inv_group_edge_count,
    scalar_t *gA, scalar_t *gw,
    int64_t BL, int64_t E, int64_t n,
    int64_t r_size, bool wot_i,
    int64_t topk_edges
) {
    int64_t BLE = BL * E;
    int64_t BLn = BL * n;
    std::memset(gA, 0, BLE * sizeof(scalar_t));
    std::memset(gw, 0, BLn * sizeof(scalar_t));

    // Identity backward
    if (!wot_i) {
        for (int64_t bl = 0; bl < BL; bl++) {
            scalar_t wid = w[bl * n + (n - 1)];
            scalar_t gw_acc = 0;
            for (int64_t e = 0; e < E; e++) {
                scalar_t gi = grad_ind[bl * E + e];
                gA[bl * E + e] += gi * wid;
                gw_acc += gi * A[bl * E + e];
            }
            gw[bl * n + (n - 1)] += gw_acc;
        }
    }

    // ORI + INV backward with top-k pruning
    for (int64_t ai = 0; ai < num_active; ++ai) {
        int64_t src = active[ai];
        for (int64_t bl = 0; bl < BL; ++bl) {
            if (A[bl * E + src] == 0) continue;
            backward_topk_row_cpu(src, bl, grad_ori, A, w,
                ori_row_group_ptr, ori_group_rel,
                ori_group_edge_start, ori_group_edge_count,
                ori_col_ind, ori_mask, gA, gw,
                topk_edges, E, n, 0);
            backward_topk_row_cpu(src, bl, grad_inv, A, w,
                inv_row_group_ptr, inv_group_rel,
                inv_group_edge_start, inv_group_edge_count,
                inv_col_ind, inv_mask, gA, gw,
                topk_edges, E, n, r_size);
        }
    }
}

// =========================================================================
// Public API
// =========================================================================

std::tuple<Tensor, Tensor, Tensor> flashlog_forward_cpu(
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_col_ind_,   const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_,   const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, int64_t topk_edges
) {
    TORCH_CHECK(A_.dim() == 2, "A must be 2D [BL, E]");
    TORCH_CHECK(w_.dim() == 2, "w must be 2D [BL, n]");

    const Tensor A    = A_.contiguous();
    const Tensor w    = w_.contiguous();
    const Tensor act  = active_nodes_.to(at::kLong).contiguous();
    const Tensor oci  = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om   = ori_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr  = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs  = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc  = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici  = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im   = inv_mask_.to(A.dtype()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr  = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs  = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc  = inv_group_edge_count_.to(at::kInt).contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    Tensor out_ind = at::empty({BL, E}, A.options());
    Tensor out_ori = at::empty({BL, E}, A.options());
    Tensor out_inv = at::empty({BL, E}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_forward_cpu", [&] {
        flashlog_forward_out_cpu<scalar_t>(
            A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            act.data_ptr<int64_t>(), act.size(0),
            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            out_ind.data_ptr<scalar_t>(),
            out_ori.data_ptr<scalar_t>(),
            out_inv.data_ptr<scalar_t>(),
            BL, E, n, r_size, wot_i, topk_edges
        );
    });

    return std::make_tuple(out_ind, out_ori, out_inv);
}

std::tuple<Tensor, Tensor> flashlog_backward_cpu(
    const Tensor &grad_ind_, const Tensor &grad_ori_, const Tensor &grad_inv_,
    const Tensor &A_, const Tensor &w_,
    const Tensor &active_nodes_,
    const Tensor &ori_col_ind_,   const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_,   const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t r_size, bool wot_i, int64_t topk_edges
) {
    const Tensor A   = A_.contiguous();
    const Tensor w   = w_.contiguous();
    const Tensor grad_ind = wot_i ? at::zeros_like(A) : grad_ind_.contiguous();
    const Tensor grad_ori = grad_ori_.contiguous();
    const Tensor grad_inv = grad_inv_.contiguous();
    const Tensor act  = active_nodes_.to(at::kLong).contiguous();
    const Tensor oci  = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om   = ori_mask_.to(A.dtype()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr  = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs  = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc  = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici  = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im   = inv_mask_.to(A.dtype()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr  = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs  = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc  = inv_group_edge_count_.to(at::kInt).contiguous();

    int64_t BL = A.size(0), E = A.size(1), n = w.size(1);
    Tensor gA = at::empty({BL, E}, A.options());
    Tensor gw = at::empty({BL, n}, w.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "flashlog_backward_cpu", [&] {
        flashlog_backward_out_cpu<scalar_t>(
            grad_ind.data_ptr<scalar_t>(),
            grad_ori.data_ptr<scalar_t>(),
            grad_inv.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            act.data_ptr<int64_t>(), act.size(0),
            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
            gA.data_ptr<scalar_t>(), gw.data_ptr<scalar_t>(),
            BL, E, n, r_size, wot_i, topk_edges
        );
    });

    return std::make_tuple(gA, gw);
}

template <class scalar_t>
static void append_sparse3d_sum_cpu(
    const int64_t *b, const int64_t *l, const int64_t *e, const scalar_t *v,
    int64_t n, int64_t L, int64_t E,
    std::vector<std::pair<int64_t, scalar_t>> &items
) {
    items.reserve(items.size() + static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        int64_t key = b[i] * (L * E) + l[i] * E + e[i];
        items.emplace_back(key, v[i]);
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> sparse3d_coalesce_sum3_cpu(
    const Tensor &b0_, const Tensor &l0_, const Tensor &e0_, const Tensor &v0_,
    const Tensor &b1_, const Tensor &l1_, const Tensor &e1_, const Tensor &v1_,
    const Tensor &b2_, const Tensor &l2_, const Tensor &e2_, const Tensor &v2_,
    int64_t B, int64_t L, int64_t E
) {
    TORCH_CHECK(!v0_.is_cuda() && !v1_.is_cuda() && !v2_.is_cuda(), "CPU coalesce expects CPU tensors");
    (void)B;

    const Tensor b0 = b0_.to(at::kLong).contiguous();
    const Tensor l0 = l0_.to(at::kLong).contiguous();
    const Tensor e0 = e0_.to(at::kLong).contiguous();
    const Tensor v0 = v0_.contiguous();
    const Tensor b1 = b1_.to(at::kLong).contiguous();
    const Tensor l1 = l1_.to(at::kLong).contiguous();
    const Tensor e1 = e1_.to(at::kLong).contiguous();
    const Tensor v1 = v1_.to(v0.scalar_type()).contiguous();
    const Tensor b2 = b2_.to(at::kLong).contiguous();
    const Tensor l2 = l2_.to(at::kLong).contiguous();
    const Tensor e2 = e2_.to(at::kLong).contiguous();
    const Tensor v2 = v2_.to(v0.scalar_type()).contiguous();

    int64_t n0 = v0.size(0);
    int64_t n1 = v1.size(0);
    int64_t n2 = v2.size(0);
    int64_t total = n0 + n1 + n2;
    auto opts_long = v0.options().dtype(at::kLong);
    auto opts_val = v0.options();
    if (total == 0) {
        auto empty_i = at::empty({0}, opts_long);
        auto empty_v = at::empty({0}, opts_val);
        return std::make_tuple(empty_i, empty_i, empty_i, empty_v);
    }

    Tensor out_b;
    Tensor out_l;
    Tensor out_e;
    Tensor out_v;
    AT_DISPATCH_FLOATING_TYPES(v0.scalar_type(), "sparse3d_coalesce_sum3_cpu", [&] {
        std::vector<std::pair<int64_t, scalar_t>> items;
        append_sparse3d_sum_cpu<scalar_t>(
            b0.data_ptr<int64_t>(), l0.data_ptr<int64_t>(), e0.data_ptr<int64_t>(), v0.data_ptr<scalar_t>(),
            n0, L, E, items);
        append_sparse3d_sum_cpu<scalar_t>(
            b1.data_ptr<int64_t>(), l1.data_ptr<int64_t>(), e1.data_ptr<int64_t>(), v1.data_ptr<scalar_t>(),
            n1, L, E, items);
        append_sparse3d_sum_cpu<scalar_t>(
            b2.data_ptr<int64_t>(), l2.data_ptr<int64_t>(), e2.data_ptr<int64_t>(), v2.data_ptr<scalar_t>(),
            n2, L, E, items);

        std::sort(items.begin(), items.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        std::vector<int64_t> keys;
        std::vector<scalar_t> values;
        keys.reserve(items.size());
        values.reserve(items.size());
        for (const auto &item : items) {
            if (!keys.empty() && keys.back() == item.first) {
                values.back() += item.second;
            } else {
                keys.push_back(item.first);
                values.push_back(item.second);
            }
        }

        int64_t out_n = static_cast<int64_t>(keys.size());
        out_b = at::empty({out_n}, opts_long);
        out_l = at::empty({out_n}, opts_long);
        out_e = at::empty({out_n}, opts_long);
        out_v = at::empty({out_n}, opts_val);
        auto *ob = out_b.data_ptr<int64_t>();
        auto *ol = out_l.data_ptr<int64_t>();
        auto *oe = out_e.data_ptr<int64_t>();
        auto *ov = out_v.data_ptr<scalar_t>();
        int64_t LE = L * E;
        for (int64_t i = 0; i < out_n; ++i) {
            int64_t key = keys[i];
            ob[i] = key / LE;
            int64_t rem = key % LE;
            ol[i] = rem / E;
            oe[i] = rem % E;
            ov[i] = values[i];
        }
    });
    return std::make_tuple(out_b, out_l, out_e, out_v);
}

template <class scalar_t>
struct Sparse3DRawCpu {
    std::vector<int64_t> b;
    std::vector<int64_t> l;
    std::vector<int64_t> e;
    std::vector<int64_t> entry;
    std::vector<int16_t> rel;
    std::vector<scalar_t> mask;
};

template <class scalar_t>
static void expand_sparse3d_direction_sum_cpu(
    const int64_t *sp_batch,
    const int64_t *sp_level,
    const int64_t *sp_entity,
    const scalar_t *sp_value,
    const scalar_t *w,
    int64_t nnz,
    const int32_t *col_ind,
    const scalar_t *mask,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    int64_t L,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges,
    Sparse3DRawCpu<scalar_t> &raw,
    std::vector<std::pair<int64_t, scalar_t>> &items
) {
    int counts[NUM_BUCKETS];
    for (int64_t entry = 0; entry < nnz; ++entry) {
        int64_t src = sp_entity[entry];
        int64_t g_start = row_group_ptr[src];
        int64_t g_end = row_group_ptr[src + 1];
        if (g_start == g_end) continue;

        int64_t b = sp_batch[entry];
        int64_t l = sp_level[entry];
        const scalar_t *w_bl = w + (b * L + l) * n;

        int threshold_bucket = -1;
        int64_t keep_in_threshold = 0;
        if (topk_edges < (1LL << 60)) {
            std::fill(std::begin(counts), std::end(counts), 0);
            int64_t total_edges = 0;
            for (int64_t g = g_start; g < g_end; ++g) {
                int64_t rel = group_rel[g];
                int bucket = bucketize_weight_cpu(w_bl[rel_offset + rel]);
                int32_t cnt = group_edge_count[g];
                counts[bucket] += cnt;
                total_edges += cnt;
            }
            if (total_edges > topk_edges) {
                int64_t cumsum = 0;
                for (int bucket = NUM_BUCKETS - 1; bucket >= 0; --bucket) {
                    if (cumsum + counts[bucket] >= topk_edges) {
                        threshold_bucket = bucket;
                        keep_in_threshold = topk_edges - cumsum;
                        break;
                    }
                    cumsum += counts[bucket];
                }
            }
        }

        int64_t threshold_seen = 0;
        for (int64_t g = g_start; g < g_end; ++g) {
            int64_t rel = group_rel[g];
            scalar_t w_val = w_bl[rel_offset + rel];
            int32_t take = group_edge_count[g];
            if (threshold_bucket >= 0) {
                int bucket = bucketize_weight_cpu(w_val);
                if (bucket < threshold_bucket) {
                    take = 0;
                } else if (bucket == threshold_bucket) {
                    int64_t left = keep_in_threshold - threshold_seen;
                    take = left > 0
                        ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(take)))
                        : 0;
                    threshold_seen += take;
                }
            }
            if (take <= 0) continue;

            int64_t edge_start = group_edge_start[g];
            for (int64_t p = edge_start; p < edge_start + take; ++p) {
                scalar_t m = mask[p];
                if (m == 0) continue;
                int64_t dst = col_ind[p];
                scalar_t value = sp_value[entry] * w_val * m;
                int64_t key = b * (L * E) + l * E + dst;
                items.emplace_back(key, value);
                raw.b.push_back(b);
                raw.l.push_back(l);
                raw.e.push_back(dst);
                raw.entry.push_back(entry);
                raw.rel.push_back(static_cast<int16_t>(rel));
                raw.mask.push_back(m);
            }
        }
    }
}

template <class scalar_t>
static void expand_sparse3d_direction_sum_cpu_range(
    const int64_t *sp_batch,
    const int64_t *sp_level,
    const int64_t *sp_entity,
    const scalar_t *sp_value,
    const scalar_t *w,
    int64_t begin,
    int64_t end,
    const int32_t *col_ind,
    const scalar_t *mask,
    const int32_t *row_group_ptr,
    const int16_t *group_rel,
    const int32_t *group_edge_start,
    const int32_t *group_edge_count,
    int64_t L,
    int64_t E,
    int64_t n,
    int64_t rel_offset,
    int64_t topk_edges,
    Sparse3DRawCpu<scalar_t> &raw,
    std::vector<std::pair<int64_t, scalar_t>> &items
) {
    int counts[NUM_BUCKETS];
    for (int64_t entry = begin; entry < end; ++entry) {
        int64_t src = sp_entity[entry];
        int64_t g_start = row_group_ptr[src];
        int64_t g_end = row_group_ptr[src + 1];
        if (g_start == g_end) continue;

        int64_t b = sp_batch[entry];
        int64_t l = sp_level[entry];
        const scalar_t *w_bl = w + (b * L + l) * n;

        int threshold_bucket = -1;
        int64_t keep_in_threshold = 0;
        if (topk_edges < (1LL << 60)) {
            std::fill(std::begin(counts), std::end(counts), 0);
            int64_t total_edges = 0;
            for (int64_t g = g_start; g < g_end; ++g) {
                int64_t rel = group_rel[g];
                int bucket = bucketize_weight_cpu(w_bl[rel_offset + rel]);
                int32_t cnt = group_edge_count[g];
                counts[bucket] += cnt;
                total_edges += cnt;
            }
            if (total_edges > topk_edges) {
                int64_t cumsum = 0;
                for (int bucket = NUM_BUCKETS - 1; bucket >= 0; --bucket) {
                    if (cumsum + counts[bucket] >= topk_edges) {
                        threshold_bucket = bucket;
                        keep_in_threshold = topk_edges - cumsum;
                        break;
                    }
                    cumsum += counts[bucket];
                }
            }
        }

        int64_t threshold_seen = 0;
        for (int64_t g = g_start; g < g_end; ++g) {
            int64_t rel = group_rel[g];
            scalar_t w_val = w_bl[rel_offset + rel];
            int32_t take = group_edge_count[g];
            if (threshold_bucket >= 0) {
                int bucket = bucketize_weight_cpu(w_val);
                if (bucket < threshold_bucket) {
                    take = 0;
                } else if (bucket == threshold_bucket) {
                    int64_t left = keep_in_threshold - threshold_seen;
                    take = left > 0
                        ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(take)))
                        : 0;
                    threshold_seen += take;
                }
            }
            if (take <= 0) continue;

            int64_t edge_start = group_edge_start[g];
            for (int64_t p = edge_start; p < edge_start + take; ++p) {
                scalar_t m = mask[p];
                if (m == 0) continue;
                int64_t dst = col_ind[p];
                scalar_t value = sp_value[entry] * w_val * m;
                int64_t key = b * (L * E) + l * E + dst;
                items.emplace_back(key, value);
                raw.b.push_back(b);
                raw.l.push_back(l);
                raw.e.push_back(dst);
                raw.entry.push_back(entry);
                raw.rel.push_back(static_cast<int16_t>(rel));
                raw.mask.push_back(m);
            }
        }
    }
}

template <class scalar_t>
static void append_identity_sum_cpu_range(
    const int64_t *sp_batch,
    const int64_t *sp_level,
    const int64_t *sp_entity,
    const scalar_t *sp_value,
    const scalar_t *w,
    int64_t begin,
    int64_t end,
    int64_t L,
    int64_t E,
    int64_t n,
    std::vector<std::pair<int64_t, scalar_t>> &items
) {
    for (int64_t i = begin; i < end; ++i) {
        int64_t b = sp_batch[i];
        int64_t l = sp_level[i];
        int64_t e = sp_entity[i];
        scalar_t value = sp_value[i] * w[(b * L + l) * n + (n - 1)];
        int64_t key = b * (L * E) + l * E + e;
        items.emplace_back(key, value);
    }
}

template <class scalar_t>
static Tensor tensor_from_i64_vector(const std::vector<int64_t> &values, const TensorOptions &opts) {
    Tensor out = at::empty({static_cast<int64_t>(values.size())}, opts.dtype(at::kLong));
    if (!values.empty()) {
        std::memcpy(out.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
    }
    return out;
}

template <class scalar_t>
static Tensor tensor_from_i16_vector(const std::vector<int16_t> &values, const TensorOptions &opts) {
    Tensor out = at::empty({static_cast<int64_t>(values.size())}, opts.dtype(at::kShort));
    if (!values.empty()) {
        std::memcpy(out.data_ptr<int16_t>(), values.data(), values.size() * sizeof(int16_t));
    }
    return out;
}

template <class scalar_t>
static Tensor tensor_from_scalar_vector(const std::vector<scalar_t> &values, const TensorOptions &opts) {
    Tensor out = at::empty({static_cast<int64_t>(values.size())}, opts);
    if (!values.empty()) {
        std::memcpy(out.data_ptr<scalar_t>(), values.data(), values.size() * sizeof(scalar_t));
    }
    return out;
}

std::tuple<Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor,Tensor,Tensor,Tensor>
flashlog_forward_sparse3d_sum_cpu(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges
) {
    TORCH_CHECK(!sp_value_.is_cuda(), "CPU sparse3d sum expects CPU tensors");
    const Tensor sp_batch = sp_batch_.to(at::kLong).contiguous();
    const Tensor sp_level = sp_level_.to(at::kLong).contiguous();
    const Tensor sp_entity = sp_entity_.to(at::kLong).contiguous();
    const Tensor sp_value = sp_value_.contiguous();
    const Tensor w = w_.to(sp_value.scalar_type()).contiguous();
    const Tensor oci = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om = ori_mask_.to(sp_value.scalar_type()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im = inv_mask_.to(sp_value.scalar_type()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc = inv_group_edge_count_.to(at::kInt).contiguous();

    Tensor out_b, out_l, out_e, out_v;
    Tensor ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m;
    Tensor inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m;

    AT_DISPATCH_FLOATING_TYPES(sp_value.scalar_type(), "flashlog_forward_sparse3d_sum_cpu", [&] {
        int64_t nnz = sp_batch.size(0);
        int64_t n = w.size(2);
        int nt = std::max<int>(1, at::get_num_threads());
        bool use_rowwise_dense = (E * static_cast<int64_t>(nt) <= 20000000LL);
        std::vector<Sparse3DRawCpu<scalar_t>> ori_parts(nt);
        std::vector<Sparse3DRawCpu<scalar_t>> inv_parts(nt);

        Sparse3DRawCpu<scalar_t> ori_raw;
        Sparse3DRawCpu<scalar_t> inv_raw;
        auto merge_raw = [](Sparse3DRawCpu<scalar_t> &dst, Sparse3DRawCpu<scalar_t> &src) {
            dst.b.insert(dst.b.end(), src.b.begin(), src.b.end());
            dst.l.insert(dst.l.end(), src.l.begin(), src.l.end());
            dst.e.insert(dst.e.end(), src.e.begin(), src.e.end());
            dst.entry.insert(dst.entry.end(), src.entry.begin(), src.entry.end());
            dst.rel.insert(dst.rel.end(), src.rel.begin(), src.rel.end());
            dst.mask.insert(dst.mask.end(), src.mask.begin(), src.mask.end());
        };

        auto opts_long = sp_batch.options().dtype(at::kLong);
        auto opts_val = sp_value.options();

        if (use_rowwise_dense) {
            std::vector<std::vector<int64_t>> row_entries(static_cast<size_t>(B * L));
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = sp_batch.data_ptr<int64_t>()[i] * L + sp_level.data_ptr<int64_t>()[i];
                row_entries[static_cast<size_t>(row)].push_back(i);
            }

            std::vector<std::vector<scalar_t>> acc_parts(nt, std::vector<scalar_t>(static_cast<size_t>(E), scalar_t(0)));
            std::vector<std::vector<int64_t>> mark_parts(nt, std::vector<int64_t>(static_cast<size_t>(E), int64_t(-1)));
            std::vector<std::vector<int64_t>> out_b_parts(nt), out_l_parts(nt), out_e_parts(nt);
            std::vector<std::vector<scalar_t>> out_v_parts(nt);

            auto add_value = [&](int tid, int64_t stamp, int64_t dst, scalar_t value, std::vector<int64_t> &touched) {
                auto &mark = mark_parts[tid];
                auto &acc = acc_parts[tid];
                if (mark[static_cast<size_t>(dst)] != stamp) {
                    mark[static_cast<size_t>(dst)] = stamp;
                    acc[static_cast<size_t>(dst)] = scalar_t(0);
                    touched.push_back(dst);
                }
                acc[static_cast<size_t>(dst)] += value;
            };

            auto process_direction = [&](int tid, int64_t entry, int64_t rel_offset,
                                         const int32_t *col_ptr, const scalar_t *mask_ptr,
                                         const int32_t *rgp, const int16_t *grel,
                                         const int32_t *gestart, const int32_t *gecount,
                                         Sparse3DRawCpu<scalar_t> &raw,
                                         std::vector<int64_t> &touched) {
                int64_t src = sp_entity.data_ptr<int64_t>()[entry];
                int64_t g_start = rgp[src];
                int64_t g_end = rgp[src + 1];
                if (g_start == g_end) return;

                int64_t b = sp_batch.data_ptr<int64_t>()[entry];
                int64_t l = sp_level.data_ptr<int64_t>()[entry];
                int64_t stamp = b * L + l;
                const scalar_t *w_bl = w.data_ptr<scalar_t>() + stamp * n;

                int counts[NUM_BUCKETS] = {};
                int threshold_bucket = -1;
                int64_t keep_in_threshold = 0;
                if (topk_edges < (1LL << 60)) {
                    int64_t total_edges = 0;
                    for (int64_t g = g_start; g < g_end; ++g) {
                        int64_t rel = grel[g];
                        int bucket = bucketize_weight_cpu(w_bl[rel_offset + rel]);
                        int32_t cnt = gecount[g];
                        counts[bucket] += cnt;
                        total_edges += cnt;
                    }
                    if (total_edges > topk_edges) {
                        int64_t cumsum = 0;
                        for (int bucket = NUM_BUCKETS - 1; bucket >= 0; --bucket) {
                            if (cumsum + counts[bucket] >= topk_edges) {
                                threshold_bucket = bucket;
                                keep_in_threshold = topk_edges - cumsum;
                                break;
                            }
                            cumsum += counts[bucket];
                        }
                    }
                }

                int64_t threshold_seen = 0;
                for (int64_t g = g_start; g < g_end; ++g) {
                    int64_t rel = grel[g];
                    scalar_t w_val = w_bl[rel_offset + rel];
                    int32_t take = gecount[g];
                    if (threshold_bucket >= 0) {
                        int bucket = bucketize_weight_cpu(w_val);
                        if (bucket < threshold_bucket) {
                            take = 0;
                        } else if (bucket == threshold_bucket) {
                            int64_t left = keep_in_threshold - threshold_seen;
                            take = left > 0
                                ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(take)))
                                : 0;
                            threshold_seen += take;
                        }
                    }
                    if (take <= 0) continue;
                    int64_t edge_start = gestart[g];
                    for (int64_t p = edge_start; p < edge_start + take; ++p) {
                        scalar_t m = mask_ptr[p];
                        if (m == 0) continue;
                        int64_t dst = col_ptr[p];
                        scalar_t value = sp_value.data_ptr<scalar_t>()[entry] * w_val * m;
                        add_value(tid, stamp, dst, value, touched);
                        raw.b.push_back(b);
                        raw.l.push_back(l);
                        raw.e.push_back(dst);
                        raw.entry.push_back(entry);
                        raw.rel.push_back(static_cast<int16_t>(rel));
                        raw.mask.push_back(m);
                    }
                }
            };

            at::parallel_for(0, B * L, 1, [&](int64_t row_begin, int64_t row_end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                std::vector<int64_t> touched_row;
                for (int64_t row = row_begin; row < row_end; ++row) {
                    const auto &entries = row_entries[static_cast<size_t>(row)];
                    if (entries.empty()) continue;
                    touched_row.clear();
                    for (int64_t entry : entries) {
                        process_direction(tid, entry, 0,
                            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                            ori_parts[tid], touched_row);
                        process_direction(tid, entry, r_size,
                            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                            inv_parts[tid], touched_row);
                        if (!wot_i) {
                            int64_t dst = sp_entity.data_ptr<int64_t>()[entry];
                            scalar_t value = sp_value.data_ptr<scalar_t>()[entry] *
                                w.data_ptr<scalar_t>()[row * n + (n - 1)];
                            add_value(tid, row, dst, value, touched_row);
                        }
                    }
                    std::sort(touched_row.begin(), touched_row.end());
                    int64_t b = row / L;
                    int64_t l = row % L;
                    auto &ob = out_b_parts[tid];
                    auto &ol = out_l_parts[tid];
                    auto &oe = out_e_parts[tid];
                    auto &ov = out_v_parts[tid];
                    for (int64_t ent : touched_row) {
                        ob.push_back(b);
                        ol.push_back(l);
                        oe.push_back(ent);
                        ov.push_back(acc_parts[tid][static_cast<size_t>(ent)]);
                    }
                }
            });

            int64_t out_n = 0;
            for (int t = 0; t < nt; ++t) out_n += static_cast<int64_t>(out_v_parts[t].size());
            out_b = at::empty({out_n}, opts_long);
            out_l = at::empty({out_n}, opts_long);
            out_e = at::empty({out_n}, opts_long);
            out_v = at::empty({out_n}, opts_val);
            int64_t pos = 0;
            for (int t = 0; t < nt; ++t) {
                int64_t nout = static_cast<int64_t>(out_v_parts[t].size());
                if (nout > 0) {
                    std::memcpy(out_b.data_ptr<int64_t>() + pos, out_b_parts[t].data(), nout * sizeof(int64_t));
                    std::memcpy(out_l.data_ptr<int64_t>() + pos, out_l_parts[t].data(), nout * sizeof(int64_t));
                    std::memcpy(out_e.data_ptr<int64_t>() + pos, out_e_parts[t].data(), nout * sizeof(int64_t));
                    std::memcpy(out_v.data_ptr<scalar_t>() + pos, out_v_parts[t].data(), nout * sizeof(scalar_t));
                    pos += nout;
                }
                merge_raw(ori_raw, ori_parts[t]);
                merge_raw(inv_raw, inv_parts[t]);
            }
        } else {
            std::vector<std::vector<std::pair<int64_t, scalar_t>>> item_parts(nt);
            at::parallel_for(0, nnz, 64, [&](int64_t begin, int64_t end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                auto &items = item_parts[tid];
                expand_sparse3d_direction_sum_cpu_range<scalar_t>(
                    sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                    sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), begin, end,
                    oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    L, E, n, 0, topk_edges, ori_parts[tid], items);
                expand_sparse3d_direction_sum_cpu_range<scalar_t>(
                    sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                    sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), begin, end,
                    ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    L, E, n, r_size, topk_edges, inv_parts[tid], items);
                if (!wot_i) {
                    append_identity_sum_cpu_range<scalar_t>(
                        sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                        sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                        begin, end, L, E, n, items);
                }
            });

            std::vector<std::unordered_map<int64_t, scalar_t>> row_maps(static_cast<size_t>(B * L));
            for (int t = 0; t < nt; ++t) {
                for (const auto &item : item_parts[t]) {
                    int64_t row = item.first / E;
                    int64_t ent = item.first % E;
                    row_maps[static_cast<size_t>(row)][ent] += item.second;
                }
                merge_raw(ori_raw, ori_parts[t]);
                merge_raw(inv_raw, inv_parts[t]);
            }

            int64_t out_n = 0;
            for (const auto &row_map : row_maps) {
                out_n += static_cast<int64_t>(row_map.size());
            }
            out_b = at::empty({out_n}, opts_long);
            out_l = at::empty({out_n}, opts_long);
            out_e = at::empty({out_n}, opts_long);
            out_v = at::empty({out_n}, opts_val);
            int64_t pos = 0;
            for (int64_t row = 0; row < B * L; ++row) {
                auto &row_map = row_maps[static_cast<size_t>(row)];
                if (row_map.empty()) continue;
                std::vector<int64_t> ents;
                ents.reserve(row_map.size());
                for (const auto &kv : row_map) ents.push_back(kv.first);
                std::sort(ents.begin(), ents.end());
                int64_t b = row / L;
                int64_t l = row % L;
                for (int64_t ent : ents) {
                    out_b.data_ptr<int64_t>()[pos] = b;
                    out_l.data_ptr<int64_t>()[pos] = l;
                    out_e.data_ptr<int64_t>()[pos] = ent;
                    out_v.data_ptr<scalar_t>()[pos] = row_maps[static_cast<size_t>(row)][ent];
                    ++pos;
                }
            }
        }

        ori_b = tensor_from_i64_vector<scalar_t>(ori_raw.b, opts_long);
        ori_l = tensor_from_i64_vector<scalar_t>(ori_raw.l, opts_long);
        ori_e = tensor_from_i64_vector<scalar_t>(ori_raw.e, opts_long);
        ori_entry = tensor_from_i64_vector<scalar_t>(ori_raw.entry, opts_long);
        ori_rel = tensor_from_i16_vector<scalar_t>(ori_raw.rel, opts_long);
        ori_m = tensor_from_scalar_vector<scalar_t>(ori_raw.mask, opts_val);
        inv_b = tensor_from_i64_vector<scalar_t>(inv_raw.b, opts_long);
        inv_l = tensor_from_i64_vector<scalar_t>(inv_raw.l, opts_long);
        inv_e = tensor_from_i64_vector<scalar_t>(inv_raw.e, opts_long);
        inv_entry = tensor_from_i64_vector<scalar_t>(inv_raw.entry, opts_long);
        inv_rel = tensor_from_i16_vector<scalar_t>(inv_raw.rel, opts_long);
        inv_m = tensor_from_scalar_vector<scalar_t>(inv_raw.mask, opts_val);
    });

    return std::make_tuple(
        out_b, out_l, out_e, out_v,
        ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m,
        inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m
    );
}

std::tuple<Tensor, Tensor> flashlog_backward_sparse3d_sum_cpu(
    const Tensor &grad_b_, const Tensor &grad_l_, const Tensor &grad_e_, const Tensor &grad_v_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_b_, const Tensor &ori_l_, const Tensor &ori_e_, const Tensor &ori_entry_,
    const Tensor &ori_rel_, const Tensor &ori_m_,
    const Tensor &inv_b_, const Tensor &inv_l_, const Tensor &inv_e_, const Tensor &inv_entry_,
    const Tensor &inv_rel_, const Tensor &inv_m_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i
) {
    TORCH_CHECK(!sp_value_.is_cuda(), "CPU sparse3d sum backward expects CPU tensors");
    const Tensor grad_b = grad_b_.to(at::kLong).contiguous();
    const Tensor grad_l = grad_l_.to(at::kLong).contiguous();
    const Tensor grad_e = grad_e_.to(at::kLong).contiguous();
    const Tensor grad_v = grad_v_.contiguous();
    const Tensor sp_batch = sp_batch_.to(at::kLong).contiguous();
    const Tensor sp_level = sp_level_.to(at::kLong).contiguous();
    const Tensor sp_entity = sp_entity_.to(at::kLong).contiguous();
    const Tensor sp_value = sp_value_.to(grad_v.scalar_type()).contiguous();
    const Tensor w = w_.to(grad_v.scalar_type()).contiguous();
    const Tensor ori_b = ori_b_.to(at::kLong).contiguous();
    const Tensor ori_l = ori_l_.to(at::kLong).contiguous();
    const Tensor ori_e = ori_e_.to(at::kLong).contiguous();
    const Tensor ori_entry = ori_entry_.to(at::kLong).contiguous();
    const Tensor ori_rel = ori_rel_.to(at::kShort).contiguous();
    const Tensor ori_m = ori_m_.to(grad_v.scalar_type()).contiguous();
    const Tensor inv_b = inv_b_.to(at::kLong).contiguous();
    const Tensor inv_l = inv_l_.to(at::kLong).contiguous();
    const Tensor inv_e = inv_e_.to(at::kLong).contiguous();
    const Tensor inv_entry = inv_entry_.to(at::kLong).contiguous();
    const Tensor inv_rel = inv_rel_.to(at::kShort).contiguous();
    const Tensor inv_m = inv_m_.to(grad_v.scalar_type()).contiguous();

    Tensor grad_sp = at::zeros({sp_value.size(0)}, sp_value.options());
    Tensor grad_w = at::zeros_like(w);

    AT_DISPATCH_FLOATING_TYPES(sp_value.scalar_type(), "flashlog_backward_sparse3d_sum_cpu", [&] {
        int64_t total_slots = B * L * E;
        bool use_dense_grad = total_slots <= 50000000LL;
        std::vector<scalar_t> dense_grad;
        std::vector<std::pair<int64_t, scalar_t>> grad_items;
        if (use_dense_grad) {
            dense_grad.assign(static_cast<size_t>(total_slots), scalar_t(0));
            for (int64_t i = 0; i < grad_v.size(0); ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                dense_grad[static_cast<size_t>(key)] += grad_v.data_ptr<scalar_t>()[i];
            }
        } else {
            grad_items.reserve(static_cast<size_t>(grad_v.size(0)));
            for (int64_t i = 0; i < grad_v.size(0); ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                grad_items.emplace_back(key, grad_v.data_ptr<scalar_t>()[i]);
            }
            std::sort(grad_items.begin(), grad_items.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
        }
        auto grad_at = [&](int64_t b, int64_t l, int64_t e) -> scalar_t {
            int64_t key = b * (L * E) + l * E + e;
            if (use_dense_grad) {
                return dense_grad[static_cast<size_t>(key)];
            }
            auto it = std::lower_bound(
                grad_items.begin(), grad_items.end(), key,
                [](const auto &a, int64_t k) { return a.first < k; });
            return (it != grad_items.end() && it->first == key) ? it->second : scalar_t(0);
        };

        const scalar_t *spv = sp_value.data_ptr<scalar_t>();
        const scalar_t *wp = w.data_ptr<scalar_t>();
        int64_t n = w.size(2);
        int nt = std::max<int>(1, at::get_num_threads());
        int64_t sp_n = sp_value.size(0);
        int64_t w_n = w.numel();
        std::vector<std::vector<scalar_t>> local_sp(nt, std::vector<scalar_t>(static_cast<size_t>(sp_n), scalar_t(0)));
        std::vector<std::vector<scalar_t>> local_w(nt, std::vector<scalar_t>(static_cast<size_t>(w_n), scalar_t(0)));

        at::parallel_for(0, ori_b.size(0), 512, [&](int64_t begin, int64_t end) {
            int tid = at::get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            auto &gsp = local_sp[tid];
            auto &gw = local_w[tid];
            for (int64_t i = begin; i < end; ++i) {
                int64_t b = ori_b.data_ptr<int64_t>()[i];
                int64_t l = ori_l.data_ptr<int64_t>()[i];
                int64_t e = ori_e.data_ptr<int64_t>()[i];
                int64_t entry = ori_entry.data_ptr<int64_t>()[i];
                int64_t rel = ori_rel.data_ptr<int16_t>()[i];
                scalar_t g = grad_at(b, l, e);
                if (g == 0) continue;
                scalar_t m = ori_m.data_ptr<scalar_t>()[i];
                gsp[static_cast<size_t>(entry)] += g * wp[(b * L + l) * n + rel] * m;
                gw[static_cast<size_t>((b * L + l) * n + rel)] += g * spv[entry] * m;
            }
        });
        at::parallel_for(0, inv_b.size(0), 512, [&](int64_t begin, int64_t end) {
            int tid = at::get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            auto &gsp = local_sp[tid];
            auto &gw = local_w[tid];
            for (int64_t i = begin; i < end; ++i) {
                int64_t b = inv_b.data_ptr<int64_t>()[i];
                int64_t l = inv_l.data_ptr<int64_t>()[i];
                int64_t e = inv_e.data_ptr<int64_t>()[i];
                int64_t entry = inv_entry.data_ptr<int64_t>()[i];
                int64_t rel = static_cast<int64_t>(inv_rel.data_ptr<int16_t>()[i]) + r_size;
                scalar_t g = grad_at(b, l, e);
                if (g == 0) continue;
                scalar_t m = inv_m.data_ptr<scalar_t>()[i];
                gsp[static_cast<size_t>(entry)] += g * wp[(b * L + l) * n + rel] * m;
                gw[static_cast<size_t>((b * L + l) * n + rel)] += g * spv[entry] * m;
            }
        });
        if (!wot_i) {
            at::parallel_for(0, sp_value.size(0), 512, [&](int64_t begin, int64_t end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                auto &gsp = local_sp[tid];
                auto &gw = local_w[tid];
                for (int64_t i = begin; i < end; ++i) {
                    int64_t b = sp_batch.data_ptr<int64_t>()[i];
                    int64_t l = sp_level.data_ptr<int64_t>()[i];
                    int64_t e = sp_entity.data_ptr<int64_t>()[i];
                    int64_t rel = n - 1;
                    scalar_t g = grad_at(b, l, e);
                    if (g == 0) continue;
                    gsp[static_cast<size_t>(i)] += g * wp[(b * L + l) * n + rel];
                    gw[static_cast<size_t>((b * L + l) * n + rel)] += g * spv[i];
                }
            });
        }
        scalar_t *gsp_out = grad_sp.data_ptr<scalar_t>();
        scalar_t *gw_out = grad_w.data_ptr<scalar_t>();
        for (int t = 0; t < nt; ++t) {
            for (int64_t i = 0; i < sp_n; ++i) gsp_out[i] += local_sp[t][static_cast<size_t>(i)];
            for (int64_t i = 0; i < w_n; ++i) gw_out[i] += local_w[t][static_cast<size_t>(i)];
        }
    });

    return std::make_tuple(grad_sp, grad_w);
}

Tensor sparse3d_normalize_backward_cpu(
    const Tensor &grad_b_, const Tensor &grad_l_, const Tensor &grad_e_, const Tensor &grad_v_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_,
    const Tensor &values_, const Tensor &denom_,
    int64_t B, int64_t L, int64_t E, double eps
) {
    TORCH_CHECK(!values_.is_cuda(), "CPU sparse3d normalize backward expects CPU tensors");
    (void)B;
    const Tensor grad_b = grad_b_.to(at::kLong).contiguous();
    const Tensor grad_l = grad_l_.to(at::kLong).contiguous();
    const Tensor grad_e = grad_e_.to(at::kLong).contiguous();
    const Tensor grad_v = grad_v_.contiguous();
    const Tensor sp_batch = sp_batch_.to(at::kLong).contiguous();
    const Tensor sp_level = sp_level_.to(at::kLong).contiguous();
    const Tensor sp_entity = sp_entity_.to(at::kLong).contiguous();
    const Tensor values = values_.to(grad_v.scalar_type()).contiguous();
    const Tensor denom = denom_.to(grad_v.scalar_type()).contiguous().view({-1});
    Tensor grad_value = at::empty_like(values);

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse3d_normalize_backward_cpu", [&] {
        int64_t total_slots = B * L * E;
        bool use_dense_grad = total_slots <= 50000000LL;
        std::vector<scalar_t> dense_grad;
        std::vector<std::pair<int64_t, scalar_t>> grad_items;
        if (use_dense_grad) {
            dense_grad.assign(static_cast<size_t>(total_slots), scalar_t(0));
            for (int64_t i = 0; i < grad_v.size(0); ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                dense_grad[static_cast<size_t>(key)] += grad_v.data_ptr<scalar_t>()[i];
            }
        } else {
            grad_items.reserve(static_cast<size_t>(grad_v.size(0)));
            for (int64_t i = 0; i < grad_v.size(0); ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                grad_items.emplace_back(key, grad_v.data_ptr<scalar_t>()[i]);
            }
            std::sort(grad_items.begin(), grad_items.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
        }
        auto grad_at = [&](int64_t b, int64_t l, int64_t e) -> scalar_t {
            int64_t key = b * (L * E) + l * E + e;
            if (use_dense_grad) {
                return dense_grad[static_cast<size_t>(key)];
            }
            auto it = std::lower_bound(
                grad_items.begin(), grad_items.end(), key,
                [](const auto &a, int64_t k) { return a.first < k; });
            return (it != grad_items.end() && it->first == key) ? it->second : scalar_t(0);
        };

        int nt = std::max<int>(1, at::get_num_threads());
        int64_t BL = B * L;
        std::vector<std::vector<scalar_t>> local_sum_gx(nt, std::vector<scalar_t>(static_cast<size_t>(BL), scalar_t(0)));
        std::vector<scalar_t> gout(static_cast<size_t>(values.size(0)), scalar_t(0));
        at::parallel_for(0, values.size(0), 1024, [&](int64_t begin, int64_t end) {
            int tid = at::get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            auto &sum_gx = local_sum_gx[tid];
            for (int64_t i = begin; i < end; ++i) {
                int64_t b = sp_batch.data_ptr<int64_t>()[i];
                int64_t l = sp_level.data_ptr<int64_t>()[i];
                int64_t e = sp_entity.data_ptr<int64_t>()[i];
                scalar_t g = grad_at(b, l, e);
                gout[static_cast<size_t>(i)] = g;
                sum_gx[static_cast<size_t>(b * L + l)] += g * values.data_ptr<scalar_t>()[i];
            }
        });
        std::vector<scalar_t> sum_gx(static_cast<size_t>(BL), scalar_t(0));
        for (int t = 0; t < nt; ++t) {
            for (int64_t i = 0; i < BL; ++i) {
                sum_gx[static_cast<size_t>(i)] += local_sum_gx[t][static_cast<size_t>(i)];
            }
        }
        at::parallel_for(0, values.size(0), 1024, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                int64_t b = sp_batch.data_ptr<int64_t>()[i];
                int64_t l = sp_level.data_ptr<int64_t>()[i];
                int64_t group = b * L + l;
                scalar_t d0 = denom.data_ptr<scalar_t>()[group];
                scalar_t d = std::max(d0, static_cast<scalar_t>(eps));
                scalar_t gv = gout[static_cast<size_t>(i)] / d;
                if (d0 >= static_cast<scalar_t>(eps)) {
                    gv -= sum_gx[static_cast<size_t>(group)] / (d * d);
                }
                grad_value.data_ptr<scalar_t>()[i] = gv;
            }
        });
    });
    return grad_value;
}

std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,
           Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor>
flashlog_forward_sparse3d_sum_norm_cpu(
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_col_ind_, const Tensor &ori_mask_,
    const Tensor &ori_row_group_ptr_, const Tensor &ori_group_rel_,
    const Tensor &ori_group_edge_start_, const Tensor &ori_group_edge_count_,
    const Tensor &inv_col_ind_, const Tensor &inv_mask_,
    const Tensor &inv_row_group_ptr_, const Tensor &inv_group_rel_,
    const Tensor &inv_group_edge_start_, const Tensor &inv_group_edge_count_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, int64_t topk_edges, double eps,
    bool save_backward
) {
    TORCH_CHECK(!sp_value_.is_cuda(), "CPU sparse3d sum+norm expects CPU tensors");
    const Tensor sp_batch = sp_batch_.to(at::kLong).contiguous();
    const Tensor sp_level = sp_level_.to(at::kLong).contiguous();
    const Tensor sp_entity = sp_entity_.to(at::kLong).contiguous();
    const Tensor sp_value = sp_value_.contiguous();
    const Tensor w = w_.to(sp_value.scalar_type()).contiguous();
    const Tensor oci = ori_col_ind_.to(at::kInt).contiguous();
    const Tensor om = ori_mask_.to(sp_value.scalar_type()).contiguous();
    const Tensor orgp = ori_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor ogr = ori_group_rel_.to(at::kShort).contiguous();
    const Tensor ogs = ori_group_edge_start_.to(at::kInt).contiguous();
    const Tensor ogc = ori_group_edge_count_.to(at::kInt).contiguous();
    const Tensor ici = inv_col_ind_.to(at::kInt).contiguous();
    const Tensor im = inv_mask_.to(sp_value.scalar_type()).contiguous();
    const Tensor irgp = inv_row_group_ptr_.to(at::kInt).contiguous();
    const Tensor igr = inv_group_rel_.to(at::kShort).contiguous();
    const Tensor igs = inv_group_edge_start_.to(at::kInt).contiguous();
    const Tensor igc = inv_group_edge_count_.to(at::kInt).contiguous();

    Tensor out_b, out_l, out_e, out_v, out_keys, out_values, denom;
    Tensor ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m, ori_pos;
    Tensor inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m, inv_pos;
    Tensor ind_pos;

    AT_DISPATCH_FLOATING_TYPES(sp_value.scalar_type(), "flashlog_forward_sparse3d_sum_norm_cpu", [&] {
        int64_t nnz = sp_batch.size(0);
        int64_t n = w.size(2);
        int nt = std::max<int>(1, at::get_num_threads());
        std::vector<Sparse3DRawCpu<scalar_t>> ori_parts(nt);
        std::vector<Sparse3DRawCpu<scalar_t>> inv_parts(nt);
        Sparse3DRawCpu<scalar_t> ori_raw;
        Sparse3DRawCpu<scalar_t> inv_raw;
        std::vector<int64_t> keys;
        std::vector<scalar_t> values;
        std::vector<int64_t> ori_offsets_vec, inv_offsets_vec;
        std::vector<int64_t> ori_key_vec, inv_key_vec;
        std::vector<int16_t> ori_rel_vec, inv_rel_vec;
        std::vector<scalar_t> ori_mask_vec, inv_mask_vec;
        auto merge_raw = [](Sparse3DRawCpu<scalar_t> &dst, Sparse3DRawCpu<scalar_t> &src) {
            dst.b.insert(dst.b.end(), src.b.begin(), src.b.end());
            dst.l.insert(dst.l.end(), src.l.begin(), src.l.end());
            dst.e.insert(dst.e.end(), src.e.begin(), src.e.end());
            dst.entry.insert(dst.entry.end(), src.entry.begin(), src.entry.end());
            dst.rel.insert(dst.rel.end(), src.rel.begin(), src.rel.end());
            dst.mask.insert(dst.mask.end(), src.mask.begin(), src.mask.end());
        };

        bool use_rowwise_dense = (E * static_cast<int64_t>(nt) <= 20000000LL);
        if (use_rowwise_dense) {
            std::vector<std::vector<int64_t>> row_entries(static_cast<size_t>(B * L));
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = sp_batch.data_ptr<int64_t>()[i] * L + sp_level.data_ptr<int64_t>()[i];
                row_entries[static_cast<size_t>(row)].push_back(i);
            }

            const char *entry_offsets_env = std::getenv("FLASHLOG_CPU_ENTRY_OFFSETS");
            bool use_entry_offsets = save_backward && entry_offsets_env && std::strcmp(entry_offsets_env, "1") == 0;
            auto count_direction = [&](int64_t entry, int64_t rel_offset,
                                       const scalar_t *mask_ptr,
                                       const int32_t *rgp, const int16_t *grel,
                                       const int32_t *gestart, const int32_t *gecount) -> int64_t {
                int64_t src = sp_entity.data_ptr<int64_t>()[entry];
                int64_t g_start = rgp[src];
                int64_t g_end = rgp[src + 1];
                if (g_start == g_end) return 0;

                int64_t b = sp_batch.data_ptr<int64_t>()[entry];
                int64_t l = sp_level.data_ptr<int64_t>()[entry];
                int64_t stamp = b * L + l;
                const scalar_t *w_bl = w.data_ptr<scalar_t>() + stamp * n;

                int counts[NUM_BUCKETS] = {};
                int threshold_bucket = -1;
                int64_t keep_in_threshold = 0;
                if (topk_edges < (1LL << 60)) {
                    int64_t total_edges = 0;
                    for (int64_t g = g_start; g < g_end; ++g) {
                        int64_t rel = grel[g];
                        int bucket = bucketize_weight_cpu(w_bl[rel_offset + rel]);
                        int32_t cnt = gecount[g];
                        counts[bucket] += cnt;
                        total_edges += cnt;
                    }
                    if (total_edges > topk_edges) {
                        int64_t cumsum = 0;
                        for (int bucket = NUM_BUCKETS - 1; bucket >= 0; --bucket) {
                            if (cumsum + counts[bucket] >= topk_edges) {
                                threshold_bucket = bucket;
                                keep_in_threshold = topk_edges - cumsum;
                                break;
                            }
                            cumsum += counts[bucket];
                        }
                    }
                }

                int64_t selected = 0;
                int64_t threshold_seen = 0;
                for (int64_t g = g_start; g < g_end; ++g) {
                    scalar_t w_val = w_bl[rel_offset + grel[g]];
                    int32_t take = gecount[g];
                    if (threshold_bucket >= 0) {
                        int bucket = bucketize_weight_cpu(w_val);
                        if (bucket < threshold_bucket) {
                            take = 0;
                        } else if (bucket == threshold_bucket) {
                            int64_t left = keep_in_threshold - threshold_seen;
                            take = left > 0
                                ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(take)))
                                : 0;
                            threshold_seen += take;
                        }
                    }
                    if (take <= 0) continue;
                    int64_t edge_start = gestart[g];
                    for (int64_t p = edge_start; p < edge_start + take; ++p) {
                        if (mask_ptr[p] != 0) ++selected;
                    }
                }
                return selected;
            };

            if (use_entry_offsets) {
                ori_offsets_vec.assign(static_cast<size_t>(nnz + 1), 0);
                inv_offsets_vec.assign(static_cast<size_t>(nnz + 1), 0);
                at::parallel_for(0, nnz, 64, [&](int64_t begin, int64_t end) {
                    for (int64_t entry = begin; entry < end; ++entry) {
                        ori_offsets_vec[static_cast<size_t>(entry + 1)] = count_direction(
                            entry, 0, om.data_ptr<scalar_t>(),
                            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>());
                        inv_offsets_vec[static_cast<size_t>(entry + 1)] = count_direction(
                            entry, r_size, im.data_ptr<scalar_t>(),
                            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>());
                    }
                });
                for (int64_t i = 0; i < nnz; ++i) {
                    ori_offsets_vec[static_cast<size_t>(i + 1)] += ori_offsets_vec[static_cast<size_t>(i)];
                    inv_offsets_vec[static_cast<size_t>(i + 1)] += inv_offsets_vec[static_cast<size_t>(i)];
                }
                ori_key_vec.resize(static_cast<size_t>(ori_offsets_vec.back()));
                inv_key_vec.resize(static_cast<size_t>(inv_offsets_vec.back()));
                ori_rel_vec.resize(static_cast<size_t>(ori_offsets_vec.back()));
                inv_rel_vec.resize(static_cast<size_t>(inv_offsets_vec.back()));
                ori_mask_vec.resize(static_cast<size_t>(ori_offsets_vec.back()));
                inv_mask_vec.resize(static_cast<size_t>(inv_offsets_vec.back()));
            }

            std::vector<std::vector<scalar_t>> acc_parts(nt, std::vector<scalar_t>(static_cast<size_t>(E), scalar_t(0)));
            std::vector<std::vector<int64_t>> mark_parts(nt, std::vector<int64_t>(static_cast<size_t>(E), int64_t(-1)));
            std::vector<std::vector<std::pair<int64_t, scalar_t>>> out_parts(nt);

            auto add_value = [&](int tid, int64_t stamp, int64_t dst, scalar_t value, std::vector<int64_t> &touched) {
                auto &mark = mark_parts[tid];
                auto &acc = acc_parts[tid];
                if (mark[static_cast<size_t>(dst)] != stamp) {
                    mark[static_cast<size_t>(dst)] = stamp;
                    acc[static_cast<size_t>(dst)] = scalar_t(0);
                    touched.push_back(dst);
                }
                acc[static_cast<size_t>(dst)] += value;
            };

            auto process_direction = [&](int tid, int64_t entry, int64_t rel_offset,
                                         const int32_t *col_ptr, const scalar_t *mask_ptr,
                                         const int32_t *rgp, const int16_t *grel,
                                         const int32_t *gestart, const int32_t *gecount,
                                         Sparse3DRawCpu<scalar_t> &raw,
                                         std::vector<int64_t> &touched) {
                int64_t src = sp_entity.data_ptr<int64_t>()[entry];
                int64_t g_start = rgp[src];
                int64_t g_end = rgp[src + 1];
                if (g_start == g_end) return;

                int64_t b = sp_batch.data_ptr<int64_t>()[entry];
                int64_t l = sp_level.data_ptr<int64_t>()[entry];
                int64_t stamp = b * L + l;
                const scalar_t *w_bl = w.data_ptr<scalar_t>() + stamp * n;

                int counts[NUM_BUCKETS] = {};
                int threshold_bucket = -1;
                int64_t keep_in_threshold = 0;
                if (topk_edges < (1LL << 60)) {
                    int64_t total_edges = 0;
                    for (int64_t g = g_start; g < g_end; ++g) {
                        int64_t rel = grel[g];
                        int bucket = bucketize_weight_cpu(w_bl[rel_offset + rel]);
                        int32_t cnt = gecount[g];
                        counts[bucket] += cnt;
                        total_edges += cnt;
                    }
                    if (total_edges > topk_edges) {
                        int64_t cumsum = 0;
                        for (int bucket = NUM_BUCKETS - 1; bucket >= 0; --bucket) {
                            if (cumsum + counts[bucket] >= topk_edges) {
                                threshold_bucket = bucket;
                                keep_in_threshold = topk_edges - cumsum;
                                break;
                            }
                            cumsum += counts[bucket];
                        }
                    }
                }

                int64_t threshold_seen = 0;
                auto &offsets = (rel_offset == 0) ? ori_offsets_vec : inv_offsets_vec;
                auto &key_vec = (rel_offset == 0) ? ori_key_vec : inv_key_vec;
                auto &rel_vec = (rel_offset == 0) ? ori_rel_vec : inv_rel_vec;
                auto &mask_vec = (rel_offset == 0) ? ori_mask_vec : inv_mask_vec;
                int64_t raw_pos = use_entry_offsets ? offsets[static_cast<size_t>(entry)] : 0;
                for (int64_t g = g_start; g < g_end; ++g) {
                    int64_t rel = grel[g];
                    scalar_t w_val = w_bl[rel_offset + rel];
                    int32_t take = gecount[g];
                    if (threshold_bucket >= 0) {
                        int bucket = bucketize_weight_cpu(w_val);
                        if (bucket < threshold_bucket) {
                            take = 0;
                        } else if (bucket == threshold_bucket) {
                            int64_t left = keep_in_threshold - threshold_seen;
                            take = left > 0
                                ? static_cast<int32_t>(std::min(left, static_cast<int64_t>(take)))
                                : 0;
                            threshold_seen += take;
                        }
                    }
                    if (take <= 0) continue;
                    int64_t edge_start = gestart[g];
                    for (int64_t p = edge_start; p < edge_start + take; ++p) {
                        scalar_t m = mask_ptr[p];
                        if (m == 0) continue;
                        int64_t dst = col_ptr[p];
                        scalar_t value = sp_value.data_ptr<scalar_t>()[entry] * w_val * m;
                        add_value(tid, stamp, dst, value, touched);
                        if (save_backward) {
                            if (use_entry_offsets) {
                                key_vec[static_cast<size_t>(raw_pos)] = stamp * E + dst;
                                rel_vec[static_cast<size_t>(raw_pos)] = static_cast<int16_t>(rel);
                                mask_vec[static_cast<size_t>(raw_pos)] = m;
                                ++raw_pos;
                            } else {
                                raw.b.push_back(stamp * E + dst);
                                raw.entry.push_back(entry);
                                raw.rel.push_back(static_cast<int16_t>(rel));
                                raw.mask.push_back(m);
                            }
                        }
                    }
                }
            };

            at::parallel_for(0, B * L, 1, [&](int64_t row_begin, int64_t row_end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                std::vector<int64_t> touched_row;
                for (int64_t row = row_begin; row < row_end; ++row) {
                    const auto &entries = row_entries[static_cast<size_t>(row)];
                    if (entries.empty()) continue;
                    touched_row.clear();
                    for (int64_t entry : entries) {
                        process_direction(tid, entry, 0,
                            oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                            orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                            ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                            ori_parts[tid], touched_row);
                        process_direction(tid, entry, r_size,
                            ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                            irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                            igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                            inv_parts[tid], touched_row);
                        if (!wot_i) {
                            int64_t dst = sp_entity.data_ptr<int64_t>()[entry];
                            scalar_t value = sp_value.data_ptr<scalar_t>()[entry] *
                                w.data_ptr<scalar_t>()[row * n + (n - 1)];
                            add_value(tid, row, dst, value, touched_row);
                        }
                    }
                    std::sort(touched_row.begin(), touched_row.end());
                    auto &out = out_parts[tid];
                    for (int64_t ent : touched_row) {
                        out.emplace_back(row * E + ent, acc_parts[tid][static_cast<size_t>(ent)]);
                    }
                }
            });

            size_t out_total = 0;
            for (int t = 0; t < nt; ++t) out_total += out_parts[t].size();
            std::vector<std::pair<int64_t, scalar_t>> out_items;
            out_items.reserve(out_total);
            for (int t = 0; t < nt; ++t) {
                out_items.insert(out_items.end(), out_parts[t].begin(), out_parts[t].end());
                if (!use_entry_offsets) {
                    merge_raw(ori_raw, ori_parts[t]);
                    merge_raw(inv_raw, inv_parts[t]);
                }
            }
            keys.reserve(out_items.size());
            values.reserve(out_items.size());
            for (const auto &item : out_items) {
                keys.push_back(item.first);
                values.push_back(item.second);
            }
        } else {
            std::vector<std::vector<std::pair<int64_t, scalar_t>>> item_parts(nt);
            at::parallel_for(0, nnz, 64, [&](int64_t begin, int64_t end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                auto &items = item_parts[tid];
                expand_sparse3d_direction_sum_cpu_range<scalar_t>(
                    sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                    sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), begin, end,
                    oci.data_ptr<int32_t>(), om.data_ptr<scalar_t>(),
                    orgp.data_ptr<int32_t>(), ogr.data_ptr<int16_t>(),
                    ogs.data_ptr<int32_t>(), ogc.data_ptr<int32_t>(),
                    L, E, n, 0, topk_edges, ori_parts[tid], items);
                expand_sparse3d_direction_sum_cpu_range<scalar_t>(
                    sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                    sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), begin, end,
                    ici.data_ptr<int32_t>(), im.data_ptr<scalar_t>(),
                    irgp.data_ptr<int32_t>(), igr.data_ptr<int16_t>(),
                    igs.data_ptr<int32_t>(), igc.data_ptr<int32_t>(),
                    L, E, n, r_size, topk_edges, inv_parts[tid], items);
                if (!wot_i) {
                    append_identity_sum_cpu_range<scalar_t>(
                        sp_batch.data_ptr<int64_t>(), sp_level.data_ptr<int64_t>(), sp_entity.data_ptr<int64_t>(),
                        sp_value.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                        begin, end, L, E, n, items);
                }
            });

            std::vector<std::unordered_map<int64_t, scalar_t>> row_maps(static_cast<size_t>(B * L));
            for (int t = 0; t < nt; ++t) {
                for (const auto &item : item_parts[t]) {
                    int64_t row = item.first / E;
                    int64_t ent = item.first % E;
                    row_maps[static_cast<size_t>(row)][ent] += item.second;
                }
                merge_raw(ori_raw, ori_parts[t]);
                merge_raw(inv_raw, inv_parts[t]);
            }
            size_t out_total = 0;
            for (const auto &row_map : row_maps) out_total += row_map.size();
            keys.reserve(out_total);
            values.reserve(out_total);
            for (int64_t row = 0; row < B * L; ++row) {
                auto &row_map = row_maps[static_cast<size_t>(row)];
                if (row_map.empty()) continue;
                std::vector<int64_t> ents;
                ents.reserve(row_map.size());
                for (const auto &kv : row_map) ents.push_back(kv.first);
                std::sort(ents.begin(), ents.end());
                for (int64_t ent : ents) {
                    keys.push_back(row * E + ent);
                    values.push_back(row_map[ent]);
                }
            }
        }

        std::vector<scalar_t> denom_vec(static_cast<size_t>(B * L), scalar_t(0));
        int64_t LE = L * E;
        for (size_t i = 0; i < keys.size(); ++i) {
            int64_t group = keys[i] / E;
            denom_vec[static_cast<size_t>(group)] += values[i];
        }

        // The row-wise path stores flattened (batch, level, entity) keys in
        // raw.b. The large-graph hash-map path stores the same destinations in
        // raw.b/raw.l/raw.e. The fused backward interface below always expects
        // flattened keys, so compact the large-graph representation before
        // exporting it. Previously raw.l/raw.e were discarded and raw.b was
        // misinterpreted as a key, which corrupted large-graph gradients.
        auto compact_raw_keys = [L, E](Sparse3DRawCpu<scalar_t> &raw) {
            if (raw.l.empty() && raw.e.empty()) return;
            TORCH_CHECK(raw.b.size() == raw.l.size() && raw.b.size() == raw.e.size(),
                        "invalid CPU sparse3d backward metadata");
            for (size_t i = 0; i < raw.b.size(); ++i) {
                raw.b[i] = raw.b[i] * (L * E) + raw.l[i] * E + raw.e[i];
            }
            raw.l.clear();
            raw.e.clear();
        };
        compact_raw_keys(ori_raw);
        compact_raw_keys(inv_raw);

        auto opts_long = sp_batch.options().dtype(at::kLong);
        auto opts_val = sp_value.options();
        int64_t out_n = static_cast<int64_t>(keys.size());
        out_b = at::empty({out_n}, opts_long);
        out_l = at::empty({out_n}, opts_long);
        out_e = at::empty({out_n}, opts_long);
        out_v = at::empty({out_n}, opts_val);
        out_keys = save_backward ? tensor_from_i64_vector<scalar_t>(keys, opts_long) : at::empty({0}, opts_long);
        out_values = save_backward ? tensor_from_scalar_vector<scalar_t>(values, opts_val) : at::empty({0}, opts_val);
        denom = save_backward ? tensor_from_scalar_vector<scalar_t>(denom_vec, opts_val) : at::empty({0}, opts_val);
        for (int64_t i = 0; i < out_n; ++i) {
            int64_t key = keys[static_cast<size_t>(i)];
            out_b.data_ptr<int64_t>()[i] = key / LE;
            int64_t rem = key % LE;
            out_l.data_ptr<int64_t>()[i] = rem / E;
            out_e.data_ptr<int64_t>()[i] = rem % E;
            int64_t group = key / E;
            scalar_t d = std::max(denom_vec[static_cast<size_t>(group)], static_cast<scalar_t>(eps));
            out_v.data_ptr<scalar_t>()[i] = values[static_cast<size_t>(i)] / d;
        }

        if (save_backward) {
            ori_b = (!ori_offsets_vec.empty())
                ? tensor_from_i64_vector<scalar_t>(ori_key_vec, opts_long)
                : tensor_from_i64_vector<scalar_t>(ori_raw.b, opts_long);
            ori_l = at::empty({0}, opts_long);
            ori_e = at::empty({0}, opts_long);
            ori_entry = (!ori_offsets_vec.empty())
                ? tensor_from_i64_vector<scalar_t>(ori_offsets_vec, opts_long)
                : tensor_from_i64_vector<scalar_t>(ori_raw.entry, opts_long);
            ori_rel = (!ori_offsets_vec.empty())
                ? tensor_from_i16_vector<scalar_t>(ori_rel_vec, opts_long)
                : tensor_from_i16_vector<scalar_t>(ori_raw.rel, opts_long);
            ori_m = (!ori_offsets_vec.empty())
                ? tensor_from_scalar_vector<scalar_t>(ori_mask_vec, opts_val)
                : tensor_from_scalar_vector<scalar_t>(ori_raw.mask, opts_val);
        } else {
            ori_b = at::empty({0}, opts_long);
            ori_l = at::empty({0}, opts_long);
            ori_e = at::empty({0}, opts_long);
            ori_entry = at::empty({0}, opts_long);
            ori_rel = at::empty({0}, opts_long.dtype(at::kShort));
            ori_m = at::empty({0}, opts_val);
        }
        ori_pos = at::empty({0}, opts_long);
        if (save_backward) {
            inv_b = (!inv_offsets_vec.empty())
                ? tensor_from_i64_vector<scalar_t>(inv_key_vec, opts_long)
                : tensor_from_i64_vector<scalar_t>(inv_raw.b, opts_long);
            inv_l = at::empty({0}, opts_long);
            inv_e = at::empty({0}, opts_long);
            inv_entry = (!inv_offsets_vec.empty())
                ? tensor_from_i64_vector<scalar_t>(inv_offsets_vec, opts_long)
                : tensor_from_i64_vector<scalar_t>(inv_raw.entry, opts_long);
            inv_rel = (!inv_offsets_vec.empty())
                ? tensor_from_i16_vector<scalar_t>(inv_rel_vec, opts_long)
                : tensor_from_i16_vector<scalar_t>(inv_raw.rel, opts_long);
            inv_m = (!inv_offsets_vec.empty())
                ? tensor_from_scalar_vector<scalar_t>(inv_mask_vec, opts_val)
                : tensor_from_scalar_vector<scalar_t>(inv_raw.mask, opts_val);
        } else {
            inv_b = at::empty({0}, opts_long);
            inv_l = at::empty({0}, opts_long);
            inv_e = at::empty({0}, opts_long);
            inv_entry = at::empty({0}, opts_long);
            inv_rel = at::empty({0}, opts_long.dtype(at::kShort));
            inv_m = at::empty({0}, opts_val);
        }
        inv_pos = at::empty({0}, opts_long);
        ind_pos = at::empty({0}, opts_long);
    });

    return std::make_tuple(
        out_b, out_l, out_e, out_v, out_keys, out_values, denom,
        ori_b, ori_l, ori_e, ori_entry, ori_rel, ori_m, ori_pos,
        inv_b, inv_l, inv_e, inv_entry, inv_rel, inv_m, inv_pos, ind_pos
    );
}

std::tuple<Tensor, Tensor> flashlog_backward_sparse3d_sum_norm_cpu(
    const Tensor &grad_b_, const Tensor &grad_l_, const Tensor &grad_e_, const Tensor &grad_v_,
    const Tensor &out_keys_, const Tensor &out_values_, const Tensor &denom_,
    const Tensor &sp_batch_, const Tensor &sp_level_, const Tensor &sp_entity_, const Tensor &sp_value_,
    const Tensor &w_,
    const Tensor &ori_b_, const Tensor &ori_l_, const Tensor &ori_e_, const Tensor &ori_entry_,
    const Tensor &ori_rel_, const Tensor &ori_m_, const Tensor &ori_pos_,
    const Tensor &inv_b_, const Tensor &inv_l_, const Tensor &inv_e_, const Tensor &inv_entry_,
    const Tensor &inv_rel_, const Tensor &inv_m_, const Tensor &inv_pos_, const Tensor &ind_pos_,
    int64_t B, int64_t L, int64_t E, int64_t r_size, bool wot_i, double eps
) {
    TORCH_CHECK(!sp_value_.is_cuda(), "CPU sparse3d sum+norm backward expects CPU tensors");
    (void)B;
    const Tensor grad_b = grad_b_.to(at::kLong).contiguous();
    const Tensor grad_l = grad_l_.to(at::kLong).contiguous();
    const Tensor grad_e = grad_e_.to(at::kLong).contiguous();
    const Tensor grad_v = grad_v_.contiguous();
    const Tensor out_keys = out_keys_.to(at::kLong).contiguous();
    const Tensor out_values = out_values_.to(grad_v.scalar_type()).contiguous();
    const Tensor denom = denom_.to(grad_v.scalar_type()).contiguous();
    const Tensor sp_batch = sp_batch_.to(at::kLong).contiguous();
    const Tensor sp_level = sp_level_.to(at::kLong).contiguous();
    const Tensor sp_entity = sp_entity_.to(at::kLong).contiguous();
    const Tensor sp_value = sp_value_.to(grad_v.scalar_type()).contiguous();
    const Tensor w = w_.to(grad_v.scalar_type()).contiguous();
    const Tensor ori_b = ori_b_.to(at::kLong).contiguous();
    const Tensor ori_l = ori_l_.to(at::kLong).contiguous();
    const Tensor ori_e = ori_e_.to(at::kLong).contiguous();
    const Tensor ori_entry = ori_entry_.to(at::kLong).contiguous();
    const Tensor ori_rel = ori_rel_.to(at::kShort).contiguous();
    const Tensor ori_m = ori_m_.to(grad_v.scalar_type()).contiguous();
    const Tensor ori_pos = ori_pos_.to(at::kLong).contiguous();
    const Tensor inv_b = inv_b_.to(at::kLong).contiguous();
    const Tensor inv_l = inv_l_.to(at::kLong).contiguous();
    const Tensor inv_e = inv_e_.to(at::kLong).contiguous();
    const Tensor inv_entry = inv_entry_.to(at::kLong).contiguous();
    const Tensor inv_rel = inv_rel_.to(at::kShort).contiguous();
    const Tensor inv_m = inv_m_.to(grad_v.scalar_type()).contiguous();
    const Tensor inv_pos = inv_pos_.to(at::kLong).contiguous();
    const Tensor ind_pos = ind_pos_.to(at::kLong).contiguous();

    Tensor grad_sp = at::zeros({sp_value.size(0)}, sp_value.options());
    Tensor grad_w = at::zeros_like(w);

    AT_DISPATCH_FLOATING_TYPES(sp_value.scalar_type(), "flashlog_backward_sparse3d_sum_norm_cpu", [&] {
        int64_t out_n = out_keys.size(0);
        int64_t grad_n = grad_v.size(0);
        int64_t BL = denom.size(0);
        int64_t n = w.size(2);
        const int64_t *keys = out_keys.data_ptr<int64_t>();
        std::vector<scalar_t> grad_z(static_cast<size_t>(out_n), scalar_t(0));
        int64_t total_slots = B * L * E;
        bool use_dense_grad = total_slots <= 50000000LL;
        int nt = std::max<int>(1, at::get_num_threads());
        if (use_dense_grad) {
            std::vector<scalar_t> dense_grad(static_cast<size_t>(total_slots), scalar_t(0));
            for (int64_t i = 0; i < grad_n; ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                dense_grad[static_cast<size_t>(key)] += grad_v.data_ptr<scalar_t>()[i];
            }
            for (int64_t i = 0; i < out_n; ++i) {
                grad_z[static_cast<size_t>(i)] = dense_grad[static_cast<size_t>(keys[i])];
            }
        } else {
            std::vector<std::pair<int64_t, scalar_t>> grad_items;
            grad_items.reserve(static_cast<size_t>(grad_n));
            for (int64_t i = 0; i < grad_n; ++i) {
                int64_t key = grad_b.data_ptr<int64_t>()[i] * (L * E)
                            + grad_l.data_ptr<int64_t>()[i] * E
                            + grad_e.data_ptr<int64_t>()[i];
                grad_items.emplace_back(key, grad_v.data_ptr<scalar_t>()[i]);
            }
            std::sort(grad_items.begin(), grad_items.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (int64_t i = 0; i < out_n; ++i) {
                auto it = std::lower_bound(
                    grad_items.begin(), grad_items.end(), keys[i],
                    [](const auto &a, int64_t k) { return a.first < k; });
                if (it != grad_items.end() && it->first == keys[i]) {
                    grad_z[static_cast<size_t>(i)] = it->second;
                }
            }
        }

        std::vector<scalar_t> sum_gx(static_cast<size_t>(BL), scalar_t(0));
        for (int64_t i = 0; i < out_n; ++i) {
            int64_t group = keys[i] / E;
            sum_gx[static_cast<size_t>(group)] += grad_z[static_cast<size_t>(i)] * out_values.data_ptr<scalar_t>()[i];
        }
        std::vector<scalar_t> grad_y(static_cast<size_t>(out_n), scalar_t(0));
        for (int64_t i = 0; i < out_n; ++i) {
            int64_t group = keys[i] / E;
            scalar_t d0 = denom.data_ptr<scalar_t>()[group];
            scalar_t d = std::max(d0, static_cast<scalar_t>(eps));
            scalar_t gy = grad_z[static_cast<size_t>(i)] / d;
            if (d0 >= static_cast<scalar_t>(eps)) {
                gy -= sum_gx[static_cast<size_t>(group)] / (d * d);
            }
            grad_y[static_cast<size_t>(i)] = gy;
        }
        bool use_dense_gy = total_slots <= 50000000LL;
        std::vector<scalar_t> dense_gy;
        std::vector<std::pair<int64_t, scalar_t>> gy_items;
        if (use_dense_gy) {
            dense_gy.assign(static_cast<size_t>(total_slots), scalar_t(0));
            for (int64_t i = 0; i < out_n; ++i) {
                dense_gy[static_cast<size_t>(keys[i])] = grad_y[static_cast<size_t>(i)];
            }
        } else {
            gy_items.reserve(static_cast<size_t>(out_n));
            for (int64_t i = 0; i < out_n; ++i) {
                gy_items.emplace_back(keys[i], grad_y[static_cast<size_t>(i)]);
            }
            std::sort(gy_items.begin(), gy_items.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
        }
        auto grad_y_at_key = [&](int64_t key) -> scalar_t {
            if (use_dense_gy) {
                return dense_gy[static_cast<size_t>(key)];
            }
            auto it = std::lower_bound(
                gy_items.begin(), gy_items.end(), key,
                [](const auto &a, int64_t k) { return a.first < k; });
            return (it != gy_items.end() && it->first == key) ? it->second : scalar_t(0);
        };
        auto grad_y_at = [&](int64_t b, int64_t l, int64_t e) -> scalar_t {
            return grad_y_at_key(b * (L * E) + l * E + e);
        };

        const scalar_t *spv = sp_value.data_ptr<scalar_t>();
        const scalar_t *wp = w.data_ptr<scalar_t>();
        int64_t sp_n = sp_value.size(0);
        int64_t w_n = w.numel();
        bool ori_offsets_format = (ori_l.size(0) == 0 && ori_e.size(0) == 0 &&
                                   ori_entry.size(0) == sp_n + 1 && ori_b.size(0) == ori_rel.size(0));
        bool inv_offsets_format = (inv_l.size(0) == 0 && inv_e.size(0) == 0 &&
                                   inv_entry.size(0) == sp_n + 1 && inv_b.size(0) == inv_rel.size(0));
        if (ori_offsets_format && inv_offsets_format) {
            std::vector<std::vector<scalar_t>> local_w(nt, std::vector<scalar_t>(static_cast<size_t>(w_n), scalar_t(0)));
            scalar_t *gsp_out = grad_sp.data_ptr<scalar_t>();
            at::parallel_for(0, sp_n, 64, [&](int64_t begin, int64_t end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                auto &gw = local_w[tid];
                for (int64_t entry = begin; entry < end; ++entry) {
                    scalar_t gsp_acc = scalar_t(0);
                    int64_t os = ori_entry.data_ptr<int64_t>()[entry];
                    int64_t oe = ori_entry.data_ptr<int64_t>()[entry + 1];
                    for (int64_t i = os; i < oe; ++i) {
                        int64_t key = ori_b.data_ptr<int64_t>()[i];
                        int64_t row = key / E;
                        int64_t rel = ori_rel.data_ptr<int16_t>()[i];
                        scalar_t g = grad_y_at_key(key);
                        if (g == 0) continue;
                        scalar_t m = ori_m.data_ptr<scalar_t>()[i];
                        gsp_acc += g * wp[row * n + rel] * m;
                        gw[static_cast<size_t>(row * n + rel)] += g * spv[entry] * m;
                    }

                    int64_t is = inv_entry.data_ptr<int64_t>()[entry];
                    int64_t ie = inv_entry.data_ptr<int64_t>()[entry + 1];
                    for (int64_t i = is; i < ie; ++i) {
                        int64_t key = inv_b.data_ptr<int64_t>()[i];
                        int64_t row = key / E;
                        int64_t rel = static_cast<int64_t>(inv_rel.data_ptr<int16_t>()[i]) + r_size;
                        scalar_t g = grad_y_at_key(key);
                        if (g == 0) continue;
                        scalar_t m = inv_m.data_ptr<scalar_t>()[i];
                        gsp_acc += g * wp[row * n + rel] * m;
                        gw[static_cast<size_t>(row * n + rel)] += g * spv[entry] * m;
                    }

                    if (!wot_i) {
                        int64_t b = sp_batch.data_ptr<int64_t>()[entry];
                        int64_t l = sp_level.data_ptr<int64_t>()[entry];
                        int64_t e = sp_entity.data_ptr<int64_t>()[entry];
                        int64_t rel = n - 1;
                        scalar_t g = grad_y_at(b, l, e);
                        if (g != 0) {
                            int64_t row = b * L + l;
                            gsp_acc += g * wp[row * n + rel];
                            gw[static_cast<size_t>(row * n + rel)] += g * spv[entry];
                        }
                    }
                    gsp_out[entry] = gsp_acc;
                }
            });
            scalar_t *gw_out = grad_w.data_ptr<scalar_t>();
            for (int t = 0; t < nt; ++t) {
                for (int64_t i = 0; i < w_n; ++i) gw_out[i] += local_w[t][static_cast<size_t>(i)];
            }
            return;
        }
        bool ori_compact = (ori_l.size(0) == 0 && ori_e.size(0) == 0 && ori_b.size(0) == ori_entry.size(0));
        bool inv_compact = (inv_l.size(0) == 0 && inv_e.size(0) == 0 && inv_b.size(0) == inv_entry.size(0));
        std::vector<std::vector<scalar_t>> local_sp(nt, std::vector<scalar_t>(static_cast<size_t>(sp_n), scalar_t(0)));
        std::vector<std::vector<scalar_t>> local_w(nt, std::vector<scalar_t>(static_cast<size_t>(w_n), scalar_t(0)));

        at::parallel_for(0, ori_b.size(0), 512, [&](int64_t begin, int64_t end) {
            int tid = at::get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            auto &gsp = local_sp[tid];
            auto &gw = local_w[tid];
            for (int64_t i = begin; i < end; ++i) {
                int64_t key = ori_compact
                    ? ori_b.data_ptr<int64_t>()[i]
                    : ori_b.data_ptr<int64_t>()[i] * (L * E) + ori_l.data_ptr<int64_t>()[i] * E + ori_e.data_ptr<int64_t>()[i];
                int64_t row = key / E;
                int64_t entry = ori_entry.data_ptr<int64_t>()[i];
                int64_t rel = ori_rel.data_ptr<int16_t>()[i];
                scalar_t g = grad_y_at_key(key);
                if (g == 0) continue;
                scalar_t m = ori_m.data_ptr<scalar_t>()[i];
                gsp[static_cast<size_t>(entry)] += g * wp[row * n + rel] * m;
                gw[static_cast<size_t>(row * n + rel)] += g * spv[entry] * m;
            }
        });
        at::parallel_for(0, inv_b.size(0), 512, [&](int64_t begin, int64_t end) {
            int tid = at::get_thread_num();
            if (tid < 0 || tid >= nt) tid = 0;
            auto &gsp = local_sp[tid];
            auto &gw = local_w[tid];
            for (int64_t i = begin; i < end; ++i) {
                int64_t key = inv_compact
                    ? inv_b.data_ptr<int64_t>()[i]
                    : inv_b.data_ptr<int64_t>()[i] * (L * E) + inv_l.data_ptr<int64_t>()[i] * E + inv_e.data_ptr<int64_t>()[i];
                int64_t row = key / E;
                int64_t entry = inv_entry.data_ptr<int64_t>()[i];
                int64_t rel = static_cast<int64_t>(inv_rel.data_ptr<int16_t>()[i]) + r_size;
                scalar_t g = grad_y_at_key(key);
                if (g == 0) continue;
                scalar_t m = inv_m.data_ptr<scalar_t>()[i];
                gsp[static_cast<size_t>(entry)] += g * wp[row * n + rel] * m;
                gw[static_cast<size_t>(row * n + rel)] += g * spv[entry] * m;
            }
        });
        if (!wot_i) {
            at::parallel_for(0, sp_value.size(0), 512, [&](int64_t begin, int64_t end) {
                int tid = at::get_thread_num();
                if (tid < 0 || tid >= nt) tid = 0;
                auto &gsp = local_sp[tid];
                auto &gw = local_w[tid];
                for (int64_t i = begin; i < end; ++i) {
                    int64_t b = sp_batch.data_ptr<int64_t>()[i];
                    int64_t l = sp_level.data_ptr<int64_t>()[i];
                    int64_t e = sp_entity.data_ptr<int64_t>()[i];
                    int64_t rel = n - 1;
                    scalar_t g = grad_y_at(b, l, e);
                    if (g == 0) continue;
                    gsp[static_cast<size_t>(i)] += g * wp[(b * L + l) * n + rel];
                    gw[static_cast<size_t>((b * L + l) * n + rel)] += g * spv[i];
                }
            });
        }
        scalar_t *gsp_out = grad_sp.data_ptr<scalar_t>();
        scalar_t *gw_out = grad_w.data_ptr<scalar_t>();
        for (int t = 0; t < nt; ++t) {
            for (int64_t i = 0; i < sp_n; ++i) gsp_out[i] += local_sp[t][static_cast<size_t>(i)];
            for (int64_t i = 0; i < w_n; ++i) gw_out[i] += local_w[t][static_cast<size_t>(i)];
        }
    });

    return std::make_tuple(grad_sp, grad_w);
}

} // namespace flashlog

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashlog_forward_cpu",  &flashlog::flashlog_forward_cpu);
    m.def("flashlog_backward_cpu", &flashlog::flashlog_backward_cpu);
    m.def("sparse3d_coalesce_sum3_cpu", &flashlog::sparse3d_coalesce_sum3_cpu);
    m.def("flashlog_forward_sparse3d_sum_cpu", &flashlog::flashlog_forward_sparse3d_sum_cpu);
    m.def("flashlog_backward_sparse3d_sum_cpu", &flashlog::flashlog_backward_sparse3d_sum_cpu);
    m.def("sparse3d_normalize_backward_cpu", &flashlog::sparse3d_normalize_backward_cpu);
    m.def("flashlog_forward_sparse3d_sum_norm_cpu", &flashlog::flashlog_forward_sparse3d_sum_norm_cpu);
    m.def("flashlog_backward_sparse3d_sum_norm_cpu", &flashlog::flashlog_backward_sparse3d_sum_norm_cpu);
#ifdef WITH_CUDA
    m.def("flashlog_forward_cuda",  &flashlog::flashlog_forward_cuda);
    m.def("flashlog_backward_cuda", &flashlog::flashlog_backward_cuda);
    m.def("flashlog_forward_max_cuda", &flashlog::flashlog_forward_max_cuda);
    m.def("flashlog_backward_max_cuda", &flashlog::flashlog_backward_max_cuda);
    m.def("compute_active_nodes_cuda", &flashlog::compute_active_nodes_cuda);
    m.def("apply_mask_cuda",      &flashlog::apply_mask_cuda);
    m.def("flashlog_forward_sparse3d_topk_cuda",  &flashlog::flashlog_forward_sparse3d_topk_cuda);
    m.def("flashlog_forward_sparse3d_alledges_cuda",  &flashlog::flashlog_forward_sparse3d_alledges_cuda);
    m.def("flashlog_forward_sparse3d_fixed_relation_cuda", &flashlog::flashlog_forward_sparse3d_fixed_relation_cuda);
    m.def("flashlog_backward_sparse3d_topk_cuda", &flashlog::flashlog_backward_sparse3d_topk_cuda);
    m.def("flashlog_backward_sparse3d_topk_aligned_cuda", &flashlog::flashlog_backward_sparse3d_topk_aligned_cuda);
    m.def("sparse3d_normalize_forward_cuda", &flashlog::sparse3d_normalize_forward_cuda);
    m.def("sparse3d_normalize_backward_cuda", &flashlog::sparse3d_normalize_backward_cuda);
    m.def("sparse3d_coalesce_sum3_cuda", &flashlog::sparse3d_coalesce_sum3_cuda);
    m.def("sparse3d_coalesce_sum3_cached_cuda", &flashlog::sparse3d_coalesce_sum3_cached_cuda);
    m.def("flashlog_backward_sparse3d_cached_cuda", &flashlog::flashlog_backward_sparse3d_cached_cuda);
    m.def("flashlog_backward_sparse3d_max_cached_cuda", &flashlog::flashlog_backward_sparse3d_max_cached_cuda);
    m.def("sparse2d_filtered_ranks_cuda", &flashlog::sparse2d_filtered_ranks_cuda);
#endif
}
