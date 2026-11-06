#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef FLASHLOG_SAMPLING_TEMPERATURE
#define FLASHLOG_SAMPLING_TEMPERATURE 1.0
#endif

#ifndef FLASHLOG_RULE_STEPS
#define FLASHLOG_RULE_STEPS 3
#endif
static_assert(FLASHLOG_RULE_STEPS > 0, "rule steps must be positive");

namespace {
constexpr int rule_steps = FLASHLOG_RULE_STEPS;
#if FLASHLOG_RULE_STEPS == 3
using BodyKey = uint64_t;
using BodyHash = std::hash<uint64_t>;
inline int body_length(BodyKey key) { return key >> 60; }
#else
struct BodyKey {
    int length;
    std::array<int32_t, rule_steps> operators{};
    bool operator==(const BodyKey& other) const {
        return length == other.length && operators == other.operators;
    }
    bool operator<(const BodyKey& other) const {
        return length != other.length ? length < other.length : operators < other.operators;
    }
};
struct BodyHash {
    size_t operator()(const BodyKey& key) const {
        size_t value = key.length;
        for (int i = 0; i < key.length; ++i)
            value ^= std::hash<int32_t>{}(key.operators[i]) + 0x9e3779b9 + (value << 6) + (value >> 2);
        return value;
    }
};
inline int body_length(const BodyKey& key) { return key.length; }
#endif
double proposal_weight(float value) {
    constexpr double temperature = FLASHLOG_SAMPLING_TEMPERATURE;
    if (temperature == 1.0) return static_cast<double>(value);
    if (temperature == 2.0) return std::sqrt(static_cast<double>(value));
    return std::pow(static_cast<double>(value), 1.0 / temperature);
}


struct CsrSide {
    const int32_t* col;
    const int32_t* ptr;
    const int16_t* rel;
    const int32_t* start;
    const int32_t* count;
};

struct Graph {
    CsrSide forward;
    CsrSide inverse;
    int32_t num_relations;
};

struct SourceOption {
    int32_t op;
    const int32_t* columns;
    int32_t count;
    double mass;
    double cumulative_mass;
};

struct SourceTransition {
    std::vector<SourceOption> options;
    double total_mass = 0.0;
};

struct SourceBodyCounts {
    int64_t walk_hits = 0;
    int64_t source_hits = 0;
    int64_t positive_walk_hits = 0;
    int64_t positive_source_hits = 0;
};

struct SourceWalkCounts {
    int64_t walk_hits = 0;
    int64_t positive_walk_hits = 0;
};

inline BodyKey encode_body(int length, const std::array<int32_t, rule_steps>& body) {
#if FLASHLOG_RULE_STEPS == 3
    constexpr uint64_t mask = (1ULL << 20) - 1;
    return (static_cast<uint64_t>(length) << 60) |
           ((static_cast<uint64_t>(body[0]) & mask) << 40) |
           ((static_cast<uint64_t>(body[1]) & mask) << 20) |
           (static_cast<uint64_t>(body[2]) & mask);
#else
    return BodyKey{length, body};
#endif
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::pair<const int32_t*, int32_t> directed_targets(
    const Graph& graph, int32_t source, int32_t head
) {
    const bool inverse = head >= graph.num_relations;
    const CsrSide& side = inverse ? graph.inverse : graph.forward;
    const int32_t relation = inverse ? head - graph.num_relations : head;
    int32_t left = side.ptr[source];
    int32_t right = side.ptr[source + 1];
    while (left < right) {
        const int32_t middle = left + (right - left) / 2;
        if (static_cast<int32_t>(side.rel[middle]) < relation) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    if (
        left >= side.ptr[source + 1] ||
        static_cast<int32_t>(side.rel[left]) != relation
    ) {
        return {nullptr, 0};
    }
    return {side.col + side.start[left], side.count[left]};
}

void append_source_side_options(
    const CsrSide& side,
    int32_t relation_offset,
    int32_t node,
    const float* weights,
    int32_t excluded_op,
    int64_t topk_edges,
    std::vector<SourceOption>& output
) {
    std::vector<int32_t> groups;
    groups.reserve(side.ptr[node + 1] - side.ptr[node]);
    for (int32_t group = side.ptr[node]; group < side.ptr[node + 1];
         ++group) {
        const int32_t op = static_cast<int32_t>(side.rel[group]) +
                           relation_offset;
        if (op != excluded_op && weights[op] > 0.0f && side.count[group] > 0) {
            groups.push_back(group);
        }
    }
    const auto better = [&](int32_t left, int32_t right) {
        const int32_t left_op =
            static_cast<int32_t>(side.rel[left]) + relation_offset;
        const int32_t right_op =
            static_cast<int32_t>(side.rel[right]) + relation_offset;
        if (weights[left_op] != weights[right_op]) {
            return weights[left_op] > weights[right_op];
        }
        return left_op < right_op;
    };
    // Each eligible group contributes at least one edge, so groups past this
    // exact sorted prefix cannot be visited before the edge budget is spent.
    const size_t keep = std::min<size_t>(groups.size(), topk_edges);
    if (keep < groups.size()) {
        std::partial_sort(
            groups.begin(), groups.begin() + keep, groups.end(), better
        );
        groups.resize(keep);
    } else {
        std::sort(groups.begin(), groups.end(), better);
    }
    int64_t remaining = topk_edges;
    for (const int32_t group : groups) {
        if (remaining <= 0) break;
        const int32_t op =
            static_cast<int32_t>(side.rel[group]) + relation_offset;
        const int32_t take = static_cast<int32_t>(
            std::min<int64_t>(side.count[group], remaining)
        );
        if (take > 0) {
            output.push_back(
                {op,
                 side.col + side.start[group],
                 take,
                 proposal_weight(weights[op]) * take,
                 0.0}
            );
            remaining -= take;
        }
    }
}

SourceTransition source_transition_options(
    const Graph& graph,
    int32_t node,
    const float* weights,
    int32_t identity,
    int32_t excluded_op,
    int64_t topk_edges
) {
    SourceTransition transition;
    auto& options = transition.options;
    append_source_side_options(
        graph.forward,
        0,
        node,
        weights,
        excluded_op,
        topk_edges,
        options
    );
    append_source_side_options(
        graph.inverse,
        graph.num_relations,
        node,
        weights,
        excluded_op,
        topk_edges,
        options
    );
    if (weights[identity] > 0.0f) {
        options.push_back(
            {identity,
             nullptr,
             1,
             proposal_weight(weights[identity]),
             0.0}
        );
    }
    for (auto& option : options) {
        transition.total_mass += option.mass;
        option.cumulative_mass = transition.total_mass;
    }
    return transition;
}

inline void decode_body(const BodyKey& key, int64_t* row) {
    row[0] = body_length(key);
#if FLASHLOG_RULE_STEPS == 3
    constexpr uint64_t mask = (1ULL << 20) - 1;
    row[1] = (key >> 40) & mask;
    row[2] = (key >> 20) & mask;
    row[3] = key & mask;
#else
    for (int i = 0; i < rule_steps; ++i) row[i + 1] = key.operators[i];
#endif
}

void accumulate_sampled_source(
    const Graph& graph,
    const float* attention_values,
    int64_t ranks,
    int64_t vocabulary,
    int32_t identity,
    int32_t head,
    int64_t source_index,
    int32_t source,
    int64_t walks_per_source,
    int64_t topk_edges,
    bool mask_query_head_from_source,
    int64_t seed,
    size_t local_cache_limit,
    std::unordered_map<uint64_t, SourceTransition>& cache,
    std::unordered_map<BodyKey, SourceBodyCounts, BodyHash>& totals
) {
    uint64_t source_seed = splitmix64(
        static_cast<uint64_t>(seed) ^
        (static_cast<uint64_t>(source_index) << 32) ^
        static_cast<uint32_t>(source)
    );
    std::mt19937_64 rng(source_seed);
    const auto positive_targets = directed_targets(graph, source, head);
    std::unordered_set<int32_t> positive_target_set;
    // Short groups can be searched exactly in-place, including duplicates and
    // unsorted columns. Large groups retain the existing hashed membership.
    // Dense membership for high-degree groups; graph order is never modified.
    std::vector<uint64_t> positive_bits;
    int32_t positive_min=0, positive_max=-1;
    if (positive_targets.second >= 4096) {
        const auto ends=std::minmax_element(positive_targets.first,positive_targets.first+positive_targets.second);
        positive_min=*ends.first; positive_max=*ends.second;
        const uint64_t words=(static_cast<uint64_t>(positive_max)-positive_min+64)/64;
        if (words*8 <= 16*1024*1024 && words*8 <= static_cast<uint64_t>(positive_targets.second)*16) {
            positive_bits.assign(words,0);
            for(int32_t i=0;i<positive_targets.second;++i) {
                const uint64_t x=static_cast<uint64_t>(positive_targets.first[i])-positive_min;
                positive_bits[x>>6] |= uint64_t{1}<<(x&63);
            }
        }
    }
    if (positive_targets.second > 4 && positive_bits.empty()) {
        positive_target_set.reserve(
            static_cast<size_t>(positive_targets.second) * 2 + 1
        );
        for (int32_t index = 0; index < positive_targets.second; ++index) {
            positive_target_set.insert(positive_targets.first[index]);
        }
    }
    // Source-hit counts are determined once per body after all walks from this
    // source. The counts-only sampler needs just two int64s in its hot map.
    using LocalStats = SourceWalkCounts;
    std::unordered_map<BodyKey, LocalStats, BodyHash> local;
    struct FirstTransition {
        const SourceTransition* value = nullptr;
        SourceTransition owned;
    };
    std::vector<FirstTransition> first_transitions(ranks);
    struct SpillEntry {uint64_t key=0;bool valid=false;SourceTransition value;};
    std::vector<SpillEntry> spill;

    for (int64_t walk = 0; walk < walks_per_source; ++walk) {
        const int64_t rank = walk % ranks;
        int32_t node = source;
        std::array<int32_t, rule_steps> body{};
        int length = 0;
        bool viable = true;
        for (int step = 0; step < rule_steps; ++step) {
            const SourceTransition* transition = step == 0
                ? first_transitions[rank].value : nullptr;
            SourceTransition uncached;
            if (transition == nullptr) {
                const bool exclude =
                    mask_query_head_from_source && node == source;
                const uint64_t cache_key =
                    (((static_cast<uint64_t>(static_cast<uint32_t>(node)) *
                       static_cast<uint64_t>(ranks) + rank) * rule_steps + step) << 1) |
                    static_cast<uint64_t>(exclude);
                SpillEntry* spill_entry=nullptr;
                if(step>0 && cache.size()>=local_cache_limit) {
                    if(spill.empty())spill.resize(256);
                    spill_entry=&spill[splitmix64(cache_key)&255];
                }
                if(spill_entry && spill_entry->valid && spill_entry->key==cache_key) {
                    transition=&spill_entry->value;
                } else {
                    const auto found = cache.find(cache_key);
                    if (found != cache.end()) {
                        transition = &found->second;
                    } else {
                        const float* weights = attention_values +
                            (step * ranks + rank) * vocabulary;
                        uncached = source_transition_options(
                            graph,
                            node,
                            weights,
                            identity,
                            exclude ? head : -1,
                            topk_edges
                        );
                        if (cache.size() < local_cache_limit) {
                            transition = &cache.emplace(
                                cache_key, std::move(uncached)
                            ).first->second;
                        } else {
                            transition = &uncached;
                        }
                    }
                    if(spill_entry && transition==&uncached) {
                        spill_entry->key=cache_key;spill_entry->valid=true;
                        spill_entry->value=std::move(uncached);transition=&spill_entry->value;
                    }
                }
                if (step == 0) {
                    // Cache insertion order stays unchanged. Map references
                    // survive rehash; a full/disabled cache needs local storage.
                    auto& first = first_transitions[rank];
                    if (transition == &uncached) {
                        first.owned = std::move(uncached);
                        transition = &first.owned;
                    }
                    first.value = transition;
                }
            }
            const auto& options = transition->options;
            const double total_mass = transition->total_mass;
            if (total_mass <= 0.0) {
                viable = false;
                break;
            }
            const SourceOption* selected = nullptr;

            // Sample proportionally to attention-weighted edge multiplicity.
            const double draw = std::generate_canonical<double, 64>(rng) *
                                total_mass;
            const auto selected_iterator = std::upper_bound(
                options.begin(), options.end(), draw,
                [](double value, const SourceOption& option) {
                    return value < option.cumulative_mass;
                }
            );
            selected = selected_iterator == options.end()
                ? &options.back() : &*selected_iterator;

            if (selected->op != identity) {
                std::uniform_int_distribution<int32_t> neighbor_index(
                    0, selected->count - 1
                );
                node = selected->columns[neighbor_index(rng)];
                body[length++] = selected->op;
            }
        }
        if (!viable || length == 0) continue;
        const BodyKey key = encode_body(length, body);
        auto& stats = local[key];
        ++stats.walk_hits;
        bool positive = false;
        if (!positive_bits.empty()) {
            if(node>=positive_min && node<=positive_max) {
                const uint64_t x=static_cast<uint64_t>(node)-positive_min;
                positive=(positive_bits[x>>6]>>(x&63))&1;
            }
        } else if (positive_targets.second <= 4) {
            for (int32_t index = 0; index < positive_targets.second; ++index) {
                positive |= node == positive_targets.first[index];
            }
        } else {
            positive = positive_target_set.count(node) != 0;
        }
        if (positive) {
            ++stats.positive_walk_hits;
        }

    }

    for (const auto& item : local) {
        auto& total = totals[item.first];
        total.walk_hits += item.second.walk_hits;
        total.positive_walk_hits += item.second.positive_walk_hits;

        ++total.source_hits;
        if (item.second.positive_walk_hits > 0) ++total.positive_source_hits;

    }
}

template <typename Stats>
using SourceStatsRow = std::pair<BodyKey, Stats>;

template <typename Stats>
std::vector<SourceStatsRow<Stats>> merge_and_cap_source_stats(
    std::vector<std::unordered_map<BodyKey, Stats, BodyHash>>& thread_stats,
    const std::vector<int64_t>& budgets,
    int64_t& raw_row_count
) {
    std::unordered_map<BodyKey, Stats, BodyHash> merged;
    size_t reused = thread_stats.size();

    // Integer addition is exact and order-independent. Reuse the largest
    // map's nodes and buckets.
    if (!thread_stats.empty()) {
        const auto largest = std::max_element(
            thread_stats.begin(), thread_stats.end(),
            [](const auto& left, const auto& right) {
                return left.size() < right.size();
            }
        );
        reused = static_cast<size_t>(largest - thread_stats.begin());
        merged = std::move(*largest);
    }

    for (size_t index = 0; index < thread_stats.size(); ++index) {
        if (index == reused) continue;
        const auto& local = thread_stats[index];
        for (const auto& item : local) {
            auto& total = merged[item.first];
            total.walk_hits += item.second.walk_hits;
            total.source_hits += item.second.source_hits;
            total.positive_walk_hits += item.second.positive_walk_hits;
            total.positive_source_hits += item.second.positive_source_hits;

        }
    }
    raw_row_count = static_cast<int64_t>(merged.size());

    std::vector<SourceStatsRow<Stats>> capped;
    size_t total_budget = 0;
    for (auto budget : budgets) total_budget += static_cast<size_t>(budget);
    capped.reserve(total_budget);
    std::array<std::vector<SourceStatsRow<Stats>>, rule_steps> by_length;
    for (const auto& row : merged) {
        const size_t length_index = static_cast<size_t>(body_length(row.first) - 1);
        if (budgets[length_index] > 0) by_length[length_index].push_back(row);
    }
    for (int64_t length = 1; length <= rule_steps; ++length) {
        auto& length_rows = by_length[length - 1];
        const auto better = [](const auto& left, const auto& right) {
            if (left.second.walk_hits != right.second.walk_hits) {
                return left.second.walk_hits > right.second.walk_hits;
            }
            return left.first < right.first;
        };
        const size_t keep = std::min<size_t>(
            length_rows.size(), static_cast<size_t>(budgets[length - 1])
        );
        if (keep == length_rows.size()) {
            std::sort(length_rows.begin(), length_rows.end(), better);
        } else if (keep > 64) {
            // The hit-count/key comparator is a total order. Selecting the
            // exact prefix before sorting it avoids a large partial-sort heap.
            std::nth_element(
                length_rows.begin(), length_rows.begin() + keep,
                length_rows.end(), better
            );
            std::sort(length_rows.begin(), length_rows.begin() + keep, better);
        } else {
            std::partial_sort(
                length_rows.begin(), length_rows.begin() + keep,
                length_rows.end(), better
            );
        }
        capped.insert(
            capped.end(), length_rows.begin(), length_rows.begin() + keep
        );
    }
    return capped;
}

torch::Tensor union_encoded_source_folds(
    const std::vector<std::vector<SourceStatsRow<SourceBodyCounts>>>& folds,
    int64_t head
) {
    struct Aggregate {
        BodyKey key;
        std::array<int64_t, 4> totals;
        std::array<int64_t, 4> minimums;
        size_t observed_folds = 1;
    };
    const auto smallest = std::min_element(
        folds.begin(), folds.end(), [](const auto& left, const auto& right) {
            return left.size() < right.size();
        }
    );
    if (std::all_of(folds.begin(), folds.end(), [](const auto& fold) { return fold.empty(); })) {
        return torch::empty({0, rule_steps + 9}, torch::TensorOptions().dtype(torch::kInt64));
    }
    const auto counts = [](const SourceBodyCounts& stats) {
        return std::array<int64_t, 4>{
            stats.walk_hits, stats.source_hits,
            stats.positive_walk_hits, stats.positive_source_hits
        };
    };
    // Merge capped candidates across folds without decoding intermediate rows.
    std::vector<Aggregate> stable;
    stable.reserve(smallest->size());
    std::unordered_map<BodyKey, size_t, BodyHash> lookup;
    lookup.reserve(smallest->size());
    std::array<int32_t, rule_steps> trivial_body{};
    trivial_body[0] = static_cast<int32_t>(head);
    const BodyKey trivial = encode_body(1, trivial_body);
    for (const auto& row : *smallest) {
        if (row.first == trivial) continue;
        lookup.emplace(row.first, stable.size());
        const auto values = counts(row.second);
        stable.push_back({row.first, values, values, 1});
    }
    for (auto fold = folds.begin(); fold != folds.end(); ++fold) {
        if (fold == smallest) continue;
        for (const auto& row : *fold) {
            const auto found = lookup.find(row.first);
            if (found == lookup.end()) {
                if (row.first == trivial) continue;
                const auto values = counts(row.second);
                lookup.emplace(row.first, stable.size());
                stable.push_back({row.first, values, values, 1});
                continue;
            }
            auto& aggregate = stable[found->second];
            const auto values = counts(row.second);
            for (int column = 0; column < 4; ++column) {
                aggregate.totals[column] += values[column];
                aggregate.minimums[column] = std::min(
                    aggregate.minimums[column], values[column]
                );
            }
            ++aggregate.observed_folds;
        }
    }
    for (auto& row : stable) {
        if (row.observed_folds != folds.size()) row.minimums.fill(0);
    }
    std::sort(
        stable.begin(), stable.end(), [](const auto& left, const auto& right) {
            return left.key < right.key;
        }
    );
    auto output = torch::empty(
        {static_cast<int64_t>(stable.size()), rule_steps + 9},
        torch::TensorOptions().dtype(torch::kInt64)
    );
    auto rows = output.accessor<int64_t, 2>();
    for (size_t index = 0; index < stable.size(); ++index) {
        decode_body(stable[index].key, &rows[index][0]);
        for (int column = 0; column < 4; ++column) {
            rows[index][rule_steps + 1 + column] = stable[index].totals[column];
            rows[index][rule_steps + 5 + column] = stable[index].minimums[column];
        }
    }
    return output;
}

std::vector<torch::Tensor> ground_source_folds_sampled_capped_counts(
    torch::Tensor attention,
    int64_t head,
    int64_t num_relations,
    std::vector<torch::Tensor> source_folds,
    torch::Tensor ori_col,
    torch::Tensor inv_col,
    torch::Tensor ori_ptr,
    torch::Tensor ori_rel,
    torch::Tensor ori_start,
    torch::Tensor ori_count,
    torch::Tensor inv_ptr,
    torch::Tensor inv_rel,
    torch::Tensor inv_start,
    torch::Tensor inv_count,
    int64_t walks_per_source,
    int64_t topk_edges,
    bool mask_query_head_from_source,
    int64_t seed,
    int64_t transition_cache_size,
    int64_t num_threads,
    std::vector<int64_t> budgets
) {
    TORCH_CHECK(!attention.is_cuda(), "attention must be on CPU");
    TORCH_CHECK(attention.scalar_type() == torch::kFloat32);
    TORCH_CHECK(attention.dim() == 3 && attention.size(0) == rule_steps);
    TORCH_CHECK(attention.size(2) == 2 * num_relations + 1);
    TORCH_CHECK(!source_folds.empty(), "at least one source fold is required");
    TORCH_CHECK(head >= 0 && head < 2 * num_relations);
    TORCH_CHECK(num_relations > 0 && 2 * num_relations < (1 << 20));
    TORCH_CHECK(walks_per_source > 0);
    TORCH_CHECK(topk_edges > 0);
    TORCH_CHECK(transition_cache_size >= 0);
    TORCH_CHECK(num_threads > 0);
    TORCH_CHECK(
        budgets.size() == rule_steps && std::all_of(budgets.begin(), budgets.end(), [](int64_t b) { return b >= 0; }),
        "candidate budgets must be nonnegative"
    );
    std::vector<int64_t> fold_offsets(source_folds.size() + 1, 0);
    for (size_t fold_index = 0; fold_index < source_folds.size(); ++fold_index) {
        const auto& sources = source_folds[fold_index];
        TORCH_CHECK(!sources.is_cuda(), "source folds must be on CPU");
        TORCH_CHECK(sources.scalar_type() == torch::kInt64);
        TORCH_CHECK(sources.dim() == 1);
        TORCH_CHECK(sources.is_contiguous(), "source folds must be contiguous");
        fold_offsets[fold_index + 1] =
            fold_offsets[fold_index] + sources.numel();
    }

    const Graph graph{
        {ori_col.data_ptr<int32_t>(),
         ori_ptr.data_ptr<int32_t>(),
         ori_rel.data_ptr<int16_t>(),
         ori_start.data_ptr<int32_t>(),
         ori_count.data_ptr<int32_t>()},
        {inv_col.data_ptr<int32_t>(),
         inv_ptr.data_ptr<int32_t>(),
         inv_rel.data_ptr<int16_t>(),
         inv_start.data_ptr<int32_t>(),
         inv_count.data_ptr<int32_t>()},
        static_cast<int32_t>(num_relations),
    };
    const int32_t identity = static_cast<int32_t>(2 * num_relations);
    const int64_t ranks = attention.size(1);
    const int64_t vocabulary = attention.size(2);
    const float* attention_values = attention.data_ptr<float>();
    const int64_t total_source_count = fold_offsets.back();
    const int thread_count = static_cast<int>(
        std::max<int64_t>(
            1, std::min<int64_t>(num_threads, total_source_count)
        )
    );
    const int64_t fold_count = static_cast<int64_t>(source_folds.size());
    std::vector<
        std::vector<std::unordered_map<BodyKey, SourceBodyCounts, BodyHash>>
    > fold_thread_stats(fold_count);
    for (auto& stats : fold_thread_stats) stats.resize(thread_count);

#ifdef _OPENMP
    omp_set_num_threads(thread_count);
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = 0;
#endif
        std::unordered_map<uint64_t, SourceTransition> cache;
        const size_t local_cache_limit = static_cast<size_t>(
            transition_cache_size / thread_count
        );
        const size_t sources_per_thread = static_cast<size_t>(
            (total_source_count + thread_count - 1) / thread_count
        );
        const size_t cache_reserve = std::min(
            local_cache_limit,
            sources_per_thread * static_cast<size_t>(walks_per_source) * rule_steps
        );
        if (cache_reserve > 0) cache.reserve(cache_reserve);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t task = 0; task < total_source_count; ++task) {
            const int64_t fold_index = static_cast<int64_t>(
                std::upper_bound(
                    fold_offsets.begin(), fold_offsets.end(), task
                ) - fold_offsets.begin() - 1
            );
            const int64_t source_index = task - fold_offsets[fold_index];
            const int64_t* source_values =
                source_folds[fold_index].data_ptr<int64_t>();
            accumulate_sampled_source(
                graph,
                attention_values,
                ranks,
                vocabulary,
                identity,
                static_cast<int32_t>(head),
                source_index,
                static_cast<int32_t>(source_values[source_index]),
                walks_per_source,
                topk_edges,
                mask_query_head_from_source,
                seed,
                local_cache_limit,
                cache,
                fold_thread_stats[fold_index][thread_index]
            );
        }
    }

    std::vector<std::vector<SourceStatsRow<SourceBodyCounts>>> capped_folds;
    capped_folds.reserve(source_folds.size());
    auto raw_counts = torch::empty(
        {fold_count}, torch::TensorOptions().dtype(torch::kInt64)
    );
    int64_t* raw_count_values = raw_counts.data_ptr<int64_t>();
    for (int64_t fold_index = 0; fold_index < fold_count; ++fold_index) {
        auto rows = merge_and_cap_source_stats(
            fold_thread_stats[fold_index],
            budgets,
            raw_count_values[fold_index]
        );
        capped_folds.push_back(std::move(rows));
    }
    return {
        union_encoded_source_folds(capped_folds, head),
        raw_counts,
    };
}

void ground_head_lane_batch(
    const torch::Tensor& attentions,
    const std::vector<int64_t>& heads,
    const std::vector<std::vector<torch::Tensor>>& source_folds,
    const Graph& graph,
    const std::vector<int64_t>& indices,
    const std::vector<int64_t>& source_counts,
    int64_t walks_per_source,
    int64_t topk_edges,
    bool mask_query_head_from_source,
    const std::vector<int64_t>& seeds,
    int64_t transition_cache_size,
    int64_t num_threads,
    const std::vector<int64_t>& budgets,
    std::vector<torch::Tensor>& candidates,
    std::vector<torch::Tensor>& raw_counts
) {
    struct HeadWork {
        std::vector<std::vector<std::unordered_map<BodyKey, SourceBodyCounts, BodyHash>>>
            fold_stats;
        std::vector<int64_t> fold_offsets;
        int64_t lanes;
    };
    struct LaneTask {
        size_t job;
        int64_t lane;
        int64_t begin;
        int64_t end;
    };
    std::vector<HeadWork> jobs(indices.size());
    std::vector<std::atomic<int64_t>> remaining(indices.size());
    std::vector<LaneTask> tasks;
    for (size_t job = 0; job < indices.size(); ++job) {
        const int64_t index = indices[job];

        const auto& folds = source_folds[index];
        auto& stats = jobs[job].fold_stats;
        const int64_t lanes = std::min(num_threads, source_counts[index]);
        jobs[job].lanes = lanes;
        remaining[job].store(lanes, std::memory_order_relaxed);
        auto& offsets = jobs[job].fold_offsets;
        offsets.resize(folds.size() + 1, 0);
        stats.resize(folds.size());
        for (size_t fold = 0; fold < folds.size(); ++fold) {
            stats[fold].resize(lanes);
            offsets[fold + 1] = offsets[fold] + folds[fold].numel();
        }
        // Preserve the contiguous static partition of the original CPU team:
        // each lane retains its source order and one cache across those sources.
        const int64_t base = source_counts[index] / lanes;
        const int64_t remainder = source_counts[index] % lanes;
        for (int64_t lane = 0; lane < lanes; ++lane) {
            const int64_t begin = lane * base + std::min(lane, remainder);
            tasks.push_back({job, lane, begin, begin + base + (lane < remainder)});
        }
    }
    const int64_t ranks = attentions.size(2);
    const int64_t vocabulary = attentions.size(3);
    const int32_t identity = 2 * graph.num_relations;
    const float* values = attentions.data_ptr<float>();
    const int workers = static_cast<int>(
        std::min<int64_t>(num_threads, tasks.size())
    );
    const auto finalize_head = [&](size_t job) {
        const int64_t index = indices[job];
        // The last completed lane owns finalization, including map teardown.
        auto stats = std::move(jobs[job].fold_stats);
        auto counts = torch::empty(
            {static_cast<int64_t>(stats.size())},
            torch::TensorOptions().dtype(torch::kInt64)
        );
        auto* raw = counts.data_ptr<int64_t>();
        std::vector<std::vector<SourceStatsRow<SourceBodyCounts>>> capped;
        capped.reserve(stats.size());
        for (size_t fold = 0; fold < stats.size(); ++fold) {
            capped.push_back(merge_and_cap_source_stats(
                stats[fold], budgets, raw[fold]
            ));
        }
        candidates[index] = union_encoded_source_folds(capped, heads[index]);
        raw_counts[index] = std::move(counts);
    };
#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
            const auto& task = tasks[task_index];
            {
                const int64_t index = indices[task.job];
                const auto& work = jobs[task.job];
                const size_t limit = static_cast<size_t>(
                    transition_cache_size / work.lanes
                );
                std::unordered_map<uint64_t, SourceTransition> cache;
                const size_t reserve = std::min(
                    limit, static_cast<size_t>(
                        (source_counts[index] + work.lanes - 1) / work.lanes
                    ) * static_cast<size_t>(walks_per_source) * rule_steps
                );
                if (reserve > 0) cache.reserve(reserve);
                for (int64_t source_task = task.begin; source_task < task.end; ++source_task) {
                    const size_t fold = static_cast<size_t>(std::upper_bound(
                        work.fold_offsets.begin(), work.fold_offsets.end(), source_task
                    ) - work.fold_offsets.begin() - 1);
                    const int64_t source_index = source_task - work.fold_offsets[fold];
                    const auto* sources = source_folds[index][fold].data_ptr<int64_t>();
                    accumulate_sampled_source(
                        graph, values + index * rule_steps * ranks * vocabulary,
                        ranks, vocabulary, identity, static_cast<int32_t>(heads[index]),
                        source_index, static_cast<int32_t>(sources[source_index]),
                        walks_per_source, topk_edges, mask_query_head_from_source,
                        seeds[index], limit, cache,
                        jobs[task.job].fold_stats[fold][task.lane]
                    );
                }
            }
            // Publish this lane's writes after its cache is released. The
            // acquire/release chain makes every completed lane visible to the
            // single finalizer, without waiting for unrelated heads.
            if (remaining[task.job].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                finalize_head(task.job);
            }
        }
    }
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor>
ground_head_batch_source_folds_sampled_capped_counts(
    torch::Tensor attentions,
    std::vector<int64_t> heads,
    int64_t num_relations,
    std::vector<std::vector<torch::Tensor>> source_folds,
    torch::Tensor ori_col,
    torch::Tensor inv_col,
    torch::Tensor ori_ptr,
    torch::Tensor ori_rel,
    torch::Tensor ori_start,
    torch::Tensor ori_count,
    torch::Tensor inv_ptr,
    torch::Tensor inv_rel,
    torch::Tensor inv_start,
    torch::Tensor inv_count,
    int64_t walks_per_source,
    int64_t topk_edges,
    bool mask_query_head_from_source,
    std::vector<int64_t> seeds,
    int64_t transition_cache_size,
    int64_t num_threads,
    std::vector<int64_t> budgets
) {
    TORCH_CHECK(!attentions.is_cuda(), "attentions must be on CPU");
    TORCH_CHECK(attentions.scalar_type() == torch::kFloat32);
    TORCH_CHECK(attentions.dim() == 4 && attentions.size(1) == rule_steps);
    TORCH_CHECK(attentions.is_contiguous(), "attentions must be contiguous");
    TORCH_CHECK(attentions.size(2) > 0);
    TORCH_CHECK(attentions.size(3) == 2 * num_relations + 1);
    const int64_t count = attentions.size(0);
    TORCH_CHECK(static_cast<int64_t>(heads.size()) == count);
    TORCH_CHECK(static_cast<int64_t>(seeds.size()) == count);
    TORCH_CHECK(static_cast<int64_t>(source_folds.size()) == count);
    std::vector<torch::Tensor> candidates(count);
    std::vector<torch::Tensor> raw_counts(count);
    auto seconds = torch::empty({count}, torch::TensorOptions().dtype(torch::kFloat64));
    auto* elapsed = seconds.data_ptr<double>();
    std::vector<int64_t> source_counts(count, 0);
    std::vector<int64_t> small_heads;
    std::vector<int64_t> large_heads;
    int64_t small_sources = 0;
    // Validate before entering OpenMP; do not throw validation errors inside
    // workers. Empty batches retain their existing no-op behavior.
    if (count > 0) {
        TORCH_CHECK(num_relations > 0 && 2 * num_relations < (1 << 20));
        TORCH_CHECK(walks_per_source > 0 && topk_edges > 0);
        TORCH_CHECK(transition_cache_size >= 0 && num_threads > 0);
        TORCH_CHECK(budgets.size() == rule_steps && std::all_of(budgets.begin(), budgets.end(), [](int64_t b) { return b >= 0; }));
    }
    for (int64_t index = 0; index < count; ++index) {
        TORCH_CHECK(heads[index] >= 0 && heads[index] < 2 * num_relations);
        TORCH_CHECK(!source_folds[index].empty(), "at least one source fold is required");
        for (const auto& sources : source_folds[index]) {
            TORCH_CHECK(!sources.is_cuda(), "source folds must be on CPU");
            TORCH_CHECK(sources.scalar_type() == torch::kInt64);
            TORCH_CHECK(sources.dim() == 1);
            TORCH_CHECK(sources.is_contiguous(), "source folds must be contiguous");
            source_counts[index] += sources.numel();
        }
        if (source_counts[index] > 0 && source_counts[index] < num_threads) {
            small_heads.push_back(index);
            small_sources += source_counts[index];
        } else if (source_counts[index] >= num_threads && source_counts[index] > 0) {
            large_heads.push_back(index);
        }
    }
    bool batch_large = false;
#ifdef _OPENMP
    batch_large = num_threads > 1 && !omp_get_dynamic() && !omp_in_parallel() &&
                  omp_get_max_active_levels() > 0 && omp_get_thread_limit() >= num_threads;
#endif
    const auto run_group = [&](const std::vector<int64_t>& group, int64_t total_sources) {
        const Graph graph{
            {ori_col.data_ptr<int32_t>(), ori_ptr.data_ptr<int32_t>(),
             ori_rel.data_ptr<int16_t>(), ori_start.data_ptr<int32_t>(), ori_count.data_ptr<int32_t>()},
            {inv_col.data_ptr<int32_t>(), inv_ptr.data_ptr<int32_t>(),
             inv_rel.data_ptr<int16_t>(), inv_start.data_ptr<int32_t>(), inv_count.data_ptr<int32_t>()},
            static_cast<int32_t>(num_relations),
        };
        const auto started = std::chrono::steady_clock::now();
        ground_head_lane_batch(
            attentions, heads, source_folds, graph, group, source_counts,
            walks_per_source, topk_edges, mask_query_head_from_source, seeds,
            transition_cache_size, num_threads,
            budgets, candidates, raw_counts
        );
        const double shared_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        // These are attributed wall-time contributions, not concurrent head
        // latencies. Their sum remains the measured shared region wall time.
        for (const int64_t index : group) {
            elapsed[index] = shared_seconds *
                (static_cast<double>(source_counts[index]) / total_sources);
        }
    };
    if (small_heads.size() > 1) run_group(small_heads, small_sources);
    if (batch_large) {
        // Bound retained per-head integer maps while sharing workers across
        // complete logical lanes. Singleton groups retain the original path.
        constexpr size_t large_group_size = 16;
        for (size_t begin = 0; begin < large_heads.size(); begin += large_group_size) {
            const size_t end = std::min(large_heads.size(), begin + large_group_size);
            if (end - begin < 2) continue;
            const std::vector<int64_t> group(large_heads.begin() + begin, large_heads.begin() + end);
            int64_t total_sources = 0;
            for (const int64_t index : group) total_sources += source_counts[index];
            run_group(group, total_sources);
        }
    }
    for (int64_t index = 0; index < count; ++index) {
        if (candidates[index].defined()) continue;
        const auto started = std::chrono::steady_clock::now();
        auto result = ground_source_folds_sampled_capped_counts(
            attentions[index], heads[index], num_relations, source_folds[index],
            ori_col, inv_col, ori_ptr, ori_rel, ori_start, ori_count,
            inv_ptr, inv_rel, inv_start, inv_count,
            walks_per_source, topk_edges, mask_query_head_from_source,
            seeds[index], transition_cache_size, num_threads,
            budgets
        );
        elapsed[index] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        candidates[index] = std::move(result[0]);
        raw_counts[index] = std::move(result[1]);
    }
    return {std::move(candidates), std::move(raw_counts), std::move(seconds)};
}

std::vector<torch::Tensor> sample_relation_sources_reservoir(
    torch::Tensor row_group_ptr,
    torch::Tensor group_relations,
    torch::Tensor relations,
    int64_t max_sources,
    int64_t seed
) {
    TORCH_CHECK(!row_group_ptr.is_cuda(), "row_group_ptr must be on CPU");
    TORCH_CHECK(!group_relations.is_cuda(), "group_relations must be on CPU");
    TORCH_CHECK(!relations.is_cuda(), "relations must be on CPU");
    TORCH_CHECK(row_group_ptr.scalar_type() == torch::kInt32);
    TORCH_CHECK(group_relations.scalar_type() == torch::kInt16);
    TORCH_CHECK(relations.scalar_type() == torch::kInt64);
    TORCH_CHECK(row_group_ptr.dim() == 1 && row_group_ptr.numel() > 0);
    TORCH_CHECK(group_relations.dim() == 1);
    TORCH_CHECK(relations.dim() == 1);
    TORCH_CHECK(max_sources > 0);
    TORCH_CHECK(
        row_group_ptr[-1].item<int32_t>() == group_relations.numel(),
        "row_group_ptr does not cover group_relations"
    );

    const int64_t relation_count = relations.numel();
    auto sampled = torch::full(
        {relation_count, max_sources},
        -1,
        torch::TensorOptions().dtype(torch::kInt32)
    );
    auto available = torch::zeros(
        {relation_count}, torch::TensorOptions().dtype(torch::kInt64)
    );
    if (relation_count == 0 || group_relations.numel() == 0) {
        return {sampled, available};
    }

    constexpr int32_t max_relation_id = 32767;
    std::vector<int32_t> relation_to_slot(max_relation_id + 1, -1);
    const int64_t* requested = relations.data_ptr<int64_t>();
    for (int64_t slot = 0; slot < relation_count; ++slot) {
        const int64_t relation = requested[slot];
        TORCH_CHECK(
            relation >= 0 && relation <= max_relation_id,
            "relation ID is outside int16 range"
        );
        TORCH_CHECK(
            relation_to_slot[relation] == -1,
            "relations contains duplicates"
        );
        relation_to_slot[relation] = static_cast<int32_t>(slot);
    }

    const int32_t* pointers = row_group_ptr.data_ptr<int32_t>();
    const int16_t* group_relation = group_relations.data_ptr<int16_t>();
    int32_t* output = sampled.data_ptr<int32_t>();
    int64_t* counts = available.data_ptr<int64_t>();
    const int64_t row_count = row_group_ptr.numel() - 1;
    for (int64_t row = 0; row < row_count; ++row) {
        for (int32_t group = pointers[row]; group < pointers[row + 1]; ++group) {
            const int32_t relation = static_cast<int32_t>(group_relation[group]);
            if (relation < 0) continue;
            const int32_t slot = relation_to_slot[relation];
            if (slot < 0) continue;
            const uint64_t seen = static_cast<uint64_t>(++counts[slot]);
            int64_t position = static_cast<int64_t>(seen - 1);
            if (seen > static_cast<uint64_t>(max_sources)) {
                const uint64_t random = splitmix64(
                    static_cast<uint64_t>(seed) ^
                    (static_cast<uint64_t>(relation) << 32) ^ seen
                );
                position = static_cast<int64_t>(
                    (static_cast<unsigned __int128>(random) * seen) >> 64
                );
                if (position >= max_sources) continue;
            }
            output[static_cast<int64_t>(slot) * max_sources + position] =
                static_cast<int32_t>(row);
        }
    }
    for (int64_t slot = 0; slot < relation_count; ++slot) {
        const int64_t kept = std::min<int64_t>(counts[slot], max_sources);
        const uint64_t relation = static_cast<uint64_t>(requested[slot]);
        int32_t* relation_output = output + slot * max_sources;
        for (int64_t index = kept - 1; index > 0; --index) {
            const uint64_t random = splitmix64(
                static_cast<uint64_t>(seed) ^ (relation << 32) ^
                0xd1b54a32d192ed03ULL ^ static_cast<uint64_t>(index)
            );
            const int64_t other = static_cast<int64_t>(
                (static_cast<unsigned __int128>(random) *
                 static_cast<uint64_t>(index + 1)) >> 64
            );
            std::swap(relation_output[index], relation_output[other]);
        }
    }
    return {sampled, available};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "ground_head_batch_source_folds_sampled_capped_counts",
        &ground_head_batch_source_folds_sampled_capped_counts,
        "Graph-guided rule sampling with source folds and ordered head batches",
        pybind11::call_guard<pybind11::gil_scoped_release>()
    );
    module.def(
        "sample_relation_sources_reservoir",
        &sample_relation_sources_reservoir,
        "Uniform per-relation Train-source reservoir sampling",
        pybind11::call_guard<pybind11::gil_scoped_release>()
    );
}
