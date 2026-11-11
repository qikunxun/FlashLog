"""Dataset-independent shared grounding for SUM and MAX rule inference.

Channels are normalized only after all rules have contributed. This implements
scalar rule inference, not the model's intermediate propagation normalizers.
"""
from collections import OrderedDict, Counter

import torch


@torch.inference_mode()
def score(ev, sources, rules, csr, relations, entities, chunk_size,
          final_max=False, reduction="sum", boolean_paths=False, dense_budget_bytes=None,
          prefix_budget_bytes=512 * 1024**2, stats=None,
          source_frontier_budget_bytes=64 * 1024**2):
    if reduction not in ('sum','max'):raise ValueError('Unknown reduction')
    boolean_paths = boolean_paths or reduction == 'max'
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    stats = {} if stats is None else stats
    for name in ('ground_calls', 'prefix_hits', 'chunks', 'oom_retries'):
        stats.setdefault(name, 0)
    batch = sources.numel()
    shape = (batch, entities)
    empty = lambda: torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long, device=sources.device),
        torch.empty(0, device=sources.device), shape, is_coalesced=True)
    if not rules:
        return empty()
    original_rules = rules
    original_rule_count = len(rules)
    first_degrees = []
    stats['rules_before'] = original_rule_count
    if len(csr) >= 16:
        positions = torch.stack((sources, sources + 1), dim=1).reshape(-1)
        bounds = torch.stack((csr[8][positions], csr[12][positions])).cpu().tolist()
        parts = []
        for direction, (ri, ci) in enumerate(((9, 11), (13, 15))):
            for i in range(batch):
                start, end = bounds[direction][2*i:2*i+2]
                rels = csr[ri][start:end].long() + direction * relations
                counts = csr[ci][start:end].long()
                parts.append(torch.stack((rels, counts), dim=1))
        first_degrees = torch.cat(parts).cpu().tolist() if parts else []
        available = {rel for rel, count in first_degrees if count > 0}
        available.add(2 * relations)
        rules = [r for r in rules if not r['body'] or r['body'][0] in available]
    stats['rules_after'] = len(rules)
    if not rules:
        return empty()
    first_counts = Counter(r['body'][0] for r in rules if r['body'])
    max_first_degree = max((count for rel, count in first_degrees
                            if rel in first_counts), default=0)
    # Bound the replicated first-hop COO frontier using actual CSR degrees.
    # Large frontiers benefit from independent source execution and prefix reuse;
    # no relation IDs, dataset names or empirical score thresholds are involved.
    frontier_bytes = max((count * min(chunk_size, first_counts.get(rel, 0)) * 28
                          for rel, count in first_degrees), default=0)
    stats['first_frontier_bytes'] = frontier_bytes
    if batch > 1 and max_first_degree * 28 > source_frontier_budget_bytes:
        stats['source_splits'] = stats.get('source_splits', 0) + 1
        indices, values = [], []
        for row in range(batch):
            part = score(ev, sources[row:row + 1], original_rules, csr, relations,
                         entities, chunk_size, final_max=final_max, reduction=reduction,
                         boolean_paths=boolean_paths, dense_budget_bytes=dense_budget_bytes,
                         prefix_budget_bytes=prefix_budget_bytes, stats=stats,
                         source_frontier_budget_bytes=source_frontier_budget_bytes).coalesce()
            ix = part.indices().clone()
            ix[0] += row
            indices.append(ix)
            values.append(part.values())
        return torch.sparse_coo_tensor(torch.cat(indices, dim=1), torch.cat(values),
                                       shape, is_coalesced=True)
    widths = {len(r['rank_weights']) for r in rules}
    if len(widths) != 1 or next(iter(widths)) < 1:
        raise ValueError('nonempty rank_weights must have matching widths')
    ranks = next(iter(widths))
    weights = torch.tensor([r['rank_weights'] for r in rules],
                           dtype=torch.float32, device=sources.device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('MAX rule weights must be finite and nonnegative')

    # Use reclaimable allocator memory, not only driver-reported free memory.
    # Dense accumulation avoids repeated sparse merges for sufficiently large banks.
    # Large entity vocabularies use bounded sparse accumulators instead.
    dense = None
    if dense_budget_bytes is None:
        dense_budget_bytes = ((torch.cuda.mem_get_info(sources.device)[0] + torch.cuda.memory_reserved(sources.device) - torch.cuda.memory_allocated(sources.device)) // 3
                              if sources.is_cuda else 256 * 1024**2)
    if ranks * batch * entities * 4 <= dense_budget_bytes and (entities <= 100000 or original_rule_count >= 1024
                                                   or frontier_bytes > source_frontier_budget_bytes):
        try:
            dense = torch.zeros((ranks, batch * entities), device=sources.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    sparse = [None] * ranks
    packed = None
    packed_sum = reduction == "sum" and not final_max
    stats['dense_accumulation'] = dense is not None
    cache = OrderedDict()
    cache_bytes = 0
    identity = 2 * relations
    # Multi-source batches already amortize propagation launches. Splitting them
    # into individual prefix operations defeats that vectorization.
    prefix_counts = (Counter(tuple(r['body'][:n]) for r in rules
                             for n in range(1, len(r['body'])))
                     if batch == 1 and (max_first_degree >= 32 or len(csr) < 16) else Counter())
    prefix_depth_limit = max((len(r['body']) for r in rules), default=0) - 1
    depths = [(min(len(r['body']), max(0, prefix_depth_limit))
               if batch == 1 and frontier_bytes > source_frontier_budget_bytes
               else max([0] + [n for n in range(1, len(r['body']))
                               if prefix_counts[tuple(r['body'][:n])] >= 4]))
              for r in rules]
    use_prefix = any(depths)
    remaining = [tuple(r['body'][n:]) for r, n in zip(rules, depths)]
    max_steps = max(map(len, remaining), default=0)
    operators = torch.tensor([body + (identity,) * (max_steps - len(body))
                              for body in remaining], device=sources.device,
                             dtype=torch.long)
    stats['prefix_enabled'] = use_prefix

    def boolean(state):
        state = state.coalesce()
        return torch.sparse_coo_tensor(state.indices(), (state.values() > 0).to(state.dtype),
                                       state.shape, is_coalesced=True)

    def advance(state, ops):
        stats['ground_calls'] += 1
        try:
            out = ev.vectorized_operation_sparse3d_fixed_relation(state, ops, csr, relations)
        except torch.cuda.OutOfMemoryError:
            if batch != 1 or state.shape[1] != 1:
                raise
            torch.cuda.empty_cache()
            out = ev.advance_single_rule_state_memory_bounded(state, ops, csr, relations)
        # Multiple graph paths are Boolean evidence, never multiplicity weights.
        return boolean(out) if boolean_paths else out.coalesce()

    def prefix_state(prefix):
        nonlocal cache_bytes
        if prefix in cache:
            stats['prefix_hits'] += 1
            cache.move_to_end(prefix)
            return cache[prefix]
        if not prefix:
            return ev.initial_rule_state(sources, 1, entities)
        state=prefix_state(prefix[:-1])
        if state._nnz() and prefix[-1] != identity:
            state=advance(state,torch.full((batch,1),prefix[-1],device=sources.device,dtype=torch.long))
        size=state._nnz()*(24+state.element_size())
        if prefix_counts[prefix]>1 and size<=prefix_budget_bytes:
            while cache and cache_bytes+size>prefix_budget_bytes:
                _,old=cache.popitem(last=False)
                cache_bytes-=old._nnz()*(24+old.element_size())
            cache[prefix]=state;cache_bytes+=size
        return state

    def ground(chunk):
        if use_prefix:
            states = [prefix_state(tuple(r['body'][:n]))
                      for r, n in zip(chunk, depths[offset:offset + len(chunk)])]
            # With one source, rule-major concatenation is already COO sorted.
            endpoints = torch.cat([st.indices()[2] for st in states])
            levels = torch.cat([torch.full_like(st.indices()[2], i)
                                for i, st in enumerate(states)])
            indices = torch.stack((torch.zeros_like(endpoints), levels, endpoints))
            state = torch.sparse_coo_tensor(indices, torch.cat([st.values() for st in states]),
                                           (batch, len(chunk), entities), is_coalesced=True)
        else:
            state = ev.initial_rule_state(sources, len(chunk), entities)
        steps = max(map(len, remaining[offset:offset + len(chunk)]), default=0)
        ops = operators[offset:offset + len(chunk)]
        for step in range(steps):
            if not state._nnz():
                break
            state = advance(state, ops[:, step].unsqueeze(0).expand(batch, -1))
        return state

    offset, size = 0, chunk_size
    while offset < len(rules):
        chunk = rules[offset:offset + size]
        state = None
        failed = False
        try:
            state = ground(chunk)
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:
            del state
            cache.clear()
            cache_bytes = 0
            torch.cuda.empty_cache()
            stats['oom_retries'] += 1
            if size == 1:
                raise torch.cuda.OutOfMemoryError('MAX grounding needs a smaller query batch')
            size = max(1, size // 2)
            continue
        w = weights[offset:offset + len(chunk)]
        if dense is not None:
            ix = state.indices()
            keys = ix[0] * entities + ix[2]
            for rank in range(ranks):
                values=state.values()*w[ix[1],rank]
                if reduction=='max':dense[rank].scatter_reduce_(0,keys,values,reduce='amax',include_self=True)
                else:dense[rank].index_add_(0,keys,values)
        elif packed_sum:
            # One grouped reduction/merge for all channels instead of one per
            # channel. The grouped normalizer retains independent channel mass.
            part = ev.reduce_rule_state_by_rank(state, w, boolean=boolean_paths)
            packed = ev.add_sparse(packed, part)
        else:
            # Stage all channels before committing; never partially replay a chunk.
            parts = [(ev.max_reduce_rule_state(state,w[:,rank]) if reduction=='max' else ev.reduce_rule_state(state,w[:,rank],boolean=boolean_paths)) for rank in range(ranks)]
            merged = [(ev.max_sparse if reduction=='max' else ev.add_sparse)(old,part) for old,part in zip(sparse,parts)]
            sparse = merged
        offset += len(chunk)
        stats['chunks'] += 1
    cache.clear()
    if dense is None and packed_sum:
        return ev.normalize_and_sum_rank_scores(packed)
    if dense is not None:
        # Normalize each channel with the same sparse row reducer, then combine
        # into a flat buffer. Avoid repeatedly sorting large endpoint unions.
        total = torch.zeros(batch * entities, device=sources.device)
        for rank in range(ranks):
            flat = dense[rank]
            keys = flat.nonzero().flatten()
            part = torch.sparse_coo_tensor(torch.stack((keys // entities, keys % entities)),
                                          flat[keys], shape, is_coalesced=True)
            part = ev.normalize_sparse_rows(part)
            if final_max:
                total[keys] = torch.maximum(total[keys], part.values())
            else:
                total[keys] += part.values()
        keys = total.nonzero().flatten()
        return torch.sparse_coo_tensor(torch.stack((keys // entities, keys % entities)),
                                       total[keys], shape, is_coalesced=True)
    output = None
    for rank in range(ranks):
        part = sparse[rank]
        sparse[rank] = None
        part = ev.normalize_sparse_rows(part)
        output = (ev.max_sparse if final_max else ev.add_sparse)(output, part)
    return output
