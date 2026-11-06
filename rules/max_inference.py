"""MAX rule inference through the common SUM/MAX execution engine."""
from .shared_inference import score as shared_score

def score(ev, sources, rules, csr, relations, entities, chunk_size, final_max=False,
          dense_budget_bytes=None, prefix_budget_bytes=128*1024**2, stats=None):
    return shared_score(ev,sources,rules,csr,relations,entities,chunk_size,
        final_max=final_max,reduction='max',dense_budget_bytes=dense_budget_bytes,
        prefix_budget_bytes=prefix_budget_bytes,stats=stats)
