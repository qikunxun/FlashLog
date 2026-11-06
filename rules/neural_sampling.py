"""Bounded temporary storage for NeuralLP's normalized sampling proposal."""
import numpy as np


def sampling_proposal(attentions):
    """Preserve float64 row reductions and division before float32 conversion.

    Work one head at a time: the expanded temporal-branch tensor can otherwise
    require several simultaneous whole-batch float64 copies.
    """
    values = np.asarray(attentions)
    result = np.empty(values.shape, dtype=np.float32)
    for head in range(len(values)):
        block = np.asarray(values[head], dtype=np.float64)
        totals = block.sum(axis=-1, keepdims=True)
        normalized = np.divide(block, totals, out=np.zeros_like(block), where=totals > 0)
        result[head] = normalized
    return result
