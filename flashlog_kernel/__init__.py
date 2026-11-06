"""CSR construction and dense/sparse FlashLog operators."""

from .flashlog_kernel import (
    FlashLogFunction,
    _compute_active_nodes,
    build_csr_structure,
    prepare_csr_masks,
    vectorized_operation_csr,
    vectorized_operation_sparse3d_csr_sum,
    vectorized_operation_sparse3d_csr_sum_norm,
    vectorized_operation_sparse3d_fixed_relation,
)

__all__ = [
    "FlashLogFunction",
    "_compute_active_nodes",
    "build_csr_structure",
    "prepare_csr_masks",
    "vectorized_operation_csr",
    "vectorized_operation_sparse3d_csr_sum",
    "vectorized_operation_sparse3d_csr_sum_norm",
    "vectorized_operation_sparse3d_fixed_relation",
]
