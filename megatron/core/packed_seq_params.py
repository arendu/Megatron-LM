# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# Maximum number of packed sequences supported by CUDA graph capture.
# cu_seqlens tensors are padded to this length + 1 for fixed-shape graph inputs.
CUDA_GRAPH_MAX_PACKED_SEQS: int = 2048


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    max_seqlen_q_tensor: Tensor = None
    max_seqlen_kv_tensor: Tensor = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    # Pre-computed seq_idx for Mamba CUDA graph capture. When set, _create_packed_seq_idx
    # returns this directly, avoiding torch.tensor/torch.cat allocations inside the graph.
    seq_idx: Optional[Tensor] = None

    @staticmethod
    def pad_cu_seqlens(cu_seqlens: Tensor, target_len: int) -> Tensor:
        """Pad cu_seqlens to a fixed length using the last element as fill value."""
        actual_len = cu_seqlens.shape[0]
        if actual_len >= target_len:
            return cu_seqlens[:target_len]
        padded = cu_seqlens.new_empty(target_len)
        padded[:actual_len] = cu_seqlens
        padded[actual_len:] = cu_seqlens[-1]
        return padded

    @classmethod
    def create_dummy_for_cuda_graph(
        cls, seq_length: int, max_seqs: int = CUDA_GRAPH_MAX_PACKED_SEQS
    ) -> Tuple['PackedSeqParams', Dict[str, Tensor]]:
        """Create a dummy PackedSeqParams for CUDA graph capture.

        Returns the dummy PSP and a dict of tensor buffer references that can be
        updated via copy_() during graph replay.
        """
        cu_seqlens_len = max_seqs + 1
        device = torch.cuda.current_device()
        dtype = torch.int32

        cu_seqlens_q = torch.zeros(cu_seqlens_len, dtype=dtype, device=device)
        cu_seqlens_q[1:] = seq_length
        cu_seqlens_kv = cu_seqlens_q.clone()
        cu_seqlens_q_padded = cu_seqlens_q.clone()
        cu_seqlens_kv_padded = cu_seqlens_q.clone()
        max_seqlen_q_tensor = torch.tensor([seq_length], dtype=dtype, device=device)
        max_seqlen_kv_tensor = torch.tensor([seq_length], dtype=dtype, device=device)

        psp = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=seq_length,
            max_seqlen_kv=seq_length,
            max_seqlen_q_tensor=max_seqlen_q_tensor,
            max_seqlen_kv_tensor=max_seqlen_kv_tensor,
        )
        buffers = {
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_kv': cu_seqlens_kv,
            'cu_seqlens_q_padded': cu_seqlens_q_padded,
            'cu_seqlens_kv_padded': cu_seqlens_kv_padded,
            'max_seqlen_q_tensor': max_seqlen_q_tensor,
            'max_seqlen_kv_tensor': max_seqlen_kv_tensor,
        }
        return psp, buffers
