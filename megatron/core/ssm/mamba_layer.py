# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormInterface
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


class LayerNormBuilder(Protocol):
    """A protocol showing how MambaLayer expects to construct its LayerNorm."""

    def __call__(self, config: TransformerConfig, hidden_size: int, /) -> LayerNormInterface: ...


@dataclass
class MambaLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    norm: LayerNormBuilder = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaLayer(GraphableMegatronModule):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
    ):
        """Initialize Mamba Layer."""
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for MambaLayer"

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )
        self.norm = submodules.norm(self.config, self.config.hidden_size)
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the mamba layer for cudagraphs."""
        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if not self.config.cuda_graph_scope or CudaGraphScope.mamba in self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = apply_module(self.norm)(hidden_states)

        mixer_out_with_bias = self.mixer(
            hidden_states, inference_context=inference_context, packed_seq_params=packed_seq_params
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mamba_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mixer_out_with_bias, residual, self.hidden_dropout)

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Include dummy PackedSeqParams in static inputs so TE captures the packed-sequence
        code path (THD FlashAttention, Mamba with seq_idx).
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        from megatron.training import get_args

        if getattr(get_args(), 'sft', False):
            self._cuda_graph_seq_length = seq_length
            dummy_psp, self._cuda_graph_psp_buffers = (
                PackedSeqParams.create_dummy_for_cuda_graph(seq_length)
            )
            self._cuda_graph_psp = dummy_psp
            # NOTE: do NOT put dummy_psp in static_inputs — TE's make_graphed_callables
            # calls .shape on every sample_kwarg and PackedSeqParams is not a tensor.

            # Pre-allocate seq_idx buffer for _ssm_training.
            # After pre_conv_ssm all_to_all, the sequence dimension = seq_length * cp_size.
            # _create_packed_seq_idx creates tensors dynamically (torch.tensor, torch.cat,
            # torch.repeat_interleave) which are forbidden inside CUDA graph capture.
            # Solution: pre-allocate the buffer here; fill it in _te_cuda_graph_replay
            # (outside the graph); _create_packed_seq_idx short-circuits when seq_idx is set.
            mamba_cp_size = self.mixer.cp.cp_size
            total_tokens = (seq_length // self.config.context_parallel_size) * mamba_cp_size
            seq_idx_buf = torch.zeros(
                1, total_tokens, dtype=torch.int32, device=torch.cuda.current_device()
            )
            self._cuda_graph_psp_buffers['seq_idx'] = seq_idx_buf
            dummy_psp.seq_idx = seq_idx_buf

            # Fix cu_seqlens to reflect total_tokens seen by Mamba SSM after CP all_to_all.
            # pre_conv_ssm gathers the sequence dim: [seq_length/cp, b, d] → [seq_length, b, d/cp]
            # _undo/_redo_attention_load_balancing require cu_seqlens[-1] == total_tokens.
            # create_dummy_for_cuda_graph used seq_length (per-rank) not full_seq — correct here.
            if mamba_cp_size > 1:
                for key in (
                    'cu_seqlens_q',
                    'cu_seqlens_kv',
                    'cu_seqlens_q_padded',
                    'cu_seqlens_kv_padded',
                ):
                    self._cuda_graph_psp_buffers[key][1:] = total_tokens
                dummy_psp.max_seqlen_q = total_tokens
                dummy_psp.max_seqlen_kv = total_tokens
                dummy_psp.max_seqlen_q_tensor.fill_(total_tokens)
                dummy_psp.max_seqlen_kv_tensor.fill_(total_tokens)

        return static_inputs

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Inject dummy PSP for CUDA graph capture so Mamba captures the packed-seq code path."""
        # PSP is NOT in static_inputs/sample_kwargs because TE's make_graphed_callables
        # calls .shape on every kwarg value, which fails for non-tensor dataclasses.
        if hasattr(self, '_cuda_graph_psp') and kwargs.get('packed_seq_params') is None:
            kwargs = dict(kwargs)
            kwargs['packed_seq_params'] = self._cuda_graph_psp
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for Mamba layer using TE interface.
        Passes PackedSeqParams through to the TE graphed callable so its tensor fields
        (cu_seqlens, max_seqlen tensors) are copied into the captured graph buffers.
        Non-tensor int fields are set to capture-time constants.
        """
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        psp = kwargs.get('packed_seq_params')
        if psp is not None and hasattr(self, '_cuda_graph_psp_buffers'):
            bufs = self._cuda_graph_psp_buffers
            target_len = bufs['cu_seqlens_q'].shape[0]  # = CUDA_GRAPH_MAX_PACKED_SEQS + 1

            # Pad variable-length cu_seqlens to fixed capture size, copy in-place
            bufs['cu_seqlens_q'].copy_(PackedSeqParams.pad_cu_seqlens(psp.cu_seqlens_q, target_len))
            bufs['cu_seqlens_kv'].copy_(
                PackedSeqParams.pad_cu_seqlens(psp.cu_seqlens_kv, target_len)
            )
            if psp.cu_seqlens_q_padded is not None:
                bufs['cu_seqlens_q_padded'].copy_(
                    PackedSeqParams.pad_cu_seqlens(psp.cu_seqlens_q_padded, target_len)
                )
            if psp.cu_seqlens_kv_padded is not None:
                bufs['cu_seqlens_kv_padded'].copy_(
                    PackedSeqParams.pad_cu_seqlens(psp.cu_seqlens_kv_padded, target_len)
                )
            if psp.max_seqlen_q_tensor is not None:
                bufs['max_seqlen_q_tensor'].copy_(psp.max_seqlen_q_tensor)
            if psp.max_seqlen_kv_tensor is not None:
                bufs['max_seqlen_kv_tensor'].copy_(psp.max_seqlen_kv_tensor)

            # Set int constants on dummy psp (captured as Python constants in graph)
            self._cuda_graph_psp.max_seqlen_q = self._cuda_graph_seq_length
            self._cuda_graph_psp.max_seqlen_kv = self._cuda_graph_seq_length

            # Pre-compute seq_idx outside the graph (torch.tensor/cat/repeat_interleave are
            # forbidden inside CUDA graph capture). Copy into the pre-allocated buffer so the
            # captured graph sees a fixed-address tensor with the correct content.
            if 'seq_idx' in bufs:
                total_tokens = bufs['seq_idx'].shape[1]
                real_seq_idx = self.mixer._create_packed_seq_idx(psp, total_tokens)
                bufs['seq_idx'].copy_(real_seq_idx)

            # Replace real psp with fixed-size dummy psp (buffers now updated in-place)
            kwargs = dict(kwargs)
            kwargs['packed_seq_params'] = self._cuda_graph_psp

        kwargs_filtered = {
            k: v
            for k, v in kwargs.items()
            if v is None or isinstance(v, torch.Tensor) or isinstance(v, PackedSeqParams)
        }

        cg_index = getattr(self, 'current_microbatch', 0) % len(self.cuda_graphs)
        cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(
            *args, **kwargs_filtered
        )
        for hook, hook_args in self.cuda_graph_manual_hooks:
            hook(*hook_args)
        return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return True
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs.get('inference_context') is not None
            and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
        ):
            context = kwargs['inference_context']
            using_cuda_graph = (context.is_static_batching() and context.is_decode_only()) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False
