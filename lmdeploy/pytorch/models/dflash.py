# Copyright (c) OpenMMLab. All rights reserved.
"""DFlash draft model for speculative decoding.

DFlash (Block Diffusion for Flash Speculative Decoding) uses a draft model
that predicts multiple tokens in parallel using hidden states from the target
model's intermediate layers.

Reference: https://github.com/thu-coai/DFlash
"""

from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import add_prefix
from .utils.cudagraph import CudaGraphMixin, CudaGraphMeta

__all__ = ['DFlashDraftModel', 'DFlashAttention', 'DFlashDecoderLayer']


class DFlashAttention(nn.Module):
    """Multi-headed attention for DFlash that takes target_hidden as context.

    Key concatenation mechanism:
    - Query from draft token hidden states
    - Key/Value from concatenation of target_hidden (context) + draft hidden states
    """

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_attention_heads)
        quantization_config = getattr(config, 'quantization_config', None)

        # qkv projections - using LMDeploy primitives
        self.qkv_proj = build_qkv_proj(
            self.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_size=self.head_dim,
            bias=config.attention_bias if hasattr(config, 'attention_bias') else False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('qkv_proj', prefix),
        )

        self.o_proj = build_o_proj(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias if hasattr(config, 'attention_bias') else False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('o_proj', prefix),
        )

        # q, k norm
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype, device=device)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_dim,
            sliding_window=getattr(config, 'sliding_window', None),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: tuple[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Draft token hidden states [bsz, q_len, hidden_size]
            target_hidden: Target model hidden states [bsz, ctx_len, hidden_size]
            rotary_pos_emb: Rotary position embeddings
            past_key_value: Past key value cache
            attn_metadata: Attention metadata

        Returns:
            Attention output [bsz, q_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from draft tokens
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, _, _ = self.qkv_proj.split_qkv(qkv_states)

        query_states = self.q_norm(query_states)

        # K and V from concatenation of target (context) and draft (noise)
        k_ctx_states = self.qkv_proj.k_proj(target_hidden) if hasattr(self.qkv_proj, 'k_proj') else None
        k_ctx = k_ctx_states if k_ctx_states is not None else self.qkv_proj(target_hidden)

        k_draft_states = self.qkv_proj(hidden_states)
        k_draft = k_draft_states

        # Apply q, k norm
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        # For now use simple projection
        k = torch.cat([target_hidden, hidden_states], dim=1)

        # Apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states.flatten(0, -2),
            k.flatten(0, -2),
            cos,
            sin,
            inplace=True,
        )

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = k.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            v,
            past_key_value[0] if past_key_value else None,
            past_key_value[1] if past_key_value else None,
            attn_metadata,
            k_scales_zeros=None if past_key_value is None or len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if past_key_value is None or len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return self.o_proj(attn_output)


class DFlashDecoderLayer(nn.Module):
    """DFlash decoder layer that uses target_hidden as context."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = DFlashAttention(config, layer_idx, dtype=dtype, device=device, prefix=add_prefix('self_attn', prefix))

        # build MLP using LMDeploy primitives
        self.gate_up_proj = build_gateup_linear(
            self.hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            prefix=add_prefix('gate_up_proj', prefix),
        )
        self.act_fn = SiluAndMul(inplace=True)
        self.down_proj = build_down_linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('down_proj', prefix),
        )

        # build layer norms
        self.input_layernorm = RMSNorm(
            self.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', prefix),
        )

    def forward(
        self,
        target_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: list[torch.FloatTensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """
        Args:
            target_hidden: Target model hidden states [bsz, ctx_len, hidden_size]
            hidden_states: Draft token hidden states [bsz, q_len, hidden_size]
            rotary_pos_emb: Rotary position embeddings
            past_key_value: Past key value cache
            attn_metadata: Attention metadata

        Returns:
            Output hidden states [bsz, q_len, hidden_size]
        """
        # Self-attention with target_hidden as context
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashDraftModel(nn.Module, CudaGraphMixin):
    """DFlash draft model wrapper for LMDeploy.

    This model takes:
    - target_hidden_states: Concatenated hidden states from target model's intermediate layers
    - input_ids: Draft token IDs (mask tokens or previously generated tokens)
    - position_ids: Position IDs for draft tokens

    And outputs hidden states for computing draft token logits.

    Note: DFlash shares embed_tokens and lm_head with the target model,
    as the checkpoint only contains intermediate layer weights.
    """

    packed_modules_mapping = {
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    # Flags for weight sharing with target model
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype

        # Get target layer IDs from dflash config
        dflash_cfg = getattr(config, 'dflash_config', None)
        if dflash_cfg is not None and isinstance(dflash_cfg, dict):
            self.target_layer_ids = dflash_cfg.get('target_layer_ids', [1, 10, 19, 28, 37])
            self.mask_token_id = dflash_cfg.get('mask_token_id', 151669)
            self.block_size = getattr(config, 'block_size', 16)
        else:
            self.target_layer_ids = [1, 10, 19, 28, 37]
            self.mask_token_id = 151669
            self.block_size = 16

        self.num_aux_layers = len(self.target_layer_ids)

        quantization_config = getattr(config, 'quantization_config', None)

        # Decoder layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config, layer_idx, dtype=dtype, device=device,
                             prefix=add_prefix(f'layers.{layer_idx}', prefix))
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('norm', prefix),
        )

        # FC layer to project target_hidden from concatenated aux states
        # to draft model hidden_size
        from lmdeploy.pytorch.nn.linear import build_colwise_linear
        self.target_hidden_proj = build_colwise_linear(
            self.num_aux_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            prefix=add_prefix('target_hidden_proj', prefix),
        )

        self.target_hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps,
                                         dtype=dtype, device=device)

        # Rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config, device=device)

        # Embeddings and lm_head are shared with target model
        self.embed_tokens = None
        self.lm_head = None

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, embed_tokens: nn.Embedding):
        """Set embed tokens."""
        self.embed_tokens = embed_tokens

    def set_lm_head(self, lm_head: nn.Linear):
        """Set lm head."""
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        DFlash forward pass.

        Args:
            input_ids: Draft token IDs [batch, num_spec_tokens]
            position_ids: Position IDs [batch, num_spec_tokens]
            target_hidden_states: Concatenated hidden states from target model
                [batch, 1, num_aux_layers * hidden_size]
            past_key_values: Past key values for cache
            attn_metadata: Attention metadata
            inputs_embeds: Pre-computed input embeddings

        Returns:
            Output hidden states [batch, num_spec_tokens, hidden_size]
        """
        # Embed input tokens
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Reshape and project target_hidden_states
        # [batch, 1, num_aux_layers * hidden_size] -> [batch, 1, hidden_size]
        batch_size = target_hidden_states.shape[0]
        target_hidden = target_hidden_states.view(batch_size, 1, self.num_aux_layers, -1)
        target_hidden = target_hidden.mean(dim=2)  # Simple average of aux layers

        target_hidden = self.target_hidden_proj(target_hidden_states)
        target_hidden = self.target_hidden_norm(target_hidden)

        # Rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # Pass through decoder layers
        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values else None
            hidden_states = layer(
                target_hidden=target_hidden,
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_kv,
                attn_metadata=attn_metadata,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        target_hidden_states = context.target_hidden_states

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            target_hidden_states=target_hidden_states,
        )

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(
            1, 1, self.num_aux_layers * self.config.hidden_size, dtype=self.dtype)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: torch.Tensor, **kwargs):
        """Fill cudagraph buffers from forward inputs."""
        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, input_ids=input_ids, **kwargs)

        input_buffers = graph_meta.input_buffers
        target_hidden_states = kwargs.get('target_hidden_states')
        assert target_hidden_states is not None
        input_buffers['target_hidden_states'][:, :, :] = target_hidden_states
        new_inputs['target_hidden_states'] = input_buffers['target_hidden_states']
        return new_inputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""
        stacked_params_mapping = [
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
