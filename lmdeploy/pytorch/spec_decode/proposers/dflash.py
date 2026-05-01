# Copyright (c) OpenMMLab. All rights reserved.
"""DFlash proposer for speculative decoding."""

import torch

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.strategies.base.model_agent import ExtraInputs

from ...config import SpecDecodeConfig
from ..proposers.base import BaseSpecProposer, SPEC_PROPOSERS

__all__ = ['DFlashProposer']


@SPEC_PROPOSERS.register_module(name='dflash')
class DFlashProposer(BaseSpecProposer):
    """DFlash proposer for block diffusion speculative decoding.

    DFlash uses hidden states from 5 intermediate layers of the target model
    to generate multiple tokens in parallel through a diffusion process.

    Config from draft model:
    - target_layer_ids: [1, 10, 19, 28, 37] (indices of target model layers to use)
    - mask_token_id: 151669 (special token for diffusion masking)
    - num_target_layers: 36 (total layers in target model)
    """

    def __init__(self, specdecode_config: SpecDecodeConfig, device: str = 'cuda'):
        super().__init__(specdecode_config, device=device)
        self.target_layer_ids = None
        self.mask_token_id = None
        self.num_target_layers = None
        self.num_aux_layers = 5  # Number of target layers used
        self.hidden_size = None

    def build_model(self, empty_init: bool, target_model=None, build_model_ctx=None):
        """Build draft model with weight sharing."""
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')

        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)

        # Read config from draft model
        hf_config = self.specdecode_config.model_config.hf_config
        dflash_cfg = getattr(hf_config, 'dflash_config', {})
        self.target_layer_ids = dflash_cfg.get('target_layer_ids', [1, 10, 19, 28, 37])
        self.mask_token_id = dflash_cfg.get('mask_token_id', 151669)
        self.num_target_layers = getattr(hf_config, 'num_target_layers', 36)
        self.num_aux_layers = len(self.target_layer_ids)

        # Set hidden size from draft model config
        self.hidden_size = getattr(hf_config, 'hidden_size', 2048)

        # Share embed_tokens and lm_head from target model.
        # DFlash draft model doesn't have its own embedding or lm_head.
        # The draft model may be wrapped (e.g., graph_runner -> model),
        # so resolve the inner model first.
        if target_model is None:
            logger.warning('DFlashProposer: target_model is None, skipping weight sharing.')
            return

        inner_model = self.model
        while hasattr(inner_model, 'model'):
            inner_model = inner_model.model

        logger.info('DFlash: sharing embed_tokens and lm_head from target model.')

        # embed_tokens: always use target model's embedding layer
        if hasattr(target_model, 'get_input_embeddings'):
            inner_model.embed_tokens = target_model.get_input_embeddings()
        elif hasattr(target_model, 'model') and hasattr(target_model.model, 'get_input_embeddings'):
            inner_model.embed_tokens = target_model.model.get_input_embeddings()
        else:
            logger.warning('DFlashProposer: cannot find target embed_tokens, keeping draft default.')

        # lm_head: always use target model's logits projection
        if hasattr(target_model, 'get_logits'):
            inner_model.lm_head = target_model
        elif hasattr(target_model, 'lm_head'):
            inner_model.lm_head = target_model.lm_head
        else:
            logger.warning('DFlashProposer: cannot find target lm_head, keeping draft default.')

    def get_outputs(self, model_outputs: dict, model_inputs: ModelInputs, extra_inputs: ExtraInputs = None):
        """Extract draft token predictions from model outputs.

        Called after draft model forward pass.

        Args:
            model_outputs: Output dict from draft model
            model_inputs: Current model inputs
            extra_inputs: Extra inputs from target model (contains aux_hidden_states)

        Returns:
            draft_token_ids: [batch_size, num_speculative_tokens]
            model_metas: Model metadata for next step
            target_hidden_states: Hidden states for next draft step (aux states)
        """
        hidden_states = model_outputs.get('hidden_states', None)
        model_metas = model_outputs.get('model_metas', None)

        if hidden_states is None:
            return None, model_metas, None

        # Get logits from shared lm_head
        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)

        # For DFlash, we use the same aux_hidden_states for all spec steps
        # The next iteration will use the same target_hidden_states
        target_hidden_states = None
        if extra_inputs is not None and hasattr(extra_inputs, 'target_hidden_states'):
            target_hidden_states = extra_inputs.target_hidden_states

        return draft_token_ids, model_metas, target_hidden_states

    def get_target_hidden_size(self, model_config: ModelConfig):
        """Get target hidden size for DFlash.

        DFlash needs 5 concatenated hidden states from target model.
        """
        return model_config.hidden_size * self.num_aux_layers
