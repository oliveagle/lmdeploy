# Copyright (c) OpenMMLab. All rights reserved.
"""DFlash proposer for speculative decoding.

DFlash generates all draft tokens in parallel using a diffusion-based approach,
unlike autoregressive methods like EAGLE or DeepseekMTP.
"""

import torch

from lmdeploy.utils import get_logger

from ...config import ModelConfig, SpecDecodeConfig
from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='dflash')
class DFlashProposer(BaseSpecProposer):
    """DFlash proposer - parallel draft token generation.

    Key difference from EAGLE/DeepseekMTP:
    - Generates ALL draft tokens in ONE forward pass (not autoregressively)
    - Uses target model intermediate hidden states as context
    - Uses mask token embeddings as diffusion noise
    """

    def __init__(self, specdecode_config: SpecDecodeConfig, device: torch.device = None):
        super().__init__(specdecode_config, device=device)

        # Read DFlash config
        hf_config = specdecode_config.model_config.hf_config
        dflash_cfg = getattr(hf_config, 'dflash_config', None)

        if dflash_cfg is not None:
            if isinstance(dflash_cfg, dict):
                self.target_layer_ids = dflash_cfg.get('target_layer_ids', [1, 10, 19, 28, 37])
                self.block_size = dflash_cfg.get('block_size', 16)
                self.mask_token_id = dflash_cfg.get('mask_token_id', 248070)
            else:
                self.target_layer_ids = getattr(dflash_cfg, 'target_layer_ids', [1, 10, 19, 28, 37])
                self.block_size = getattr(dflash_cfg, 'block_size', 16)
                self.mask_token_id = getattr(dflash_cfg, 'mask_token_id', 248070)
        else:
            self.target_layer_ids = [1, 10, 19, 28, 37]
            self.block_size = 16
            self.mask_token_id = 248070

        logger.info(f'DFlash config: target_layer_ids={self.target_layer_ids}, '
                     f'block_size={self.block_size}, mask_token_id={self.mask_token_id}')

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build DFlash draft model and share weights with target."""
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)

        # Share embed_tokens and lm_head from target model
        if target_model is not None and self.model is not None:
            # Unwrap PatchedModel to get to DFlashForCausalLM
            draft_model = self.model
            unwrap_steps = 0
            # DFlashForCausalLM is wrapped by PatchedModel (has .model attribute)
            # We need to unwrap until we get DFlashForCausalLM (not its .model which is DFlashModel)
            while hasattr(draft_model, 'model') and unwrap_steps < 5:
                # Check if draft_model itself is DFlashForCausalLM
                if type(draft_model).__name__ == 'DFlashForCausalLM':
                    break
                draft_model = draft_model.model
                unwrap_steps += 1

            logger.info(f'Unwrapped draft model after {unwrap_steps} steps, type: {type(draft_model)}')

            # Share embed_tokens
            if hasattr(draft_model, 'embed_tokens'):
                if hasattr(target_model, 'get_input_embeddings'):
                    logger.info('Sharing embed_tokens from target model.')
                    target_embed = target_model.get_input_embeddings()
                    del draft_model.embed_tokens
                    draft_model.embed_tokens = target_embed
                elif hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
                    logger.info('Sharing embed_tokens from target.model.')
                    target_embed = target_model.model.embed_tokens
                    del draft_model.embed_tokens
                    draft_model.embed_tokens = target_embed

            # Share lm_head from target model
            if hasattr(draft_model, 'lm_head'):
                if hasattr(target_model, 'get_output_embeddings'):
                    logger.info('Sharing lm_head from target model via get_output_embeddings.')
                    draft_model.lm_head = target_model.get_output_embeddings()
                elif hasattr(target_model, 'lm_head'):
                    logger.info('Sharing lm_head directly from target model.')
                    draft_model.lm_head = target_model.lm_head
                elif hasattr(target_model, 'model') and hasattr(target_model.model, 'lm_head'):
                    logger.info('Sharing lm_head from target.model.')
                    draft_model.lm_head = target_model.model.lm_head
                else:
                    logger.warning('Could not find lm_head on target model, using draft model lm_head.')

        # Set requires_grad=False for all parameters to avoid CUDA graph capture issues
        for param in self.model.parameters():
            param.requires_grad = False

    def get_outputs(self,
                    model_outputs: dict,
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None):
        """Get draft token outputs from DFlash forward pass.

        DFlash generates all tokens in one pass, so this is called once
        (unlike EAGLE which calls this in a loop).

        Returns:
            (draft_token_ids, model_metas, hidden_states)
        """
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs.get('model_metas', None)

        # Debug
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        logger.info(f'DFlash get_outputs: hidden_states={hidden_states.shape}, is_decoding={model_inputs.is_decoding}, '
                     f'max_q_seqlen={model_inputs.max_q_seqlen}, num_spec={self.num_speculative_tokens}, '
                     f'has_last_token_idx={extra_inputs is not None and extra_inputs.last_token_indices is not None}')

        # DFlash hidden_states shape: [batch, seq_len, hidden]
        #
        # During decoding (is_decoding=True):
        #   seq_len = 1 + num_spec_tokens (1 real token + 8 draft tokens)
        #   hidden_states = [batch, 9, hidden]
        #   Position 0 is the real token, positions 1-8 are draft tokens
        #
        # During prefill (is_decoding=False):
        #   seq_len = prompt_len or num_spec_tokens
        #   We extract the last token for the next token prediction

        if model_inputs.is_decoding:
            # Decoding phase: extract draft token positions (1:num_spec_tokens+1)
            # These are the LAST num_spec_tokens positions
            if hidden_states.size(1) > 1:
                hidden_states = hidden_states[:, 1:1 + self.num_speculative_tokens]
                logger.info(f'  DFlash decode: extracted positions 1:{1+self.num_speculative_tokens}, '
                           f'new hidden_states shape={hidden_states.shape}, '
                           f'hidden_states_std={hidden_states.std().item():.4f}')
        else:
            # Prefill phase: extract last token
            if extra_inputs is not None and extra_inputs.last_token_indices is not None:
                if model_inputs.seq_length.size(0) == 1:
                    hidden_states = hidden_states[:, -1:]
                else:
                    last_token_loc = extra_inputs.last_token_indices
                    hidden_states = hidden_states[:, last_token_loc]
            else:
                hidden_states = hidden_states[:, -1:]

        # Compute logits and greedy sample
        # hidden_states shape: [batch, num_tokens, hidden]
        logits = self.get_logits(hidden_states)
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        logger.info(f'  DFlash: logits shape={logits.shape}, logits_std={logits.std().item():.4f}, '
                    f'draft_token_ids shape={draft_token_ids.shape}, '
                    f'draft_token_ids={draft_token_ids.flatten().tolist()}')

        return draft_token_ids, model_metas, hidden_states

    def get_target_hidden_size(self, model_config: ModelConfig):
        """Get target hidden size for DFlash.

        DFlash concatenates hidden states from multiple target layers,
        so the total size is num_target_layers * hidden_size.
        """
        return len(self.target_layer_ids) * model_config.hidden_size

    def update_inputs_decoding(self, model_inputs: ModelInputs, extra_inputs, next_input_ids,
                               target_hidden_states, model_metas):
        """Update inputs for decoding step.

        DFlash generates all tokens at once (num_spec_tokens in parallel),
        so we use prefill mode instead of decoding mode to support multi-token attention.
        """
        # Use prefill mode for DFlash's parallel multi-token generation
        model_inputs.is_decoding = False
        batch_size = model_inputs.seq_length.size(0)
        model_inputs.input_ids = next_input_ids

        # Set sequence length to number of draft tokens (e.g., 8)
        num_spec_tokens = next_input_ids.shape[1] if next_input_ids.dim() > 1 else 1
        model_inputs.seq_length = model_inputs.seq_length.new_ones(batch_size) * num_spec_tokens

        # Update KV sequence tracking
        model_inputs.max_kv_seqlen += 1  # Target model advances by 1
        model_inputs.sum_kv_seqlen += model_inputs.seq_length.numel()
        model_inputs.history_lengths += 1  # Each batch adds 1 real token
        if extra_inputs.num_rejected_tokens is not None:
            model_inputs.history_lengths -= extra_inputs.num_rejected_tokens

        # Set max_q_seqlen for prefill attention
        model_inputs.max_q_seqlen = num_spec_tokens
        model_inputs.target_position_ids = model_inputs.history_lengths.unsqueeze(0).clone()
        model_inputs.model_metas = model_metas
        model_inputs.target_hidden_states = target_hidden_states
        return model_inputs
