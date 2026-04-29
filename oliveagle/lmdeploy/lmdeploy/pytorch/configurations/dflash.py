# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class DFlashModelConfigBuilder(DefaultModelConfigBuilder):
    """DFlash draft model configuration builder."""

    @classmethod
    def condition(cls, hf_config):
        """Check if this builder should be used for DFlash models."""
        # DFlash models have architecture 'DFlashDraftModel'
        return (hasattr(hf_config, 'architectures') and
                hf_config.architectures[0] == 'DFlashDraftModel')

    @classmethod
    def build(cls, hf_config, model_path: str = None, is_draft_model: bool = False, spec_method: str = None, **kwargs):
        """Build DFlash model configuration."""
        # Use default builder to get base config
        cfg = super().build(hf_config, model_path=model_path, **kwargs)

        # DFlash draft model uses ARSpecStrategyFactory to support:
        # 1. target_hidden_states (for intermediate layer hidden states)
        # 2. max_q_seqlen parameter (for parallel multi-token generation)
        # But DFlash has its own attention mechanism, not FlashAttention-3
        cfg.model_paradigm = 'ar_spec'

        return cfg
