# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for DFlash speculative decoding configuration.

Tests cover:
- SpeculativeConfig dataclass initialization and defaults
- TurbomindEngineConfig with speculative_config integration
- TurbomindModelConfig speculative_config serialization
- DFlash draft model quantization support (STORY-008)
- Edge cases and invalid configurations
"""

import pytest

from lmdeploy.messages import (DraftQuantPolicy, SpeculativeConfig, TurbomindEngineConfig)


class TestSpeculativeConfig:
    """Test SpeculativeConfig dataclass."""

    def test_default_values(self):
        """Test default field values."""
        config = SpeculativeConfig()
        assert config.method == 'dflash'
        assert config.model == ''
        assert config.num_speculative_tokens == 8
        assert config.quant_policy == DraftQuantPolicy.FP16
        assert config.group_size == 128

    def test_custom_values(self):
        """Test custom field values."""
        config = SpeculativeConfig(
            method='dflash',
            model='/path/to/draft/model',
            num_speculative_tokens=4,
            quant_policy=DraftQuantPolicy.INT4,
            group_size=64,
        )
        assert config.method == 'dflash'
        assert config.model == '/path/to/draft/model'
        assert config.num_speculative_tokens == 4
        assert config.quant_policy == DraftQuantPolicy.INT4
        assert config.group_size == 64

    def test_num_speculative_tokens_boundary(self):
        """Test edge values for num_speculative_tokens."""
        # Minimum valid value
        config_min = SpeculativeConfig(num_speculative_tokens=1)
        assert config_min.num_speculative_tokens == 1

        # Large value
        config_large = SpeculativeConfig(num_speculative_tokens=64)
        assert config_large.num_speculative_tokens == 64


class TestDraftQuantPolicy:
    """Test DraftQuantPolicy enum (STORY-008)."""

    def test_quant_policy_values(self):
        """Test DraftQuantPolicy enum values."""
        assert DraftQuantPolicy.FP16 == 0
        assert DraftQuantPolicy.INT8 == 1
        assert DraftQuantPolicy.INT4 == 2
        assert DraftQuantPolicy.AWQ == 3
        assert DraftQuantPolicy.GPTQ == 4

    def test_quant_policy_names(self):
        """Test DraftQuantPolicy enum names."""
        assert DraftQuantPolicy.FP16.name == 'FP16'
        assert DraftQuantPolicy.INT8.name == 'INT8'
        assert DraftQuantPolicy.INT4.name == 'INT4'
        assert DraftQuantPolicy.AWQ.name == 'AWQ'
        assert DraftQuantPolicy.GPTQ.name == 'GPTQ'

    def test_speculative_config_with_quantization(self):
        """Test SpeculativeConfig with quantization parameters."""
        config = SpeculativeConfig(
            quant_policy=DraftQuantPolicy.INT4,
            group_size=128,
        )
        assert config.quant_policy == DraftQuantPolicy.INT4
        assert config.group_size == 128

    def test_speculative_config_awq_quantization(self):
        """Test SpeculativeConfig with AWQ quantization."""
        config = SpeculativeConfig(
            quant_policy=DraftQuantPolicy.AWQ,
            group_size=128,
            num_groups_per_channel=1,
        )
        assert config.quant_policy == DraftQuantPolicy.AWQ
        assert config.group_size == 128
        assert config.num_groups_per_channel == 1


class TestTurbomindEngineConfigWithSpeculative:
    """Test TurbomindEngineConfig with speculative_config."""

    def test_speculative_config_attribute(self):
        """Test that TurbomindEngineConfig accepts speculative_config."""
        spec_config = SpeculativeConfig(
            method='dflash',
            model='/path/to/draft',
            num_speculative_tokens=8,
        )
        engine_config = TurbomindEngineConfig(
            session_len=4096,
            speculative_config=spec_config,
        )
        assert engine_config.speculative_config is not None
        assert engine_config.speculative_config.method == 'dflash'
        assert engine_config.speculative_config.model == '/path/to/draft'
        assert engine_config.speculative_config.num_speculative_tokens == 8

    def test_no_speculative_config(self):
        """Test default None for speculative_config."""
        engine_config = TurbomindEngineConfig(session_len=4096)
        assert engine_config.speculative_config is None

    def test_speculative_config_with_other_options(self):
        """Test speculative_config combined with other engine options."""
        spec_config = SpeculativeConfig(num_speculative_tokens=16)
        engine_config = TurbomindEngineConfig(
            dtype='bfloat16',
            tp=2,
            session_len=8192,
            cache_max_entry_count=0.8,
            speculative_config=spec_config,
        )
        assert engine_config.dtype == 'bfloat16'
        assert engine_config.tp == 2
        assert engine_config.session_len == 8192
        assert engine_config.speculative_config.num_speculative_tokens == 16


class TestTurbomindModelConfigSpeculative:
    """Test TurbomindModelConfig speculative_config serialization."""

    def test_update_from_engine_config(self):
        """Test speculative_config propagation from engine config."""
        from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

        spec_config = SpeculativeConfig(
            method='dflash',
            model='/path/to/draft',
            num_speculative_tokens=8,
        )
        engine_config = TurbomindEngineConfig(
            session_len=4096,
            speculative_config=spec_config,
        )

        model_config = TurbomindModelConfig()
        model_config.update_from_engine_config(engine_config)

        assert model_config.speculative_config is not None
        assert model_config.speculative_config['method'] == 'dflash'
        assert model_config.speculative_config['model'] == '/path/to/draft'
        assert model_config.speculative_config['num_speculative_tokens'] == 8

    def test_to_dict_with_speculative(self):
        """Test to_dict includes speculative_config."""
        from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

        model_config = TurbomindModelConfig(
            speculative_config={
                'method': 'dflash',
                'model': '/path/to/draft',
                'num_speculative_tokens': 8,
            }
        )
        result = model_config.to_dict()
        assert 'speculative_config' in result
        assert result['speculative_config']['method'] == 'dflash'

    def test_from_dict_with_speculative(self):
        """Test from_dict parses speculative_config."""
        from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

        config_dict = {
            'model_config': {},
            'attention_config': {},
            'lora_config': {},
            'speculative_config': {
                'method': 'dflash',
                'model': '/path/to/draft',
                'num_speculative_tokens': 16,
            },
        }
        model_config = TurbomindModelConfig.from_dict(config_dict)
        assert model_config.speculative_config is not None
        assert model_config.speculative_config['method'] == 'dflash'
        assert model_config.speculative_config['num_speculative_tokens'] == 16

    def test_from_dict_without_speculative(self):
        """Test from_dict handles missing speculative_config."""
        from lmdeploy.turbomind.deploy.config import TurbomindModelConfig

        config_dict = {
            'model_config': {},
            'attention_config': {},
            'lora_config': {},
        }
        model_config = TurbomindModelConfig.from_dict(config_dict)
        assert model_config.speculative_config is None
