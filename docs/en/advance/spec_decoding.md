# Speculative Decoding

Speculative decoding is an optimization technique that introcude a lightweight draft model to propose multiple next tokens and then, the main model verify and choose the longest matched tokens in a forward pass. Compared with standard auto-regressive decoding, this methold lets the system generate multiple tokens at once.

> \[!NOTE\]
> This is an experimental feature in lmdeploy.

## Examples

Here are some examples.

### Eagle 3

#### Prepare

Install [flash-atten3 ](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release)

```shell
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

#### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    model_path = 'meta-llama/Llama-3.1-8B-Instruct'
    spec_cfg = SpeculativeConfig(
        method='eagle3',
        num_speculative_tokens=3,
        model='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    )
    pipe = pipeline(model_path, backend_config=PytorchEngineConfig(max_batch_size=128), speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)

```

#### serving

```shell
lmdeploy serve api_server \
meta-llama/Llama-3.1-8B-Instruct \
--backend pytorch \
--server-port 24545 \
--speculative-draft-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
--speculative-algorithm eagle3 \
--speculative-num-draft-tokens 3 \
--max-batch-size 128 \
--enable-metrics
```

### Deepseek MTP

#### Prepare

Install [FlashMLA](https://github.com/deepseek-ai/FlashMLA?tab=readme-ov-file#installation)

```shell
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```

#### pipeline

```python
from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    model_path = 'deepseek-ai/DeepSeek-V3'
    spec_cfg = SpeculativeConfig(
        method='deepseek_mtp',
        num_speculative_tokens=3,
    )
    pipe = pipeline(model_path,
                    backend_config=PytorchEngineConfig(tp=16, max_batch_size=128),
                    speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)

```

#### serving

```shell
lmdeploy serve api_server \
deepseek-ai/DeepSeek-V3 \
--backend pytorch \
--server-port 24545 \
--tp 16 \
--speculative-algorithm deepseek_mtp \
--speculative-num-draft-tokens 3 \
--max-batch-size 128 \
--enable-metrics
```

### DFlash (Turbomind Backend)

DFlash is a speculative decoding method optimized for the Turbomind backend. It uses a lightweight draft model with non-causal attention for efficient token prediction.

> [!NOTE]
> DFlash is only available with the `turbomind` backend.

#### Features

- **Non-causal attention**: Draft model sees all draft tokens simultaneously for better prediction quality
- **Configurable draft tokens**: Adjust number of speculative tokens (8-16 recommended)
- **Quantization support**: Draft model weights can be quantized to INT8/INT4 to reduce memory footprint
- **EP+TP support**: Works with Expert Parallelism and Tensor Parallelism configurations

#### Quantization for Draft Model

Draft model weights support quantization to reduce GPU memory usage. Use `DraftQuantPolicy` enum to specify the quantization method:

```python
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import DraftQuantPolicy, SpeculativeConfig

# FP16 draft model (no quantization, ~2GB memory)
spec_cfg = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.FP16,  # No quantization (default)
)

# INT4 quantized draft model (~1GB memory)
spec_cfg_int4 = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.INT4,
    group_size=128,
)

# AWQ quantized draft model
spec_cfg_awq = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.AWQ,
    group_size=128,
    num_groups_per_channel=1,
)

# Use with Turbomind engine
pipe = pipeline(
    '/path/to/target/model',
    backend_config=TurbomindEngineConfig(tp=4, ep=4),
    speculative_config=spec_cfg,
)
```

| Policy | Memory | Use Case |
|--------|--------|----------|
| `FP16` | ~2GB | Default, best quality |
| `INT8` | ~1GB | Balanced quality/memory |
| `INT4` | ~0.5GB | Minimum memory |
| `AWQ` | ~1GB | Activation-aware quantization |

#### Pipeline Example

```python
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':
    target_model = 'Qwen/Qwen3-0.6B'
    draft_model = '/path/to/draft/model'

    spec_cfg = SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=16,
    )

    pipe = pipeline(
        target_model,
        backend_config=TurbomindEngineConfig(
            tp=4,
            ep=4,
            quant_policy=42,  # TurboQuant KV cache
            session_len=2048,
        ),
        speculative_config=spec_cfg,
    )

    response = pipe(['Hello, how are you?'])
    print(response)
```

#### Serving Example

```shell
lmdeploy serve api_server \
Qwen/Qwen3-0.6B \
--backend turbomind \
--tp 4 \
--ep 4 \
--speculative-draft-model /path/to/draft/model \
--speculative-algorithm dflash \
--speculative-num-draft-tokens 16 \
--speculative-quant-policy 2 \
--speculative-group-size 128 \
--server-port 24545
```
