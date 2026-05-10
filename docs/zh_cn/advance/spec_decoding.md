# Speculative Decoding

投机解码是一种优化技术，它通过引入轻量级草稿模型来预测多个后续token，再由主模型在前向推理过程中验证并选择匹配度最高的长token序列。与标准的自回归解码相比，这种方法可使系统一次性生成多个token。

> \[!NOTE\]
> 请注意，这是lmdeploy中的实验性功能。

## 示例

请参考如下使用示例。

### Eagle 3

#### 安装依赖

安装 [flash-atten3 ](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release)

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

### DFlash (Turbomind 后端)

DFlash 是一种专为 Turbomind 后端优化的投机解码方法。它使用轻量级草稿模型和非因果注意力来实现高效的 token 预测。

> [!NOTE]
> DFlash 仅在 `turbomind` 后端可用。

#### 功能特性

- **非因果注意力**: 草稿模型可以同时看到所有草稿 token，从而提高预测质量
- **可配置的草稿 token 数量**: 调整投机 token 数量（推荐 8-16）
- **量化支持**: 草稿模型权重可以量化为 INT8/INT4 以减少内存占用
- **EP+TP 支持**: 支持专家并行和张量并行配置

#### 草稿模型量化

草稿模型权重支持量化以减少 GPU 内存使用。使用 `DraftQuantPolicy` 枚举指定量化方法：

```python
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import DraftQuantPolicy, SpeculativeConfig

# FP16 草稿模型（无量化，约 2GB 内存）
spec_cfg = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.FP16,  # 无量化（默认）
)

# INT4 量化草稿模型（约 1GB 内存）
spec_cfg_int4 = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.INT4,
    group_size=128,
)

# AWQ 量化草稿模型
spec_cfg_awq = SpeculativeConfig(
    method='dflash',
    model='/path/to/draft/model',
    num_speculative_tokens=16,
    quant_policy=DraftQuantPolicy.AWQ,
    group_size=128,
    num_groups_per_channel=1,
)

# 与 Turbomind 引擎一起使用
pipe = pipeline(
    '/path/to/target/model',
    backend_config=TurbomindEngineConfig(tp=4, ep=4),
    speculative_config=spec_cfg,
)
```

| 量化策略 | 内存占用 | 适用场景 |
|---------|---------|---------|
| `FP16` | ~2GB | 默认，最高精度 |
| `INT8` | ~1GB | 精度/内存平衡 |
| `INT4` | ~0.5GB | 最小内存 |
| `AWQ` | ~1GB | 激活感知量化 |

#### Pipeline 示例

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

#### Serving 示例

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
