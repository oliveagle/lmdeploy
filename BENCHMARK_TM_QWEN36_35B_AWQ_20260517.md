# LMDeploy TurboMind 性能测试记录

## 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3.6-35B-A3B-AWQ |
| 量化 | AWQ 4-bit |
| GPU | Tesla V100 32GB |
| 后端 | TurboMind (C++) |
| TP | 1 |

## 测试结果 (2026-05-17)

### TurboMind Python API 测试

| Context Length | Prompt Tokens | Decode Tokens | Total Time | Decode Speed |
|----------------|---------------|---------------|------------|---------------|
| 128 | 35 | 128 | 3.12s | 41.1 tokens/s |
| 512 | 83 | 128 | 3.12s | 41.1 tokens/s |
| 1024 | 147 | 128 | 3.13s | 40.8 tokens/s |
| 2048 | 275 | 128 | 3.15s | 40.6 tokens/s |

### 配置参数

```bash
lmdeploy serve api_server /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ \
  --backend turbomind \
  --tp 1 \
  --session-len 8192 \
  --cache-max-entry-count 0.8 \
  --server-port 8001 \
  --log-level INFO \
  --trust-remote-code
```

### 引擎配置

- dtype: float16
- session_len: 8192
- max_batch_size: 128
- cache_max_entry_count: 0.8
- cache_block_seq_len: 64
- enable_prefix_caching: False
- model_format: awq

### 注意事项

1. **PyTorch 后端不支持 AWQ 量化** - 会报错 `RuntimeError: Unsupported quant method: awq`
2. **TurboMind 后端支持 HuggingFace safetensors 格式** - 自动加载和转换
3. **模型包含 thinking process** - Qwen3.6 会输出思考过程，影响实际输出速度

### 下一步测试

- 对比 Rust C API 版本性能
- 测试并发请求性能
- 测试不同 max_tokens 配置的影响

## Python API Server vs Rust C API 对比

### Python TurboMind API (via lmdeploy serve)

- **启动方式**: `lmdeploy serve api_server --backend turbomind`
- **模型加载**: Python 桥接自动处理 HF→TurboMind 权重转换
- **性能**: ~40 tokens/s (V100 32GB, TP=1)

### Rust C API (libturbomind_c.so)

- **问题**: `InitFromPath` 无法直接加载 HF safetensors 格式
- **错误**: `Check failed: l0` (ModelWeight::prepare 期望 layer 结构已填充)
- **原因**: C API 缺少 HF 模型图构建逻辑

### 解决方案

**推荐**: 使用 Python TurboMind API
```bash
lmdeploy serve api_server /path/to/hf/model \
  --backend turbomind \
  --tp 1 \
  --session-len 8192
```

**原因**: Python API 自动处理：
- HuggingFace safetensors 加载
- 权重名称映射 (HF → TurboMind)
- 头重排序 (RoPE)
- TP 分片
- AWQ 反量化
