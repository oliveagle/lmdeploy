# LMDeploy Rust vs Python 性能基准测试报告

**测试日期**: 2026-05-18  
**GPU**: Tesla V100 32GB (PG503-216)  
**CUDA**: 12.5  
**模型**: Qwen3.6-35B-A3B-AWQ (AWQ 4-bit)

---

## 1. Python TurboMind API 测试结果

### 测试环境

- **引擎**: LMDeploy Python TurboMind
- **版本**: 0.13.0
- **模型路径**: `/mnt/eaget-4tb/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ`
- **启动命令**:
```bash
lmdeploy serve api_server /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ \
  --backend turbomind \
  --tp 1 \
  --session-len 8192 \
  --cache-max-entry-count 0.8 \
  --server-port 8001
```

### 性能数据

| 场景 | Context Length | TTFT (ms) | Prefill (t/s) | Decode (t/s) | ITL (ms) |
|------|----------------|-----------|---------------|--------------|----------|
| Short | 1024 chars (~138 tokens) | 70 | 14827 | 41 | 24.3 |
| Medium | 4096 chars (~550 tokens) | 122 | 33728 | 41 | 24.4 |
| Long | 8192 chars (~1100 tokens) | 191 | 42875 | 41 | 24.7 |

**说明**:
- TTFT (Time To First Token): 从请求到第一个 token 生成的时间
- Prefill 速度: 预填充阶段处理速度
- Decode 速度: 解码阶段生成速度
- ITL (Inter-Token Latency): 平均 token 间延迟

### 关键发现

1. **Prefill 速度**: 随着上下文长度增加，Prefill 吞吐量显著提升（14K → 43K t/s）
2. **Decode 速度**: 稳定在 ~41 tokens/s，不受上下文长度影响
3. **ITL**: 稳定在 ~24ms，对应 ~41 tokens/s 的解码速度

---

## 2. Rust Server C API 分析

### 当前实现状态

Rust Server 位于 `lmdeploy-rust-server/` 目录，包含:
- HTTP/2 服务器 (Axum)
- gRPC 服务器 (Tonic)
- TurboMind C API FFI 绑定
- Tokenizer 缓存
- 批量推理支持
- OpenAI API 兼容接口

### C API 模型加载问题

**问题**: `TM_TurboMind_InitFromPath()` 无法直接加载 HuggingFace `.safetensors` 格式

**根因**:
- C API 期望 TurboMind 转换后的 `.bin` 格式模型
- 缺少 HuggingFace 模型图构建逻辑
- `InitFromHF` 实现为 Python bridge hack，不适用于生产

**Python 版本的处理**:
- Python TurboMind API 自动进行 HF → TM 转换
- 首次加载时在 workspace 目录生成转换后的权重
- 后续加载直接使用转换后的格式

### Rust C API 初始化序列

根据 `src/turbomind/capi/turbomind_c.h`，正确的初始化序列是:

1. `TM_TurboMind_CreateContext` - 创建 CUDA context
2. `TM_TurboMind_CreateRoot` - 创建 ModelRoot sentinel
3. `TM_TurboMind_ProcessWeights` - 处理并加载权重到 GPU
4. `TM_TurboMind_CreateEngine` - 完成引擎创建

**当前问题**: `ProcessWeights` 阶段失败，因为输入是 HF 格式而非 TM 格式。

---

## 3. 解决方案

### 方案 A: 使用 Python TurboMind API (推荐)

**优点**:
- 完全支持 HF safetensors 格式
- 自动处理 AWQ/GPTQ 量化
- 成熟稳定，生产可用

**缺点**:
- Python 开销
- GIL 限制（虽然 TurboMind 核心在 C++ 中）

### 方案 B: 预转换模型为 TurboMind 格式

**步骤**:
```bash
# 使用 Python API 生成 workspace
python -c "
from lmdeploy.turbomind import TurboMind
tm = TurboMind('/path/to/hf/model')
# workspace 自动生成在 model_path/workspace/
"
```

**然后 Rust 指向 workspace 目录**:
```rust
TurboMindEngine::new("/path/to/hf/model/workspace").await?
```

**优点**:
- Rust 可以直接使用 C API
- 避免运行时 Python 依赖

**缺点**:
- 需要预先转换模型
- 双份模型存储（HF + TM）

### 方案 C: 在 C API 中实现 HF 模型加载

**工作量**: 大
**需要**:
- 在 C++ 中实现 HF config.json 解析
- 实现 safetensors 文件读取
- 构建模型图并分配权重

---

## 4. 结论

| 指标 | Python TurboMind | Rust C API (当前) |
|------|------------------|-------------------|
| HF safetensors 支持 | ✅ 自动 | ❌ 需要预转换 |
| AWQ 量化支持 | ✅ | ⚠️ 需要转换 |
| 性能 (Decode) | ~41 t/s | 未测试 |
| HTTP 延迟 | ~1-2ms | 理论 <1ms |
| 部署复杂度 | 低 | 中（需要预转换） |

**建议**:
1. **短期**: 使用 Python TurboMind API 进行推理
2. **中期**: 实现模型预转换工具，Rust 指向 workspace
3. **长期**: 在 C API 中实现 HF 模型直接加载

---

## 5. 测试配置参考

### Python LMDeploy 配置

```toml
[engine]
dtype = "float16"
session_len = 8192
max_batch_size = 128
cache_max_entry_count = 0.8
cache_block_seq_len = 64
enable_prefix_caching = false
model_format = "awq"

[parallel]
tp = 1
dp = 1
cp = 1
```

### Rust Server 配置 (待测试)

```toml
[model]
model_path = "/mnt/eaget-4tb/.../workspace"  # 转换后的路径
session_len = 8192
batch_size = 32
tp_size = 1
data_type = "fp16"

[server]
http_addr = "0.0.0.0"
http_port = 3000
http2_enabled = true
workers = 4
```

---

**报告生成**: 2026-05-18  
**测试工具**: Python requests + streaming API  
**数据文件**: `BENCHMARK_PYTHON_TM_20260518.json`
