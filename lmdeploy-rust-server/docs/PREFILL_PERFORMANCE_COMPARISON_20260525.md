# Prefill 性能对比分析 - 2026-05-25

## 测试环境
- 模型: Qwen3.6-35B-A3B-AWQ (4-bit 量化)
- GPU: Tesla V100 32GB
- CUDA: 12.5

## Python TurboMind 基线 (BENCHMARK_PYTHON_TM_20260518.json)

| Context | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|---------|-----------|-----------------|----------------|
| 1K      | 69.62     | 14,827         | 41.2           |
| 4K      | 122.06    | 33,727         | 41.0           |
| 8K      | 191.37    | 42,875         | 40.6           |

## Rust Server 当前性能 (prefill_benchmark_real.json)

| Context | Avg Time (ms) | Prefill (tok/s) | Max Prefill (tok/s) |
|---------|--------------|-----------------|---------------------|
| 1K      | 281.47       | 3,556          | 5,351               |
| 4K      | 1,173.61     | 3,409          | 4,215               |
| 8K      | 3,008.14     | 2,660          | 2,660               |

## 性能差距分析

| Context | Python TM | Rust Server | 差距 (倍数) | 状态 |
|---------|-----------|-------------|-----------|------|
| 1K      | 14,827 tok/s | 3,556 tok/s | 4.2x 慢 | 需优化 |
| 4K      | 33,727 tok/s | 3,409 tok/s | 9.9x 慢 | 严重差距 |
| 8K      | 42,875 tok/s | 2,660 tok/s | 16.1x 慢 | 严重差距 |

## 根本原因分析

1. **测量方法差异**:
   - Python: TTFT 测量从请求开始到第一个 token
   - Rust: 包含 tokenization + gRPC 序列化 + C++ 引擎

2. **已知瓶颈**:
   - Tokenization: 每次 prompt 重新计算
   - gRPC 序列化: protobuf 开销约 10-15ms
   - KV Cache 传递: 可能存在 CPU→GPU 拷贝
   - 批处理: 未充分利用 GPU batch 并行

3. **已实施的优化**:
   - TASK-1: DLPack zero-copy tensor 传递
   - TASK-2: Pre-tokenization 减少同步竞争
   - TASK-FIX: CUDA 库依赖修复

## 下一步优化方向

1. **立即** (TASK-3, TASK-4):
   - 消除冗余 RMSNorm 调用
   - CUDA Graph 集成减少 kernel 启动开销

2. **中期**:
   - Batch prefill 深度优化
   - Tokenization 缓存
   - gRPC 零拷贝传输

3. **验证**:
   - 重新测量优化后性能
   - 确认达到 50% Python 性能目标
