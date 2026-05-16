# LMDeploy MoonBit API Server PRD

## 项目概述

**目标**: 用 MoonBit 重写 LMDeploy 的 Python OpenAI API Server，保持与 Python 版本相同的 API 接口和行为。

**蓝本**: 当前 Python API Server (`lmdeploy/serve/openai/`) 是完整的 FastAPI 实现，包含：
- `api_server.py` — FastAPI 路由和端点实现
- `protocol.py` — Pydantic 数据模型（请求/响应格式）
- `serving_chat_completion.py` / `serving_completion.py` / `serving_generate.py` — 核心 serving 逻辑
- `api_client.py` — 客户端实现

**当前状态**:
- Rust 版本 (`lmdeploy-rust-server/`) 仍在早期开发中，尚未实现完整功能
- 本项目以 Python 版本为开发蓝本和对比基准，而非 Rust 版本
- Rust 版本完成后将作为性能对比的参考项，但不影响 MoonBit 版本的开发优先级

**预期收益**:
- 消除 Python FastAPI 框架开销
- 原生 gRPC + Protobuf 支持（替代 JSON 序列化）
- Tokenize 缓存机制（消除重复编码开销）
- 更小的内存占用和启动时间
- 整体 API 延迟降低 50%+，吞吐量提升 2-4x

---

## 开发蓝本：Python API Server

MoonBit 版本将以 Python 版 LMDeploy OpenAI API Server 为直接蓝本，保持相同的 API 接口、数据格式和业务流程。

### Python 版核心模块映射

| Python 模块 | 功能 | MoonBit 对应模块 |
|-------------|------|------------------|
| `openai/api_server.py` | FastAPI 路由、端点、中间件 | `server/` (HTTP 路由层) |
| `openai/protocol.py` | Pydantic 请求/响应模型 | `protocol/` (数据结构定义) |
| `serving_chat_completion.py` | Chat 补全逻辑 | `serving/chat.mbt` |
| `serving_completion.py` | Completion 补全逻辑 | `serving/completion.mbt` |
| `serving_generate.py` | 原生 generate 接口 | `serving/generate.mbt` |
| `core/async_engine.py` | 异步推理引擎封装 | `engine/` (TurboMind 桥接) |
| `core/vl_async_engine.py` | 视觉语言模型引擎 | `engine/vl.mbt` (可选) |
| `managers/session_manager.py` | Session 管理 | `session/` |
| `model.py` | Chat 模板 | `template/` |

### API 端点清单（需逐一实现）

以下端点来自 Python `api_server.py`，MoonBit 版本必须全部兼容：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话补全（流式/非流式） |
| `/v1/completions` | POST | 文本补全（流式/非流式） |
| `/v1/models` | GET | 可用模型列表 |
| `/v1/embeddings` | POST | Embedding 请求 |
| `/v1/pooling` | POST | Pooling 请求 |
| `/v1/encode` | POST | 编码请求 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |

### Python protocol.py 数据结构清单

必须兼容的核心数据结构（来自 `protocol.py`）：

- `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionStreamResponse`
- `CompletionRequest` / `CompletionResponse` / `CompletionStreamResponse`
- `ChatMessage` / `DeltaMessage`
- `ModelCard` / `ModelList` / `ModelPermission`
- `UsageInfo` / `ErrorResponse`
- `LogProbs` / `ChoiceLogprobs` / `ChatCompletionTokenLogprob`
- `ToolCall` / `Function` / `ToolChoice`
- `EmbeddingsRequest` / `EmbeddingsResponse`
- `EncodeRequest` / `EncodeResponse`

---

## 对比目标（完成后对比）

**注意**: 以下对比仅在 Rust 版本和 MoonBit 版本均完成后进行，不影响开发优先级。

| 指标 | Python 基准 | MoonBit 目标 | Rust 参考 | 对比方法 |
|------|------------|--------------|-----------|----------|
| **HTTP 吞吐量** | 基准 | 2-4x Python | 参考 | wrk 压测 |
| **gRPC 性能** | N/A | 可用 | 参考 | ghz 压测 |
| **内存占用** | ~500 MB | 预期 <100 MB | 参考 | 监控对比 |
| **启动时间** | ~2-5s | 预期 <500ms | 参考 | 冷启动测试 |
| **首 token 延迟** | ~100-200ms | <50ms (短 context) | 参考 | 微基准测试 |
| **API 兼容性** | 100% | 与 Python 一致 | 与 Python 一致 | 单元测试 |

---

## 用户故事

### US-001: 高性能 gRPC API

**作为** API 用户
**我想要** 使用 gRPC 协议调用 LMDeploy
**以便** 获得比 HTTP/JSON 更高的性能

**验收标准**:
- [ ] 支持 gRPC 流式和非流式推理
- [ ] Protobuf 消息格式定义
- [ ] 同时兼容 OpenAI-compatible HTTP API (向后兼容，与 Python 版本一致)
- [ ] gRPC 性能比 HTTP API 快 2x 以上
- [ ] **基准**: 性能对比以 Python 版本为基准

---

### US-002: Tokenize 缓存系统

**作为** API Server
**我想要** 缓存已 tokenize 的 prompt
**以便** 避免重复编码相同输入

**验收标准**:
- [ ] 实现 LRU 缓存（最大 1000 条，可配置）
- [ ] 缓存 key 为 prompt 的 hash（SHA-256）
- [ ] 缓存命中时直接返回 token_ids
- [ ] 支持 prefix cache（相同前缀复用部分 tokens）
- [ ] 缓存命中率 >80%（典型对话场景）
- [ ] 缓存淘汰策略：LRU + TTL
- [ ] **参考**: Python 版本无此功能，此为 MoonBit 版本新增优化

---

### US-003: 高性能 HTTP/2 服务器

**作为** API Server
**我想要** 使用高性能 MoonBit HTTP 框架
**以便** 充分利用硬件性能

**验收标准**:
- [ ] 使用 MoonBit HTTP 框架（或绑定）
- [ ] 支持 HTTP/2
- [ ] 连接池管理
- [ ] 请求批处理
- [ ] 优雅关闭
- [ ] 健康检查端点 `/health`
- [ ] **参考**: 参考 Rust Axum 框架实现

---

### US-004: 批量推理支持

**作为** API 用户
**我想要** 发送批量请求
**以便** 提高吞吐量

**验收标准**:
- [ ] 支持 `/v1/chat/completions` 批量模式
- [ ] 支持 `/v1/completions` 批量模式
- [ ] 批量大小可配置（默认 8）
- [ ] 批量请求正确路由到 TurboMind
- [ ] **参考**: 参考 Rust 版本批量处理实现

---

### US-005: 流式响应优化

**作为** API 用户
**我想要** 低延迟的流式输出
**以便** 获得更好的用户体验

**验收标准**:
- [ ] SSE (Server-Sent Events) 流式响应
- [ ] 首个 token 延迟 < 50ms (短 context)
- [ ] 支持 gRPC 双向流
- [ ] 连接超时可配置
- [ ] **参考**: 参考 Rust 版本流式响应实现

---

### US-006: OpenAI API 兼容性

**作为** API 用户
**我想要** 使用标准的 OpenAI API 格式
**以便** 无缝迁移现有应用

**验收标准**:
- [ ] 兼容 `/v1/chat/completions` 端点
- [ ] 兼容 `/v1/completions` 端点
- [ ] 兼容 `/v1/models` 端点
- [ ] 兼容 `/v1/embeddings` 端点
- [ ] 支持 streaming 参数
- [ ] 支持 `temperature`, `top_p`, `max_tokens` 等参数

---

### US-007: 配置管理

**作为** 运维人员
**我想要** 通过配置文件管理服务
**以便** 灵活部署

**验收标准**:
- [ ] 支持 TOML/JSON 配置文件
- [ ] 支持环境变量覆盖
- [ ] 配置热重载（SIGHUP）
- [ ] 默认配置文件路径：`/etc/lmdeploy/config.toml`

---

### US-008: 日志和监控

**作为** 运维人员
**我想要** 结构化的日志输出
**以便** 问题排查和性能分析

**验收标准**:
- [ ] 结构化 JSON 日志
- [ ] 日志级别可配置（DEBUG/INFO/WARN/ERROR）
- [ ] Prometheus metrics 端点 `/metrics`
- [ ] 请求延迟直方图
  - `request_duration_seconds`
  - `prefill_duration_seconds`
  - `decode_tokens_per_second`
- [ ] 吞吐量计数器
  - `requests_total`
  - `tokens_generated_total`
- [ ] **参考**: 参考 Rust 版本 metrics 实现（`metrics.rs`）

---

### US-009: 模型加载和管理

**作为** API Server
**我想要** 动态加载模型
**以便** 支持多模型部署

**验收标准**:
- [ ] 启动时加载模型（可配置）
- [ ] 支持热加载模型（API 触发）
- [ ] 支持多模型（模型路由）
- [ ] 模型卸载释放内存
- [ ] 模型加载进度查询

---

### US-010: 错误处理和限流

**作为** API Server
**我想要** 优雅的错误处理和限流
**以便** 保护服务稳定性

**验收标准**:
- [ ] 统一的错误响应格式
- [ ] 请求速率限制（可配置）
- [ ] 请求超时处理
- [ ] 模型 OOM 处理
- [ ] 优雅降级（503 服务不可用）

---

### US-011: Wasm 边缘部署 (MoonBit 独有)

**作为** 边缘计算用户
**我想要** 将 API Server 编译为 Wasm
**以便** 在边缘设备上运行轻量级推理服务

**验收标准**:
- [ ] 编译为 Wasm32-unknown-unknown
- [ ] 在 wasmedge 运行时运行
- [ ] 支持 WASI 接口调用 TurboMind
- [ ] 内存占用 < 10 MB
- [ ] **对比项**: Rust 版本 Wasm 产物大小和性能对比

---

## 技术架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                  MoonBit API Server                         │
│              (moonbit-http + effect handlers)                │
└──────────────────┬────────────────────────────────────────┘
                   │
        ┌───────────┼───────────┬───────────────┐
        │           │           │               │
        ↓           ↓           ↓               ↓
┌──────────┐ ┌──────────┐ ┌──────┐  ┌──────────────┐
│Tokenizer│ │  Router  │ │Queue│  │  gRPC/HTTP  │
│ Cache   │ │(moonbit) │ │     │  │   Handler    │
└──────────┘ └──────────┘ └──────┘  └──────────────┘
        │                       │
        └───────────────────────┘
                    │
                    ↓ C FFI (foreign function)
┌─────────────────────────────────────────────────────────────┐
│              TurboMind C API (libturbomind.so)             │
│                   (extern "C" wrapper)                      │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ↓
                        ┌───────▼────────┐
                        │   TurboMind     │
                        │   (C++ Engine)  │
                        └─────────────────┘
```

### 核心组件

1. **Tokenizer Cache**
   - LRU 缓存实现（MoonBit 纯实现）
   - SHA-256 key 生成
   - Prefix Cache 支持

2. **gRPC Service**
   - Protobuf 消息定义
   - 流式/非流式 RPC
   - 双向流支持

3. **HTTP Handler**
   - OpenAI 兼容 API
   - SSE 流式响应
   - 请求解析和验证

4. **TurboMind Bridge**
   - C FFI 调用 (`foreign` 函数声明)
   - 数据结构转换
   - 错误处理和内存管理

---

## 非功能需求

### 性能指标

| 指标 | 目标 | 测量方法 | 对比基准 |
|------|------|----------|----------|
| P50 延迟 | < 50ms (1K context) | benchmark | Python 版本 |
| P99 延迟 | < 200ms (1K context) | benchmark | Python 版本 |
| Tokenize 缓存命中率 | >80% | metrics | 新增功能，无基准 |
| 吞吐量 | 2x Python server | 压测对比 | Python 版本 |
| 内存占用 | < Python server | 监控 | Python 版本 |
| **二进制大小** | **< 5 MB** | **ls -lh** | N/A |
| **启动时间** | **< 300ms** | **time 命令** | Python (~2-5s) |

### 兼容性

- **向后兼容**: 保持与 Python API 相同的接口
- **多平台**: Linux (优先), macOS (实验性), Wasm (边缘)
- **TurboMind**: 复用现有的 TurboMind 引擎

---

## 开发阶段

### Phase 0: TurboMind C API (Week 1) - 共享
- [ ] 设计稳定的 C API 接口 (`turbomind_c_api.h`)
- [ ] 实现 `extern "C"` 包装层
- [ ] 添加构建脚本 (CMake)
- [ ] 单元测试和集成测试
- [ ] 文档和示例代码

### Phase 1: 基础框架 (Week 2-3)
- [ ] MoonBit FFI 绑定 (`foreign` 声明)
- [ ] 项目初始化（moon.mod.json, 目录结构）
- [ ] 基础 HTTP 服务器（moonbit-http）
- [ ] 基础 `/v1/chat/completions` 端点（对齐 Python API）
- [ ] Tokenize Cache 实现（参考 Rust `tokenizer_cache.rs`）
- [ ] 单元测试

### Phase 2: 核心 API 端点 (Week 4-5)
- [ ] 完整 OpenAI 兼容端点（对标 Python `api_server.py`）
  - `/v1/completions`
  - `/v1/models`
  - `/v1/embeddings`
  - `/v1/pooling`
  - `/v1/encode`
- [ ] 流式响应（SSE）
- [ ] gRPC 服务实现
- [ ] Protobuf 消息定义
- [ ] 请求批处理

### Phase 3: 高级功能 (Week 6-7)
- [ ] 连接池管理
- [ ] 多模型支持
- [ ] 配置管理
- [ ] 日志和监控（对标 Python metrics）
- [ ] 错误处理和限流
- [ ] 性能测试和优化

### Phase 4: 测试和部署 (Week 8-9)
- [ ] 集成测试（与 Python API 对比验证）
- [ ] 性能压测（对比 Python 版本）
- [ ] Wasm 部署测试（MoonBit 独有）
- [ ] 文档编写
- [ ] 部署脚本
- [ ] **与 Rust 版本对比**（可选，如果 Rust 版本已完成）

---

## 风险和依赖

### 风险

| 风险 | 影响 | 缓解措施 |
|------|------|-----------|
| MoonBit 生态不成熟 | 高 | 核心功能自研，必要时使用 C 绑定 |
| FFI 调用性能未知 | 中 | Phase 1 微基准测试验证 |
| HTTP 框架功能有限 | 中 | 必要时使用 C 绑定调用成熟库 |
| Python API 复杂度高 | 中 | 严格对标 Python `protocol.py` 数据结构 |
| 开发周期长 | 中 | 分阶段交付，先核心功能 |

### 技术选型参考

**以下功能参考 Rust 版本实现**:
- **Tokenize Cache**: `lmdeploy-rust-server/src/cache/tokenizer_cache.rs`
  - LRU 缓存 + TTL
  - Prefix cache 支持
  - Cache metrics (hit/miss/eviction)
  
- **配置管理**: `lmdeploy-rust-server/src/config.rs`
  - TOML/JSON 配置文件
  - 环境变量覆盖
  - 默认值机制
  
- **HTTP Handlers**: `lmdeploy-rust-server/src/handlers/http.rs`
  - OpenAI 兼容端点格式
  - SSE 流式响应
  - 错误处理模式

**以下功能参考 Python 版本实现**:
- **核心 API**: `lmdeploy/serve/openai/api_server.py`
- **数据模型**: `lmdeploy/serve/openai/protocol.py` (完整兼容)
- **Serving 逻辑**: `lmdeploy/serve/openai/serving_*.py`

### FFI 路径分析

TurboMind 是纯 C++ 实现，当前**没有对外的 C API**，只有 pybind11 模块 `_turbomind.so`。

| 方案 | 实现方式 | 备注 |
|------|---------|------|
| **自研 C API** (推荐) | Phase 0 共同开发 | MoonBit 和 Rust 共用 |
| **内存管理** | 手动管理 + GC | 需要谨慎设计 |
| **性能开销** | 待验证 | Phase 1 微基准测试 |

**推荐路径**: Phase 0 共同开发 C API，Phase 1+ MoonBit 版本独立开发，参考 Rust 和 Python 版本的最佳实践。

---

## 依赖

### 共享依赖

- **TurboMind C++ 引擎**: LMDeploy 核心
- **TurboMind C API**: Phase 0 开发

### Rust 版本

- **Axum**: HTTP 框架
- **Tonic**: gRPC 框架
- **Tokio**: 异步运行时
- **Protobuf**: 序列化
- **bindgen**: FFI 绑定生成

### MoonBit 版本

- **moonbit-http**: HTTP 框架（官方/社区）
- **moonbit-grpc**: gRPC 框架（待开发或绑定）
- **Effect 系统**: 异步处理
- **Protobuf**: 序列化（绑定或自研）
- **foreign 函数**: FFI 声明
- **参考代码**: `lmdeploy-rust-server/src/` (Tokenizer Cache, HTTP Handlers, Config)

---

## 成功标准

### Phase 0 MVP - C API
- [ ] `turbomind_c_api.h` 头文件发布
- [ ] 单次推理流程跑通 (创建引擎 → Forward → 销毁)
- [ ] 错误处理和内存安全测试通过
- [ ] 性能不低于 pybind11 版本

### Phase 1 MVP - MoonBit Server
- [ ] MoonBit 版本 HTTP API 基本可用
- [ ] 与 Python API 接口一致
- [ ] Tokenize Cache 实现（参考 Rust 版本）
- [ ] 通过基本功能测试

### Phase 2 优化版
- [ ] MoonBit 版本性能比 Python server 快 2x
- [ ] Tokenize Cache 实现并生效（命中率 >80%）
- [ ] gRPC API 可用
- [ ] 所有核心 OpenAI 端点实现

### Phase 3 完整版
- [ ] 所有核心功能实现
- [ ] 性能达标（见性能指标表）
- [ ] 生产可用

### Phase 4 对比报告
- [ ] MoonBit vs Python 性能对比报告
- [ ] MoonBit vs Rust 性能对比报告（如果 Rust 版本已完成）
- [ ] Wasm 部署验证（MoonBit 独有）
- [ ] 技术选型建议

---

## 附录

### A. 技术选型对比

| 组件 | Rust 版本 | MoonBit 版本 | 备注 |
|------|-----------|--------------|------|
| HTTP 框架 | Axum | moonbit-http | Rust 更成熟 |
| gRPC 框架 | Tonic | moonbit-grpc | Rust 更成熟 |
| 运行时 | Tokio | Effect 系统 | 设计理念不同 |
| 序列化 | Protobuf | Protobuf | 共享 |
| 缓存 | lru-rs | 自研 LRU | 相当 |
| FFI | bindgen | foreign | Rust 更自动化 |
| C API | 自研 | 自研 (共享) | 共享 |

### B. 目录结构对比

#### Rust 版本

```
lmdeploy-rust-server/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── server.rs
│   ├── router.rs
│   ├── handlers/
│   ├── cache/
│   ├── turbomind/
│   ├── proto/
│   ├── config.rs
│   └── lib.rs
├── tests/
└── config/
```

#### MoonBit 版本

```
lmdeploy-moonbit-server/
├── moon.mod.json
├── src/
│   ├── main/
│   │   └── main.mbt
│   ├── server/
│   │   └── server.mbt
│   ├── router/
│   │   └── router.mbt
│   ├── handlers/
│   │   ├── http.mbt
│   │   └── grpc.mbt
│   ├── cache/
│   │   └── tokenizer.mbt
│   ├── turbomind/
│   │   ├── bridge.mbt
│   │   └── wrapper.mbt
│   ├── config/
│   │   └── config.mbt
│   └── metrics/
│       └── metrics.mbt
├── tests/
│   ├── integration/
│   └── unit/
├── target/
│   ├── wasm32-wasi/
│   └── x86_64-unknown-linux-gnu/
└── config/
    └── default.jsonc
```

### C. 性能对比测试计划

#### 测试环境

- **硬件**: Tesla V100 32GB
- **操作系统**: Ubuntu 22.04 LTS
- **模型**: Qwen3.5-35B-A3B-AWQ

#### 测试工具

- **HTTP 压测**: wrk, hey
- **gRPC 压测**: ghz
- **延迟测试**: hyperfine
- **内存分析**: /usr/bin/time, valgrind
- **性能分析**: perf, flamegraph

#### 测试场景

| 场景 | 并发 | 请求 | 输出长度 |
|------|------|------|----------|
| 短请求 | 1, 10, 100 | 10000 | 128 |
| 中请求 | 1, 10, 100 | 10000 | 512 |
| 长请求 | 1, 10, 100 | 5000 | 2048 |
| 流式 | 1, 10, 50 | 5000 | 512 |

#### 报告格式

```json
{
  "date": "2026-05-16",
  "environment": {
    "gpu": "Tesla V100 32GB",
    "model": "Qwen3.5-35B-A3B-AWQ"
  },
  "results": {
    "rust": {
      "binary_size": "8.5 MB",
      "startup_time": "52ms",
      "memory_rss": "32 MB",
      "http_qps": 4500,
      "grpc_qps": 5200,
      "p50_latency": "45ms",
      "p99_latency": "180ms"
    },
    "moonbit": {
      "binary_size": "3.2 MB",
      "startup_time": "25ms",
      "memory_rss": "18 MB",
      "http_qps": 4200,
      "grpc_qps": 4800,
      "p50_latency": "48ms",
      "p99_latency": "195ms"
    },
    "conclusion": "MoonBit 在二进制大小和启动时间上有显著优势，HTTP/gRPC 吞吐量接近 Rust 版本的 90-95%，适合边缘部署场景"
  }
}
```

### D. 参考资料

#### Python API Server（开发蓝本）

**路径**: `/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/serve/openai/`

**核心文件**:
- `api_server.py` — FastAPI 服务器主文件（路由、端点、中间件）
- `protocol.py` — OpenAI 兼容的请求/响应数据模型（Pydantic）
- `serving_chat_completion.py` — Chat 补全处理逻辑
- `serving_completion.py` — Completion 补全处理逻辑
- `serving_generate.py` — 原生 generate 接口
- `api_client.py` — 客户端实现
- `launch_server.py` — 服务器启动脚本

**核心模块**:
- `core/async_engine.py` — 异步推理引擎封装
- `core/vl_async_engine.py` — 视觉语言模型引擎
- `managers/session_manager.py` — Session 管理

#### Rust 版本（功能参考）

**路径**: `/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy-rust-server/src/`

**核心文件**:
- `server.rs` — Axum 服务器主文件
- `handlers/http.rs` — HTTP 端点实现（OpenAI 兼容）
- `cache/tokenizer_cache.rs` — Tokenize 缓存实现（LRU + TTL + Prefix）
- `config.rs` — 配置管理
- `grpc/` — gRPC 服务实现

#### 官方资源

- MoonBit 官网: https://moonbitlang.com/
- MoonBit GitHub: https://github.com/moonbitlang/moonbit
- Axum 文档: https://axum.rs/
- Tonic 文档: https://github.com/hyperium/tonic

---

**文档版本**: 1.1
**创建日期**: 2026-05-16
**最后更新**: 2026-05-16
**状态**: 已更正（开发蓝本改为 Python 版本，Rust 版本仅作参考）
