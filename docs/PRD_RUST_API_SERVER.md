# LMDeploy Rust API Server PRD

## 项目概述

**目标**: 用 Rust 重写 LMDeploy 的 API Server，替换当前的 Python FastAPI 实现，显著提升 HTTP API 性能。

**背景**: 当前 Python API Server 存在严重的性能瓶颈：
- JSON 序列化/反序列化开销大
- Tokenize 每次请求重复执行，无缓存
- Python FastAPI 框架开销
- HTTP/1.1 连接管理效率低

**预期收益**:
- gRPC + Protobuf 替代 JSON (2-3x 序列化性能提升)
- Tokenize 缓存机制 (消除重复编码开销)
- HTTP/2 + 连接池 (网络层优化)
- 整体 API 延迟降低 50%+，吞吐量提升 2-4x

---

## 用户故事

### US-001: 高性能 gRPC API

**作为** API 用户
**我想要** 使用 gRPC 协议调用 LMDeploy
**以便** 获得比 HTTP/JSON 更高的性能

**验收标准**:
- [ ] 支持 gRPC 流式和非流式推理
- [ ] Protobuf 消息格式定义
- [ ] 同时兼容 OpenAI-compatible HTTP API (向后兼容)
- [ ] gRPC 性能比 HTTP API 快 2x 以上

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

---

### US-003: 高性能 HTTP/2 服务器

**作为** API Server
**我想要** 使用高性能 Rust HTTP 框架
**以便** 充分利用硬件性能

**验收标准**:
- [ ] 使用 Axum 框架（或 Tonic 用于 gRPC）
- [ ] 支持 HTTP/2
- [ ] 连接池管理
- [ ] 请求批处理
- [ ] 优雅关闭
- [ ] 健康检查端点 `/health`

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
我想要** 结构化的日志输出
**以便** 问题排查和性能分析

**验收标准**:
- [ ] 结构化 JSON 日志
- [ ] 日志级别可配置（DEBUG/INFO/WARN/ERROR）
- [ ] Prometheus metrics 端点 `/metrics`
- [ ] 请求延迟直方图
- - `request_duration_seconds`
- - `prefill_duration_seconds`
- - `decode_tokens_per_second`
- [ ] 吞吐量计数器
- - `requests_total`
- - `tokens_generated_total`

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
我想要** 优雅的错误处理和限流
**以便** 保护服务稳定性

**验收标准**:
- [ ] 统一的错误响应格式
- [ ] 请求速率限制（可配置）
- [ ] 请求超时处理
- [ ] 模型 OOM 处理
- [ ] 优雅降级（503 服务不可用）

---

## 技术架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust API Server                         │
│                   (axum + tokio)                         │
└──────────────────┬────────────────────────────────────────┘
                   │
        ┌───────────┼───────────┬───────────────┐
        │           │           │               │
        ↓           ↓           ↓               ↓
┌──────────┐ ┌──────────┐ ┌──────┐  ┌──────────────┐
│Tokenizer│ │  Router  │ │Queue│  │  gRPC/HTTP  │
│ Cache   │ │  (axum)  │ │     │  │   Handler    │
└──────────┘ └──────────┘ └──────┘  └──────────────┘
        │                       │
        └───────────────────────┘
                    │
                    ↓ bindgen FFI
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
   - LRU 缓存实现
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
   - 方案 A (推荐): PyO3 FFI — 嵌入 Python 解释器，通过 `_turbomind.so` 调用
   - 方案 B (备选): C++ Shim — 写 `extern "C"` 包装层（风险高，依赖内部头文件）
   - 数据结构转换
   - 错误处理

---

## 非功能需求

### 性能指标

| 指标 | 目标 | 测量方法 |
|------|------|----------|
| P50 延迟 | < 50ms (1K context) | benchmark |
| P99 延迟 | < 200ms (1K context) | benchmark |
| Tokenize 缓存命中率 | >80% | metrics |
| 吞吐量 | 2x Python server | 压测对比 |
| 内存占用 | < Python server | 监控 |

### 兼容性

- **向后兼容**: 保持与 Python API 相同的接口
- **多平台**: Linux (优先), macOS (实验性)
- **TurboMind**: 复用现有的 TurboMind 引擎

---

## 开发阶段

### Phase 0: TurboMind C API (Week 1)
- [ ] 设计稳定的 C API 接口 (`turbomind_c_api.h`)
- [ ] 实现 `extern "C"` 包装层
- [ ] 添加构建脚本 (CMake)
- [ ] 单元测试和集成测试
- [ ] 文档和示例代码

### Phase 1: 基础框架 (Week 2-3)
- [ ] Rust FFI 绑定 (`turbomind-sys` crate)
- [ ] 项目初始化（Cargo.toml, 目录结构）
- [ ] 基础 HTTP 服务器（axum）
- [ ] 基础 `/v1/chat/completions` 端点
- [ ] 单元测试

### Phase 2: 性能优化 (Week 4-5)
- [ ] Tokenize Cache 实现
- [ ] gRPC 服务实现
- [ ] Protobuf 消息定义
- [ ] 连接池管理
- [ ] 性能测试和优化

### Phase 3: 高级功能 (Week 6-7)
- [ ] 流式响应优化
- [ ] 批量推理支持
- [ ] 多模型支持
- [ ] 配置管理
- [ ] 日志和监控

### Phase 4: 测试和部署 (Week 8-9)
- [ ] 集成测试
- [ ] 性能基准测试
- [ ] 文档编写
- [ ] 部署脚本

---

## 风险和依赖

### 风险

| 风险 | 影响 | 缓解措施 |
|------|------|-----------|
| C API 实现复杂度 | 高 | 充分测试，封装良好的 FFI 层 |
| C API 与 TurboMind 耦合 | 低 | 维护稳定的接口版本，内部实现可随时重构 |
| 开发周期长 | 中 | 分阶段交付，先核心功能 |

### FFI 路径分析

TurboMind 是纯 C++ 实现，当前**没有对外的 C API**，只有 pybind11 模块 `_turbomind.so`。

| 方案 | 描述 | 风险 |
|------|------|------|
| **自研 C API (Phase 0)** (推荐) | 为 TurboMind 添加 `extern "C"` 接口，Rust 通过 bindgen 调用 | 风险可控，一次性完成，长期收益 |
| **PyO3 嵌入 Python** | Rust 进程内嵌入 Python 解释器，通过 sys.path 加载 `_turbomind`，调用 `TurboMind`/`TurboMindInstance` Python API | 风险低但依赖 Python 环境，额外进程开销 |
| **纯 C++ 重写 TurboMind API** | 完全绕过 TurboMind，重写推理逻辑 | 极高风险，重写量等同于全新项目 |

**推荐路径**: 先为 TurboMind 开发稳定的 C API (Phase 0)，再在 Phase 1 中通过 Rust FFI 直接调用，不再依赖 Python。

### C API 设计

TurboMind 当前只有 pybind11 绑定 (`_turbomind.so`)，无对外 C API。Phase 0 目标是为 TurboMind 创建一个**稳定的、版本化的 C API**，作为 Rust FFI 的基础。

#### 设计原则

1. **ABI 稳定**: 使用 `TM_API_VERSION` 宏版本化，每次破坏性变更递增主版本号
2. **错误处理**: 统一返回 `TM_Status` 码，错误详情通过 `TM_GetLastError()` 获取
3. **内存管理**: 调用者分配/释放 tensor 数据，API 不持有 ownership
4. **同步优先**: 优先提供同步 API，异步版本通过回调扩展
5. **最小依赖**: C API 层只依赖 TurboMind 内部头文件，对外接口干净

#### 核心 API 签名

```c
// 版本和实例管理
TM_Engine     TM_CreateEngine(const char* model_dir, const TM_EngineConfig* config);
void          TM_DestroyEngine(TM_Engine engine);
const char*   TM_GetVersion(void);

// 模型信息查询
int           TM_GetVocabSize(TM_Engine engine);          // 词表大小
int           TM_GetHiddenDim(TM_Engine engine);          // hidden dimension
int           TM_GetMaxBatchSize(TM_Engine engine);       // 最大 batch size
int           TM_GetSessionLen(TM_Engine engine);          // 最大上下文长度
TM_DataType   TM_GetDataType(TM_Engine engine);           // 数据类型 (fp16/bf16/int8/...)

// 推理接口
TM_Status     TM_Forward(TM_Engine          engine,
                         const TM_Input*    input,
                         TM_Output*         output);
TM_Status     TM_ForwardAsync(TM_Engine     engine,
                              const TM_Input* input,
                              TM_Callback     callback);

// Tokenizer
TM_Tokenizer  TM_CreateTokenizer(const char* model_dir);
void          TM_DestroyTokenizer(TM_Tokenizer tokenizer);
TM_Status     TM_Encode(TM_Tokenizer tokenizer, const char* text, int* tokens, int* len);
TM_Status     TM_Decode(TM_Tokenizer tokenizer, const int* tokens, int len, char** text);
void          TM_FreeString(char* str);

// 状态查询
TM_Status     TM_GetScheduleMetrics(TM_Engine engine, TM_ScheduleMetrics* metrics);
```

#### 错误码

| 宏 | 值 | 说明 |
|----|---|------|
| `TM_SUCCESS` | 0 | 成功 |
| `TM_ERR_INVALID_ARG` | 1 | 参数无效 |
| `TM_ERR_OOM` | 2 | 显存不足 |
| `TM_ERR_TIMEOUT` | 3 | 推理超时 |
| `TM_ERR_MODEL_NOT_FOUND` | 4 | 模型文件不存在 |
| `TM_ERR_DEVICE_ERROR` | 5 | GPU 错误 |
| `TM_ERR_INTERNAL` | 99 | 内部错误 |

#### 文件结构

```
src/turbomind/api/
├── CMakeLists.txt
├── turbomind_c_api.h      # 公共 C API 头文件
├── turbomind_c_api.cc     # C API 实现 (调用 TurboMind)
└── test/
    ├── test_c_api.cc      # C API 单元测试
    └── bench_c_api.cc     # 性能基准测试
```

#### 构建方式

```bash
# 随 TurboMind 一起编译
mkdir build && cd build
cmake .. -DLMDEPLOY_BUILD_C_API=ON
make -j$(nproc)
# 产物: lib/libturbomind.so, include/turbomind_c_api.h
```

### 依赖

- **TurboMind C++ 引擎**: LMDeploy 核心
- **PyO3**: Python-Rust FFI
- **Axum**: Rust Web 框架
- **Tokio**: 异步运行时
- **Protobuf**: 序列化

---

## 成功标准

### Phase 0 MVP - C API
- [ ] `turbomind_c_api.h` 头文件发布
- [ ] 单次推理流程跑通 (创建引擎 → Forward → 销毁)
- [ ] 错误处理和内存安全测试通过
- [ ] 性能不低于 pybind11 版本

### Phase 1 MVP - Rust Server
- [ ] HTTP API 基本可用
- [ ] 性能与 Python server 相当
- [ ] 通过基本功能测试

### Phase 2 优化版
- [ ] 性能比 Python server 快 2x
- [ ] Tokenize Cache 实现并生效
- [ ] gRPC API 可用

### Phase 3 完整版
- [ ] 所有核心功能实现
- [ ] 性能达标（见性能指标表）
- [ ] 生产可用

---

## 附录

### A. 技术选型

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| HTTP 框架 | Axum | 高性能，异步生态好 |
| gRPC 框架 | Tonic | gRPC-web 支持，与 Axum 兼容 |
| 运行时 | Tokio | 成熟稳定 |
| 序列化 | Protobuf | 高性能，兼容性好 |
| 缓存 | lru-rs | 简单高效 LRU 实现 |
| FFI | bindgen + libc | Rust 调用 C API 的标准方式 |
| TurboMind 接口 | 自研 C API | Phase 0 开发，替代 PyO3 |

### B. 目录结构

```
lmdeploy-rust-server/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── server.rs
│   ├── router.rs
│   ├── handlers/
│   │   ├── http.rs
│   │   ├── grpc.rs
│   │   └── mod.rs
│   ├── cache/
│   │   ├── tokenizer.rs
│   │   └── mod.rs
│   ├── turbomind/
│   │   ├── bridge.rs
│   │   ├── wrapper.rs
│   │   └── mod.rs
│   ├── proto/
│   │   └── lmdeploy.proto
│   ├── config.rs
│   ├── metrics.rs
│   └── lib.rs
├── tests/
│   ├── integration/
│   ├── unit/
│   └── bench/
├── config/
│   └── default.toml
└── docs/
    ├── api.md
    └── deployment.md
```

### C. 参考资料

- LMDeploy 源码: `/mnt/data/lmdeploy`
- Axum 文档: https://axum.rs/
- Tonic 文档: https://github.com/hyperium/tonic
- PyO3 文档: https://pyo3.rs/

---

**文档版本**: 1.0
**创建日期**: 2026-05-16
**最后更新**: 2026-05-16
**状态**: 待审批