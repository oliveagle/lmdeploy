# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## 2026-05-16 - US-002
- Tokenize Cache 缓存系统已实现（来自之前迭代）
- 文件：`src/cache/tokenize_cache.mbt` (LRU+TTL 缓存), `src/cache/hash.mbt` (hash 工具), `src/cache/time.mbt` (时间工具)
- Handlers 中集成了占位符的 `get_or_tokenize` 函数
- **Learnings:**
  - `Map::new()` 在 MoonBit 中已弃用，使用 `Map([], capacity=N)` 代替
  - `Array.push()` 返回 Unit，需用 `concat([item])` 返回新数组
  - `Fn` 函数类型语法未定义，回调函数需要其他方式表示
  - 缓存指标包括：total_requests, cache_hits, cache_misses, prefix_hits, evictions
  - Prefix cache 实现：对长文本截取前缀存储部分 tokens

## 2026-05-16 - US-004: 批量推理支持
- 实现了 `/v1/chat/completions` 批量模式支持
- 实现了 `/v1/completions` 批量模式支持
- 批量大小可配置（默认 8）
- 批量请求正确路由到 TurboMind（通过 FFI 占位符）
- 文件变更：
  - `src/protocol/completion.mbt` - 添加了 `BatchCompletionRequest` 和 `BatchCompletionResponse` 结构
  - `src/batch/batch.mbt` - 重写了完整的批处理逻辑，包括：
    - `BatchConfig` - 批量配置（batch_size, timeout_ms）
    - `BatchProcessingResult` - 批量处理结果
    - `process_batch_chat_completions()` - 聊天补全批处理
    - `process_batch_completions()` - 文本补全批处理
    - `build_batch_response()` - 构建批量响应 JSON
  - `src/handlers/http.mbt` - 已有的 `batch_chat_completions_handler` 和 `batch_completions_handler` 正确使用 batch 模块
- **Learnings:**
  - MoonBit 中数组使用 `append()` 方法添加元素（不是 `push()`）
  - `Option[T]` 类型用于表示可能不存在的值（如 `error: Option[String]`）
  - `match` 语句用于模式匹配（如 `match r.error { Some(err) => ..., None => ... }`）
  - 批量处理逻辑：逐个处理请求，分别统计成功和失败数量
  - MoonBit 的 while 循环语法：`while idx < total { ... idx = idx + 1 }`
  - 结构体初始化使用 `{ field1: value1, field2: value2 }` 语法

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*
- Cache pattern: LRU via `Array[String]` tracking (not linked list) + TTL expiration via `expires_at` field
- Prefix cache: store prefixes for long texts (>50 chars), derive truncated token subset proportionally
- Metrics: immutable struct updates - create new struct with updated fields (MoonBit functional style)
- Hash: djb2 algorithm (simple, fast) instead of SHA-256 (would need FFI)
- SSE streaming format: `data: {json}\n\n` per chunk, terminated by `data: [DONE]\n\n`
- Streaming config: env-var overrides with defaults (`LMDEPLOY_STREAM_TIMEOUT_SECS`, `LMDEPLOY_KEEPALIVE_INTERVAL_MS`)
- First token latency tracking: record on first chunk only, cumulative sum for averaging
- HTTP response chaining: `.sse()` sets streaming headers, `.set_header()` chains via immutable updates
- Response headers for streaming: `x-accel-buffering: no` disables nginx buffering for SSE
- `String::to_utf8_bytes()` and `String::from_byte()` for byte/string conversion in MoonBit
- JSON parsing: use pattern matching with `String::find()` to extract field values from JSON strings
- OpenAI API parameters: each handler should parse ~20 parameters using json_util helper functions
- Error response: `build_error_json(code, type, message)` for consistent error formatting
- Route registration: `server.add_route(HttpMethod::Get, path, handler)` for endpoint setup
- Validation: always check required fields (model, input) and return 400 Bad Request if missing

---

## 2026-05-16 - US-005: 流式响应优化
- 实现了完整的 SSE (Server-Sent Events) 流式响应系统
- 文件变更：
  - `src/streaming/config.mbt` - 流式配置（stream_timeout_secs, keepalive_interval_ms, max_chunk_tokens）
  - `src/streaming/metrics.mbt` - 流式指标（total_streams, avg_first_token_latency_us, total_chunks_sent, stream_timeouts）
  - `src/streaming/sse.mbt` - SSE 格式化工具（build_chat_stream_chunk, build_completion_stream_chunk, sse_done_event）
  - `src/streaming/engine.mbt` - 流式引擎集成（stream_chat_completion, stream_completion, is_stream_timeout）
  - `src/streaming/streaming.mbt` - 流式模块入口（聚合所有流式功能）
  - `src/handlers/http.mbt` - 更新了 chat_completions_stream 和 completions_stream 使用新流式模块
- **实现的功能**:
  - SSE 响应格式：`data: {json}\n\n` 每个数据块
  - 流终止标记：`data: [DONE]\n\n`
  - 首次空块（建立流 ID 和 model）
  - Usage 信息仅在最终块发送（符合 OpenAI API 规范）
  - JSON 字符串转义（`\"`, `\\`, `\n`, `\r`, `\t`）
  - Keep-alive 注释支持（`: comment\n\n`）
  - 首次 token 延迟跟踪（微秒级精度）
  - 流超时检测（stream_timeout_secs 配置）
  - Prometheus 指标导出（stream_total, stream_first_token_latency_avg_us, stream_chunks_total, stream_timeouts_total）
- **响应头优化**:
  - `content-type: text/event-stream`
  - `cache-control: no-cache`
  - `connection: keep-alive`
  - `x-accel-buffering: no` （禁用 nginx 缓冲）
  - `x-stream-timeout-secs` （配置的超时时间）
- **Learnings:**
  - MoonBit 模块系统使用 `mod` 关键字导入子模块（如 `mod config`, `mod metrics`）
  - MoonBit 函数式风格：metrics 通过不可变更新传递（如 `metrics.record_chunk()` 返回新 metrics）
  - SSE 格式严格遵循 OpenAI 规范：初始块有 `role: "assistant"`，后续块只有 `delta.content`
  - Python 参考：`completion_stream_generator()` 使用 `async for` 迭代器，MoonBit 使用占位符结构
  - Rust 参考：`StreamMetrics` 使用 AtomicU64，MoonBit 使用不可变结构体（函数式风格）
  - `Option[T]` 类型用于可选值（如 `finish_reason: Option[String]`），使用 `match` 解构
  - `String::to_bytes()` 和 `String::from_byte()` 用于字节/字符串转换
  - Prometheus 指标格式：`# HELP`, `# TYPE` 注释行 + 数据行
  - 流式响应需要在 HTTP 层设置 `sse()` 响应类型
  - 模块级常量：`let stream_config = streaming::StreamingConfig::default()` 在模块初始化时设置

---

## 2026-05-16 - US-006: OpenAI API 兼容性
- 实现了完整的 OpenAI API 兼容层
- 新增 JSON 解析工具模块 `json_util.mbt`，支持解析 string/int/float/bool/string_array
- 新增路由模块 `router.mbt`，集中管理所有 API 端点注册
- 增强 `/v1/chat/completions` handler：支持全部 OpenAI 参数（model, temperature, top_p, top_k, max_tokens, n, presence_penalty, frequency_penalty, seed, logprobs, top_logprobs, tool_choice, repetition_penalty, ignore_eos, skip_special_tokens, do_preprocess, response_format, enable_thinking, min_p, min_new_tokens, stop）
- 增强 `/v1/completions` handler：支持 prompt, echo, logprobs 等 completion 参数
- 实现 `/v1/embeddings` handler：支持 input, encoding_format, dimensions 参数，模型验证
- 实现 `/pooling` handler：支持 pooling_type 配置
- 实现 `/v1/encode` handler：支持 tokenize 缓存命中统计
- 增强 `/v1/tokenize` handler：使用 json_util 解析请求
- 新增 `build_error_json` 和 `build_chat_completion_json` 辅助函数到 common.mbt
- 所有 handler 增加 model/input 验证和 400 错误响应
- 文件变更：
  - `src/protocol/json_util.mbt` - 新增 JSON 解析工具
  - `src/protocol/common.mbt` - 添加 build_error_json, build_chat_completion_json 辅助函数
  - `src/router/router.mbt` - 新增路由模块，集中管理端点注册
  - `src/handlers/http.mbt` - 全面更新所有 handler 使用 json_util，增强参数解析和验证
- **Learnings:**
  - MoonBit 中 JSON 解析需要手动实现字符串提取，无标准库 JSON parser
  - OpenAI API 参数众多，每个 handler 需要解析 ~20 个字段
  - 错误响应格式统一使用 build_error_json 构建，保持与 Python API 一致
  - SSE 流式响应需要同时设置 content-type 和 x-accel-buffering: no 头
  - 路由模块使用 Http2Server::add_route 方法注册 handler
  - 所有 handler 都需要验证 model 参数为空时返回 400 错误

---

## 2026-05-16 - US-007: 配置管理
- 实现了完整的配置管理系统（JSON/JSONC 配置文件 + 环境变量覆盖）
- 文件变更：
  - `src/config/config.mbt` - 主配置模块，定义 `AppConfig` 结构体
  - `src/config/server.mbt` - 服务器配置（HTTP/2, 连接池, 流式, 批处理）
  - `src/config/model.mbt` - 模型配置（路径, 上下文长度, 批大小）
  - `src/config/cache.mbt` - 缓存配置（tokenizer cache 大小, TTL）
  - `src/config/logging.mbt` - 日志配置（级别, 格式）
  - `src/config/metrics.mbt` - 指标配置（Prometheus）
  - `src/config/moon.pkg.json` - 模块依赖配置
  - `config/default.jsonc` - 默认配置文件
  - `src/server/server.mbt` - 更新为使用 `config::AppConfig`
  - `src/streaming/config.mbt` - 更新为可从 `ServerConfig` 创建
  - `src/streaming/moon.pkg.json` - 添加对 config 模块的依赖
- **实现的功能**:
  - `AppConfig::default()` - 默认配置值
  - `AppConfig::load()` - 从默认路径加载配置
  - `AppConfig::load_with_path()` - 从指定路径加载
  - `AppConfig::from_json()` - 从 JSON 字符串解析
  - `AppConfig::from_env()` - 应用环境变量覆盖
  - 各子配置结构体的访问器方法
  - JSON 解析辅助函数（`parse_json_string_field`, `parse_json_int_field`, `parse_json_bool_field`）
  - 环境变量读取占位符（`get_env_string`, `get_env_int`, `get_env_bool` - TODO: FFI）
  - 文件系统访问占位符（`path_exists`, `read_file_content` - TODO: FFI）
- **配置覆盖优先级**: defaults → config file → env vars
- **环境变量命名**: `LMDEPLOY_SERVER_*`, `LMDEPLOY_MODEL_*`, `LMDEPLOY_CACHE_*`, `LMDEPLOY_LOG_*`, `LMDEPLOY_METRICS_*`
- **默认配置路径**: `/etc/lmdeploy/config.jsonc` 或 `/etc/lmdeploy/config.json`
- **Learnings**:
  - 配置模块使用 JSONC 格式（支持注释，符合 CLAUDE.md 规范）
  - 各子配置模块独立，通过 `AppConfig` 聚合
  - JSON 解析使用简单的字符串查找模式匹配（无标准 JSON 库）
  - `int_to_string` 实现了递归转换，支持到 10000 的数字
  - 服务器配置中的流式设置可直接传递给流式模块（`StreamingConfig::from_server_config`）
  - 配置系统遵循 Rust 版本的结构（`config.rs` → `config/*.mbt`）
  - 配置热重载需要 SIGHUP 信号处理（TODO: 需要 FFI 支持）

---

## 2026-05-16 - US-008: 日志和监控
- 实现了完整的结构化日志和 Prometheus 监控系统
- 文件变更：
  - `src/logger/logger.mbt` - 新增日志模块（LogLevel, LoggerConfig, LogEntry, Logger）
  - `src/metrics/metrics.mbt` - 新增指标模块（MetricsCollector, RequestMetrics, Histogram）
  - `src/logger/moon.pkg.json` - 日志模块依赖配置
  - `src/metrics/moon.pkg.json` - 指标模块依赖配置
  - `src/handlers/moon.pkg.json` - 添加对 logger 和 metrics 的依赖
  - `src/handlers/http.mbt` - 集成日志和指标到所有 handlers
- **实现的功能**:
  - **日志系统**:
    - `LogLevel` 枚举：Debug, Info, Warn, Error
    - `LoggerConfig`：配置日志级别、JSON 格式、组件名、时间戳、请求 ID
    - `LogEntry`：结构化日志条目（level, message, component, timestamp, request_id, fields, error）
    - `Logger`：提供 debug/info/warn/error 级别的日志方法
    - JSON 格式日志输出（符合现代日志标准）
    - 纯文本格式日志输出（开发调试）
  - **指标系统**:
    - `MetricsCollector`：收集所有 Prometheus 指标
    - `RequestMetrics`：单个请求的指标（endpoint, method, status_code, duration_ms, tokens, cache_hit, stream）
    - `Histogram`：Prometheus 直方图（请求延迟、Prefill 延迟、解码吞吐量）
    - 指标类型：Counter（计数器）、Gauge（仪表盘）、Histogram（直方图）
    - Prometheus 格式导出（`# HELP`, `# TYPE` 注释 + 数据行）
  - **Handler 集成**:
    - 每个请求开始时记录日志（`log_info`）
    - 每个请求结束时记录日志和指标（`end_request_metrics`）
    - 错误请求记录错误日志
    - `/metrics` 端点导出所有收集的指标
  - **指标列表**:
    - `active_requests`：当前活跃请求数（Gauge）
    - `http_requests_total`：HTTP 请求总数（Counter, 按端点、方法、状态码分组）
    - `request_duration_seconds`：请求延迟直方图
    - `prefill_duration_seconds`：Prefill 延迟直方图
    - `decode_tokens_per_second`：解码吞吐量直方图
    - `tokens_generated_total`：生成的 token 总数
    - `prompt_tokens_total`：处理的 prompt token 总数
    - `cache_hits_total` / `cache_misses_total`：缓存命中/未命中统计
    - `streaming_requests_total` / `non_streaming_requests_total`：流式/非流式请求统计
    - `errors_total`：错误总数（按类型分组）
- **Learnings**:
  - MoonBit 函数式风格：指标收集器通过不可变更新传递（`metrics.record_request()` 返回新的 metrics）
  - 模块级常量：`let request_metrics = metrics::MetricsCollector::new()` 和 `let app_logger = logger::Logger::with_component("handlers")`
  - 日志级别比较：使用 `ordinal()` 方法进行数值比较（Debug=0, Info=1, Warn=2, Error=3）
  - JSON 字符串转义：需要处理 `"`, `\`, `\n`, `\r`, `\t` 等特殊字符
  - Prometheus 直方图格式：`*_bucket{le="x"}` 计数 + `*_sum` 总和 + `*_count` 计数
  - 直方图桶（buckets）数组包含 `+Inf` 作为最后一个桶
  - MoonBit 中 `Float` 类型使用 `+Inf` 表示正无穷
  - 日志和指标是分离的关注点：日志用于人类调试，指标用于机器监控
  - Python 参考：`lmdeploy/metrics/metrics_processor.py` 使用 asyncio.Queue 异步处理指标
  - Rust 参考：`lmdeploy-rust-server/src/metrics.rs` 使用 AtomicU64 进行线程安全计数
  - MoonBit 使用不可变结构体 + 模块级变量（生产环境需要考虑并发访问）
  - `Map` 类型使用 `.insert()` 返回新 Map（函数式风格）
  - 直方图 `observe()` 方法更新所有桶的计数（桶值 <= 观察值）

---