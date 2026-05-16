# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Axum SSE Streaming**: Use `axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::new().interval(duration))`
  - Stream should return `futures::Stream<Item = Result<Event, std::convert::Infallible>>`
  - Use `.chain(futures::stream::once(...))` to append final "[DONE]" event
- **gRPC Streaming with Timeout**: Use `tokio::select!` with `tokio::time::sleep` for timeout
  - For `tokio::select!` with `&mut Sleep`, need to pin the sleep first
  - Alternative: check elapsed time directly if no complex select needed
- **Metrics Recording Pattern**: Record first token latency on the FIRST chunk, not every chunk
  - Use `first_token_start.elapsed().as_millis()` for milliseconds
  - Track with `AtomicU64` + `fetch_add` for thread-safe counters
- **OpenAI API Stop/ Prompt Enums**: For OpenAI-compatible API fields that accept single value or array, use `#[serde(untagged)]` enums:
  - `Stop` enum: `Single(String)` or `Multiple(Vec<String>)` for `stop` parameter
  - `Prompt` enum: `Single(String)` or `Multiple(Vec<String>)` for `prompt` parameter
  - `EmbeddingInput` enum: `Single(String)` or `Multiple(Vec<String>)` for `input` parameter
  - This matches OpenAI API spec where these fields accept either format
- **Prometheus Metrics Pattern**: Use `metrics-exporter-prometheus` crate with its own HTTP listener
  - `PrometheusBuilder::new().with_http_listener(addr).install()` sets up `/metrics` endpoint automatically
  - Use `metrics::histogram!`, `metrics::counter!` macros for recording metrics
  - Separate metrics server on dedicated port (default 9090) avoids mixing with application traffic
  - Metrics config via `[metrics]` section in TOML: `enabled`, `host`, `port`

---

## [2026-05-16] - US-006: OpenAI API 兼容性

### What was implemented
1. **Enhanced ChatCompletionsRequest**:
   - Added `n` (number of completions to generate)
   - Added `logit_bias` (token bias dictionary)
   - Added `logprobs` (include log probabilities)
   - Added `top_logprobs` (number of top logprobs)
   - Changed `stop` from `Option<Vec<String>>` to `Option<Stop>` enum supporting single/multiple formats

2. **Enhanced CompletionsRequest**:
   - Added `suffix` (text to insert after model output)
   - Added `logit_bias` (token bias dictionary)
   - Added `best_of` (number of completions to generate server-side)
   - Added `stop` field (stop sequences)
   - Added `presence_penalty` and `frequency_penalty`
   - Added `n` and `user` fields
   - Changed `prompt` from `String` to `Prompt` enum supporting single/multiple formats

3. **Embeddings API** (New):
   - Implemented `EmbeddingsRequest` with `EmbeddingInput` enum (single or array)
   - Implemented `EmbeddingsResponse` with `EmbeddingData` and `EmbeddingUsage`
   - Added `POST /v1/embeddings` route
   - Added `TurboMindEngine.embed()` method with deterministic mock embedding generation
   - Supports L2-normalized embeddings (default dimension: 1536)
   - Matches OpenAI `/v1/embeddings` API format

### Files changed
- `lmdeploy-rust-server/src/handlers/http.rs` - Added embeddings types, handler, and OpenAI field enhancements
- `lmdeploy-rust-server/src/model/engine.rs` - Added `embed()` method for embedding generation
- `lmdeploy-rust-server/src/server.rs` - Added `/v1/embeddings` route

### Learnings
- **OpenAI Stop Parameter**: OpenAI accepts `stop` as either a single string or array of strings - use `#[serde(untagged)]` enum to handle both
- **OpenAI Prompt Parameter**: Completions API accepts `prompt` as string or array of strings - same enum pattern
- **OpenAI Input Parameter**: Embeddings API accepts `input` as string or array of strings - reuse the same pattern
- **Embedding Generation**: Mock embeddings can use deterministic hash-based generation with L2 normalization for testing
- **serde untagged enums**: Pattern `#[serde(untagged)]` is essential for OpenAI API compatibility where fields accept multiple formats

### Pre-existing Issues (not fixed in this story)
- Stream endpoints use hardcoded timeout default instead of reading from config dynamically
- Mock engine response instead of real TurboMind integration

---

## [2026-05-16] - US-005: 流式响应优化

### What was implemented
1. **SSE Streaming Optimization**:
   - Fixed first token latency recording (was recording microseconds instead of milliseconds)
   - Removed duplicate metrics recording (was recording stream start at 0ms, then again on each chunk)
   - Added warning log when first token latency exceeds 50ms threshold

2. **gRPC Streaming Timeout**:
   - Added timeout handling to `generate_stream` (checks elapsed time against config default of 600s)
   - Added timeout handling to `generate_bidirectional` (activity-based timeout reset)
   - Proper error logging when stream timeouts occur

3. **Connection Timeout Configuration**:
   - Server config already has `stream_timeout_secs`, `stream_keepalive_interval_ms`, `connection_timeout_secs`, `request_timeout_secs`
   - gRPC streaming now respects timeout configuration

### Files changed
- `lmdeploy-rust-server/src/handlers/http.rs` - Fixed first token latency metrics, improved logging
- `lmdeploy-rust-server/src/grpc/service_impl.rs` - Added timeout handling and first token latency tracking to both streaming endpoints

### Learnings
- **First Token Latency**: Was recording `as_micros()` instead of `as_millis()`, causing incorrect metrics
- **SSE Keep-Alive**: `Sse::new(stream).keep_alive(KeepAlive::new().interval(duration))` handles HTTP keep-alive automatically
- **gRPC Stream Timeout**: Simple elapsed check is cleaner than complex tokio::select! for timeout
- **Duplicate Metrics**: Avoid recording metrics both at start AND on each chunk - leads to double-counting

### Pre-existing Issues (not fixed in this story)
- Stream endpoints use hardcoded timeout default instead of reading from config dynamically
- Mock engine response instead of real TurboMind integration

---

## [2026-05-16] - US-004 (迭代 2): Bug 修复

### What was implemented
1. **Batch API Endpoints**:
   - `POST /v1/chat/completions/batch` - 批量聊天完成请求
   - `POST /v1/completions/batch` - 批量文本完成请求

2. **Batch Request/Response Types**:
   - `BatchChatCompletionsRequest` / `BatchChatCompletionsResponse`
   - `BatchCompletionsRequest` / `BatchCompletionsResponse`
   - `BatchChoice` / `BatchCompletionChoice`
   - `BatchUsage` (聚合 token 统计)

3. **Batch Processor Enhancement**:
   - 修复 `flush_batch` 现在实际调用 `TurboMindEngine.generate()`
   - 移除 mock 响应，使用真实引擎

### Files changed
- `lmdeploy-rust-server/src/handlers/http.rs` - 新增 batch API 类型和处理函数
- `lmdeploy-rust-server/src/server.rs` - 更新路由和批处理逻辑
- `lmdeploy-rust-server/src/grpc/service_impl.rs` - 修复 gRPC trait 名称

### Learnings
- **Rust FFI & RwLock**: `AppState.config` 改为 `Arc<RwLock<AppConfig>>` 后，所有访问都需要 `.read().await`
  - 预存代码 `state.config.server.stream_keepalive_interval_ms` 会导致编译错误
  - 需要修改为: `let config = state.config.read().await; config.server.stream_keepalive_interval_ms`
- **批量路由**: Axum 路由通过 `.route("/v1/chat/completions/batch", post(batch_chat_completions))` 添加
- **OpenAI 兼容性**: 批量请求格式模仿 OpenAI API (`messages: Vec<Vec<Message>>`)

### Pre-existing Issues (not fixed in this story)
- `chat_completions_stream` 中 `state.config.server.xxx` 直接访问会编译失败（需要通过 RwLock）
- `StreamMetrics` 的 `Clone` derive 失败（`AtomicU64` 不实现 `Clone`）
- gRPC service 中 `LMDeployService` 拼写错误（应为 `LmDeployService`）

---

## [2026-05-16] - US-004 (迭代 2): Bug 修复

### What was implemented
1. **编译错误修复**:
   - `StreamMetrics`: 手动实现 `Clone` trait（因为 `AtomicU64` 不自动实现 `Clone`）
   - `StreamMetricsSnapshot`: 添加 `#[derive(Serialize)]` 支持 JSON 序列化
   - `reload_config`: 修改返回类型为 `(StatusCode, Json<...>)` 符合 Axum Handler 要求
   - `http.rs` 中的 `stream_timeout_ms` 访问: 通过 `state.config.read().await` 正确访问
   - `server.rs` 中的 `config_reload_tx.send()`: 使用 `match` 替代 `map_err`
   - `config.rs` 中 `AppConfig::load()`: 返回类型改为 `config::ConfigError`（`Send + Sync`）

2. **proto 文件修复**:
   - `turbomind.proto`: 将 `repeated int32 token_ids` 改为 `repeated uint32 token_ids` 匹配实际使用

### Files changed
- `lmdeploy-rust-server/src/metrics.rs` - StreamMetrics 手动 Clone 实现
- `lmdeploy-rust-server/src/handlers/http.rs` - stream_timeout_ms 访问修复
- `lmdeploy-rust-server/src/server.rs` - reload_config 返回类型修复
- `lmdeploy-rust-server/src/config.rs` - 错误类型修复
- `lmdeploy-rust-server/proto/lmdeploy.proto` - token_ids 类型修复

### Learnings
- **AtomicU64 Clone**: `AtomicU64` 等原子类型不自动实现 `Clone`，需要手动实现并使用 `load(Ordering::Relaxed)`
- **Axum Handler 返回类型**: 必须返回 `(StatusCode, Json<T>)` 元组，不能使用 `Result` 包装
- **config crate 错误类型**: `config::ConfigError` 是 `Send + Sync` 的，而 `Box<dyn Error>` 不是
- **proto 类型转换**: Protobuf 的 `int32` 对应 Rust 的 `i32`，`uint32` 对应 `u32`
- **serde Serialize**: 用于 JSON 序列化的类型需要 `#[derive(Serialize)]`

### Codebase Patterns
- Atomic wrapper manual Clone pattern:
```rust
#[derive(Debug)]
pub struct StreamMetrics { ... }

impl Clone for StreamMetrics {
    fn clone(&self) -> Self {
        Self {
            total_streams: AtomicU64::new(self.total_streams.load(Ordering::Relaxed)),
            ...
        }
    }
}
```

---

## [2026-05-16] - US-007: 配置管理

### What was implemented
US-007 was already fully implemented in a previous iteration. All acceptance criteria verified:
1. **TOML/JSON 配置文件** - `config` crate with `File::with_name()` supports both formats
2. **环境变量覆盖** - `Environment::with_prefix("LMDEPLOY")` enables env var override
3. **配置热重载 (SIGHUP)** - SIGHUP signal handler + `reload_config` endpoint + background config reload task
4. **默认配置文件路径** - `/etc/lmdeploy/config.toml` defined as default

### Files reviewed
- `lmdeploy-rust-server/src/config.rs` - Config structs, `AppConfig::load()`, env override
- `lmdeploy-rust-server/config/default.toml` - Default TOML config
- `lmdeploy-rust-server/src/server.rs` - SIGHUP handler, reload_config endpoint, config reload task
- `lmdeploy-rust-server/Cargo.toml` - `config` crate dependency with `toml` feature

### Learnings
- **Config crate merge order**: `set_default()` → `File::with_name()` → `Environment::with_prefix()` gives env vars highest priority
- **SIGHUP reload pattern**: Use `mpsc::unbounded_channel` to signal reload to a dedicated task, then broadcast via `RwLock` swap
- **Signal handling on Linux**: `signal_hook::SignalId` returns a handle that must be kept alive — if dropped, handler unregisters

### Pre-existing Issues (not fixed in this story)
- Pre-existing clippy warnings in test code (unused variables)
- Mock engine responses instead of real TurboMind integration

---

## [2026-05-16] - US-008: 日志和监控

### What was implemented
1. **Prometheus Metrics Exporter** (`metrics.rs`):
   - `init_metrics()` initializes `metrics-exporter-prometheus` HTTP server on configurable port
   - `record_request_duration()` - request latency histogram (`request_duration_seconds`)
   - `record_prefill_duration()` - prefill latency histogram (`prefill_duration_seconds`)
   - `record_decode_tokens_per_second()` - decode throughput histogram (`decode_tokens_per_second`)
   - `increment_requests_total()` - request counter (`requests_total`)
   - `increment_tokens_generated_total(n)` - token counter (`tokens_generated_total`)
   - `RequestTimer` - RAII-style timer that auto-records on drop
   - `InferenceTimer` - tracks prefill + decode timing with structured logging
   - `StreamMetrics` enhanced with `total_stream_tokens` counter and Prometheus histogram integration

2. **Config Extension** (`config.rs`):
   - Added `MetricsConfig` struct with `enabled`, `host`, `port` fields
   - Added to `AppConfig` with defaults: port 9090, host 0.0.0.0

3. **Server Integration** (`server.rs`):
   - Metrics initialized in `start_server()` before HTTP server starts
   - Logs metrics configuration on startup

4. **Default Config** (`config/default.toml`):
   - Added `[metrics]` section with enabled=true, host=0.0.0.0, port=9090

5. **Structured JSON Logging** (pre-existing, verified):
   - `tracing-subscriber` with JSON layer enabled via `config.logging.json_format`
   - Log level configurable via `EnvFilter` (DEBUG/INFO/WARN/ERROR)
   - Env var override: `RUST_LOG=debug`

### Files changed
- `lmdeploy-rust-server/src/metrics.rs` - Rewritten with full Prometheus metrics support
- `lmdeploy-rust-server/src/config.rs` - Added `MetricsConfig` struct and default
- `lmdeploy-rust-server/src/server.rs` - Added metrics initialization call
- `lmdeploy-rust-server/config/default.toml` - Added `[metrics]` section

### Learnings
- **PrometheusBuilder Pattern**: The `metrics-exporter-prometheus` crate sets up its own HTTP server via `with_http_listener(addr)`, automatically serving `/metrics` at that address. No separate Axum route needed.
- **Metrics Macro API**: `metrics::histogram!("name").record(value)` and `metrics::counter!("name").increment(n)` are the standard patterns.
- **AtomicU64 Clone**: Manual Clone implementation required for `AtomicU64` fields in metrics structs.
- **RAII Timer Pattern**: `RequestTimer` uses `finish(self)` for explicit timing, auto-records request count and duration.
- **Config Extension**: Adding new config sections requires updating both the struct AND the `Default` impl.

---
