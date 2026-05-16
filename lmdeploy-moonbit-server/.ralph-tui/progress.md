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

---

