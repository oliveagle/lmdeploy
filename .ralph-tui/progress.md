# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-16] - US-004: 批量推理支持

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
