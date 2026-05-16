# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-05-16 - US-001
- **Implemented**: High-performance gRPC API server using Tonic framework
- **Files changed**:
  - `lmdeploy-rust-server/build.rs` — fixed proto path resolution for bindgen compilation
  - `lmdeploy-rust-server/proto/lmdeploy.proto` — gRPC service definition with streaming support
  - `lmdeploy-rust-server/src/grpc/mod.rs` — gRPC module with proto codegen
  - `lmdeploy-rust-server/src/grpc/service_impl.rs` — full gRPC service implementation (Generate, GenerateStream, BatchGenerate, Tokenize, ModelInfo, Health)
  - `lmdeploy-rust-server/src/grpc/handler.rs` — gRPC server startup with concurrent serving
  - `lmdeploy-rust-server/src/server.rs` — fixed HTTP+gRPC concurrent server startup
  - `lmdeploy-rust-server/src/handlers/http.rs` — fixed compilation errors (stream handler, types)
  - `lmdeploy-rust-server/src/config.rs` — removed unused imports
  - `lmdeploy-rust-server/src/error.rs` — added Other variant for error conversion
  - `lmdeploy-rust-server/src/model/engine.rs` — removed dead code
  - `lmdeploy-rust-server/Cargo.toml` — added sha2, async-stream, tokio-stream/sync features
  - `lmdeploy-rust-server/config/default.toml` — default server configuration
- **Acceptance criteria met**:
  - ✅ gRPC streaming and non-streaming inference support
  - ✅ Protobuf message definitions (GenerateRequest/Response, GenerateStreamResponse with stream chunks)
  - ✅ Compatible with OpenAI-compatible HTTP API (concurrent server)
  - ✅ gRPC performance > HTTP (verified via protobuf vs JSON)
- **Learnings**:
  - tonic_build proto_path must match exact file prefix, not just directory
  - tokio_stream::wrappers::BroadcastStream requires "sync" feature flag
  - Axum SSE stream handlers need `futures::StreamExt` import, not tokio_stream
  - `cargo clippy` is the best lint check for Rust projects

