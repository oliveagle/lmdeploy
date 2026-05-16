# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **MoonBit Struct Syntax**: `pub struct Name { field: Type, ... }` with `pub` visibility prefix. Default values are NOT supported in struct definitions; use constructor functions like `pub fn Type::new(args) -> Type { { field: value, ... } }`
- **MoonBit Enum Syntax**: `pub enum Name { VariantA(Type) | VariantB(String, Int) | VariantC }` - union types with or without payload
- **MoonBit Option Type**: `Option[T]` (not `?T`) with `None` and `Some(value)` variants
- **MoonBit Package System**: Each package directory needs a `moon.pkg.json` file with link metadata. Dependencies specified in `deps` array.
- **MoonBit Function Syntax**: `pub fn Type::method(self : Type, arg : Type) -> ReturnType { ... }` - method binding with `Type::` prefix and explicit `self : Type` parameter
- **MoonBit Match**: `match expr { Pattern1 => body1, Pattern2 => body2 }` - pattern matching without `case` keyword
- **HTTP Handler Pattern**: Handlers take `HttpRequest` and return `HttpResponse`, using builder pattern for headers: `HttpResponse::ok().set_header("key", "value").json(body)`
- **SSE Streaming**: Use `text/event-stream` content-type with `data: {...}\n\n` format for Server-Sent Events
- **Batch Processing**: Use size + timeout triggering - process when queue is full OR timeout expires

---

## 2026-05-16 - US-003
- **Implemented**: High-performance HTTP/2 server framework for MoonBit
- **Files changed**:
  - `src/http/http.mbt` - HTTP/2 server with connection pooling, graceful shutdown
  - `src/handlers/http.mbt` - OpenAI-compatible HTTP endpoints (/health, /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings, /metrics, /v1/tokenize, batch endpoints)
  - `src/batch/batch.mbt` - Batch processing with timeout and size-based triggering
  - `src/http/moon.pkg.json` - HTTP package configuration
  - `src/handlers/moon.pkg.json` - Handlers package configuration
  - `src/batch/moon.pkg.json` - Batch package configuration
- **Learnings**:
  - MoonBit does not yet have a mature HTTP framework - created pragmatic FFI-based design
  - HTTP/2 implementation requires foreign function declarations to C libraries (libhttp, nghttp2)
  - Connection pooling implemented with LRU-style cleanup based on idle timeout
  - SSE (Server-Sent Events) streaming requires proper content-type headers and newline-delimited JSON
  - Batch processing uses size + timeout triggering (configurable)
  - All handlers follow OpenAI API format for compatibility with existing clients
  - Helper functions like `int_to_string` need proper implementation - currently placeholder

---
- **Implemented**: Full gRPC API scaffolding for LMDeploy MoonBit server
- **Files changed**:
  - `moon.mod.json` - Module configuration
  - `src/proto/lmdeploy.proto` - Complete Protobuf gRPC service definition (matching Python OpenAI API)
  - `src/protocol/chat_completion.mbt` - Chat completion request/response/stream data types
  - `src/protocol/completion.mbt` - Text completion data types
  - `src/protocol/common.mbt` - Shared types (ErrorResponse, ModelCard, UsageInfo, etc.)
  - `src/grpc/service.mbt` - gRPC service interface with all RPC endpoints
  - `src/error/error.mbt` - Error type hierarchy (InvalidRequest, ModelNotFound, EngineError, etc.)
  - `src/server/server.mbt` - Server scaffolding with config and lifecycle
  - `src/*/moon.pkg` - Package declarations for each module
- **Learnings**:
  - MoonBit requires `moon.pkg` files in each package directory (even if empty)
  - `moon fmt --check` validates formatting, `moon check` validates syntax, `moon build` compiles
  - MoonBit structs don't support default values in field definitions - must use constructor functions
  - The `type_` field name needed for Protobuf `type` field since `type` is reserved in MoonBit
  - All code passes `moon check` and `moon fmt --check` without errors
- **Status**: gRPC service interface and protocol definitions are defined. Actual runtime requires MoonBit HTTP/gRPC bindings (moonbit-http, moonbit-grpc) which are not yet available as published packages.
