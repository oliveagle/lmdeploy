# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **MoonBit Struct Syntax**: `pub struct Name { field: Type, ... }` with `pub` visibility prefix. Default values are NOT supported in struct definitions; use constructor functions like `pub fn Type::new(args) -> Type { { field: value, ... } }`
- **MoonBit Enum Syntax**: `pub enum Name { VariantA(Type) | VariantB(String, Int) | VariantC }` - union types with or without payload
- **MoonBit Option Type**: `Option[T]` (not `?T`) with `None` and `Some(value)` variants
- **MoonBit Package System**: Each package directory needs a `moon.pkg` file (can be empty). Subpackages referenced via `@path` annotation (e.g., `@server`)
- **MoonBit Function Syntax**: `pub fn Type::method(self : Type, arg : Type) -> ReturnType { ... }` - method binding with `Type::` prefix and explicit `self : Type` parameter
- **MoonBit Match**: `match expr { Pattern1 => body1, Pattern2 => body2 }` - pattern matching without `case` keyword

---

## 2026-05-16 - US-001
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
