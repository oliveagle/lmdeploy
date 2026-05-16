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

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*
- Cache pattern: LRU via `Array[String]` tracking (not linked list) + TTL expiration via `expires_at` field
- Prefix cache: store prefixes for long texts (>50 chars), derive truncated token subset proportionally
- Metrics: immutable struct updates - create new struct with updated fields (MoonBit functional style)
- Hash: djb2 algorithm (simple, fast) instead of SHA-256 (would need FFI)

---

