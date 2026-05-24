# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Linting:**

```bash
pre-commit run --all-files
```

Style: PEP8, max line length 120, double quotes, LF endings. C++ source under `src/` uses clang-format.

**Tests:**

```bash
pytest tests/test_lmdeploy                          # all unit tests
pytest tests/test_lmdeploy/test_model.py            # specific file
pytest tests/test_lmdeploy/test_lite/               # quantization tests
pytest tests/test_lmdeploy/test_vl/                 # vision-language tests
```

**Debug logging:**

```bash
LMDEPLOY_LOG_LEVEL=DEBUG python ...
```

**Build (TurboMind C++ extension):**

- Controlled via `setup.py` + CMake. Relevant env vars: `LMDEPLOY_TARGET_DEVICE` (default `cuda`), `DISABLE_TURBOMIND`, `CMAKE_BUILD_TYPE`, `CUDACXX`.
- Requirements split by device: `requirements/runtime_cuda.txt`, `runtime_ascend.txt`, etc.

## Architecture

### Two Backends, One Pipeline

`lmdeploy/pipeline.py` is the main user-facing entry point (`pipeline()` in `api.py`). It instantiates either the **PyTorch engine** (`lmdeploy/pytorch/`) or the **TurboMind engine** (`lmdeploy/turbomind/`) based on config.

### PyTorch Backend

**Model patching** is the core mechanism: HuggingFace models are loaded normally, then their layers are dynamically replaced with optimized LMDeploy implementations.

- `lmdeploy/pytorch/models/module_map.py` — registry mapping HF class names → LMDeploy replacement classes. Device-specific overrides in `DEVICE_SPECIAL_MODULE_MAP`.
- `lmdeploy/pytorch/models/patch.py` — applies the substitutions at runtime via `_get_rewrite_qualname()` / `_class_from_qualname()`.
- `lmdeploy/pytorch/models/` — 40+ per-model files (e.g., `llama.py`, `qwen.py`, `deepseek_v2.py`). Each reimplements attention, MLP, and embeddings using custom kernels.
- `lmdeploy/pytorch/nn/` — reusable optimized modules: `linear/` (AWQ, W8A8, blocked-FP8, LoRA variants), `attention.py`, `norm.py`, `rotary_embedding.py`, `moe/`.
- `lmdeploy/pytorch/kernels/` — Triton/CUDA kernels (e.g., `w8a8_triton_kernels.py`).
- `lmdeploy/pytorch/backends/` — kernel/operator dispatchers per quantization type (FP8, AWQ, CUDA).

**Engine execution flow (key files):**

- `engine.py` — main PyTorch engine.
- `paging/scheduler.py` — sequences → batches; prefill/decode, block eviction, prefix caching (`BlockTrie`).
- `engine/engine_loop.py` — async inference loop.
- (See `pytorch/engine/` and `pytorch/paging/` for full execution detail.)

**Configuration dataclasses** (`lmdeploy/pytorch/config.py`): `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `BackendConfig`, `DistConfig`, `MiscConfig`.

### TurboMind Backend

- Python wrapper: `lmdeploy/turbomind/turbomind.py` (~800 lines). Bridges into `lmdeploy/lib/_turbomind` (pybind11 extension built from `src/turbomind/`).
- Tensor interop via `torch.from_dlpack()` / `_tm.from_dlpack()`.
- Config and model conversion: `lmdeploy/turbomind/deploy/config.py`, `supported_models.py`.
- Parallel config helpers: `update_parallel_config()`, `complete_parallel_config()` in `messages.py`.

### Lite / Quantization

Entrypoints in `lmdeploy/lite/apis/`: `calibrate.py` (main), `auto_awq.py`, `gptq.py`, `smooth_quant.py`.

**Flow:** load HF model → `CalibrationContext` collects activation statistics → scale computation (`lmdeploy/lite/quantization/`) → write quantized weights.

- `lite/quantization/awq.py` — AWQ (NORM_FCS_MAP, FC_FCS_MAP define per-model layer structure).
- `lite/quantization/weight/quantizer.py` — weight quantizer.
- `lite/quantization/activation/observer.py` — activation statistics.
- `lite/modeling/` — model-specific GPTQ implementations (e.g., `internlm2_gptq.py`).
- `lite/utils/cal_qparams.py` — quantization parameter calculation utilities.

Layer/norm/head mappings per model family are defined directly in `calibrate.py` and `awq.py`.

### Vision-Language Models

- `lmdeploy/vl/model/` — VLM preprocessing (InternVL, Qwen-VL, LLaVA, CogVLM, etc.).
- `lmdeploy/vl/media/` — image/video loaders and base classes.
- `lmdeploy/pytorch/multimodal/` — multimodal input handling for the PyTorch engine.
- Reference VLM implementation: `lmdeploy/vl/model/qwen3.py`.

### Other Key Files

- `lmdeploy/messages.py` — core types: `GenerationConfig`, `EngineConfig`, `TurbomindEngineConfig`, `SchedulerSequence`, `MessageStatus`.
- `lmdeploy/model.py` — chat templates; critical for correct conversation formatting.
- `lmdeploy/archs.py` — architecture registry mapping model arch names to runtime patches.
- `lmdeploy/tokenizer.py` — HuggingFace/SentencePiece tokenizer wrapper.
- `lmdeploy/serve/openai/` — OpenAI-compatible API server.

- `lmdeploy-rust-server/tests/benchmark_python_official.py` — 官方 Python TurboMind benchmark 脚本
- `lmdeploy-rust-server/tests/prefill_benchmark_awq.py` — AWQ 模型 prefill 专用 benchmark
- `BENCHMARK_PYTHON_TM_20260518.json` — 历史基准数据 (42,875 tok/s prefill)

## Adding a New PyTorch Model

Use the `/support-new-model` skill for a complete step-by-step guide.

<!-- bv-agent-instructions-v2 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking and [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) (`bv`) for graph-aware triage. Issues are stored in `.beads/` and tracked in git.

### Using bv as an AI sidecar

bv is a graph-aware triage engine for Beads projects (.beads/beads.jsonl). Instead of parsing JSONL or hallucinating graph traversal, use robot flags for deterministic, dependency-aware outputs with precomputed metrics (PageRank, betweenness, critical path, cycles, HITS, eigenvector, k-core).

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). `br` handles creating, modifying, and closing beads.

**CRITICAL: Use ONLY --robot-* flags. Bare bv launches an interactive TUI that blocks your session.**

#### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns everything you need in one call:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command

# Token-optimized output (TOON) for lower LLM context usage:
bv --robot-triage --format toon
```

Before claiming, verify current state with `br show <id> --json` or `br ready --json`. `recommendations` can include graph-important blocked or assigned work; only `quick_ref.top_picks` and non-empty `claim_command` fields represent claimable work.

#### Other bv Commands

| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with unblocks lists |
| `--robot-priority` | Priority misalignment detection with confidence |
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions, cycle breaks |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |

#### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work (no blockers)
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank scores
```

### br Commands for Issue Management

```bash
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export DB to JSONL
```

### Workflow Pattern

1. **Triage**: Run `bv --robot-triage` to find the highest-impact actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

<!-- end-bv-agent-instructions -->
