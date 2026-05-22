# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-22] - lmdeploy-109 (Completed)
- **Verified**: C++ AWQ weight loading already fully implemented
- Files verified:
  - `src/turbomind/capi/turbomind_c.cc` - `CreateAwqLinearConfig()`, `ParseHfConfig()`, `MapHuggingFaceWeightToTurboMind()`, `LoadWeightsFromSafetensors()`
  - `src/turbomind/models/linear_weight.cc` - `LinearWeight::prepare()` handles AWQ format conversion
  - `src/turbomind/core/data_format.cc` - `ResolveLinearWeightFormat()` for AWQ kUint4 format
- **Learnings:**
  - AWQ detection: `ParseHfConfig()` reads `quantization_config.quant_method == "awq"` from config.json
  - AWQ config: `CreateAwqLinearConfig()` sets `format = ResolveLinearWeightFormat(data_type, kUint4, awq_group_size, 1)`
  - Weight mapping: `.qweight` → `.weight`, `.qzeros` → `.zeros`, `.weight_scale` → `.scales`
  - Format conversion: `LinearWeight::prepare()` unpacks 4-bit weights, fuses scales/zeros
  - All linear layers (attention, FFN, MoE experts, DeltaNet) use `CreateAwqLinearConfig` when `is_awq=true`

---

