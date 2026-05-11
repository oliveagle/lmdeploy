# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*EP=4 Implementation Pattern:*
- EP config flows: TurbomindEngineConfig.ep → converter.py → config.yaml → turbomind.cc → EngineParam/MoeParam
- Expert sharding: Python module.py MoeFfn.apply() exports only local expert range; C++ LlamaDenseWeight creates only local experts
- EP communication: MoeFfnLayer uses all_reduce for EP>1 (same as TP for EP=4,TP=1 case)
- Key files: llama_params.h (EngineParam, MoeParam), LlamaDenseWeight.cc (expert assignment), module.py (weight export)

---

## 2026-05-11 - STORY-003: C++ 核心实现验证

**结论**: Turbomind EP=4 C++ 核心实现已经完整，所有验收标准已满足

**已验证的实现**:
1. `EngineParam.mlp_ep_size/mlp_ep_rank` - `llama_params.h:161-162` ✅
2. `MoeFfnLayer` EP 支持 - `moe_ffn_layer.h:41-42`, `moe_ffn_layer.cc:27-28` ✅
3. 专家分片逻辑 - `LlamaDenseWeight.cc:572-587` (ep_first_expert_, ep_num_experts_) ✅
4. MoE 权重加载 - `converter.py:280-282` + `module.py:204-218` + `LlamaDenseWeight.cc:608-621` ✅
5. EP 集合通信 - `moe_ffn_layer.cc` all_reduce for EP>1 ✅

**参数传递链路 (完整)**:
```
TurbomindEngineConfig.ep → converter.py → config.yaml → turbomind.cc → EngineParam → MoeParam
```

**Learnings:**
- EP=4, TP=1 场景下 EP group 与 TP group 重合，通信复用 TP 的 all_reduce
- 真正的专家分片发生在两个阶段: Python 权重导出时 (只导出本地专家) 和 C++ 权重创建时 (只创建本地专家)
- 测试文件已存在: `tests/test_lmdeploy/test_turbomind/test_ep4_model_loading.py`

