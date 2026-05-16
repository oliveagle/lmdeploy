#!/usr/bin/env python3
import os
os.environ['LD_LIBRARY_PATH'] = f'/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
os.environ['TM_LOG_LEVEL'] = 'INFO'
os.environ['LMDEPLOY_LOG_LEVEL'] = 'INFO'

from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

print("Creating TurbomindEngineConfig...")
tm_config = TurbomindEngineConfig(
    model_format='awq',
    tp=1,
    cache_max_entry_count=0.4,
    quant_policy=8,
    session_len=8192,
    speculative_config=SpeculativeConfig(
        method='dflash',
        model=draft_model,
        num_speculative_tokens=8,
    ),
)
print(f"speculative_config: {tm_config.speculative_config}")

print("\nCreating Pipeline...")
pipe = pipeline(target_model, backend_config=tm_config)
print("Pipeline created")

print("\nRunning inference...")
gen_config = GenerationConfig(max_new_tokens=128, do_sample=False)
resp = pipe([{"role": "user", "content": "Write a Python quicksort function."}], gen_config=gen_config, sequence_start=True, sequence_end=True, chat_template_kwargs={'enable_thinking': False})
print(f"Response: {resp.text[:100]}")

print("\nChecking DFlash stats...")
try:
    stats = pipe.async_engine.engine.get_dflash_stats()
    print(f"Stats: {stats}")
except Exception as e:
    print(f"Failed to get stats: {e}")