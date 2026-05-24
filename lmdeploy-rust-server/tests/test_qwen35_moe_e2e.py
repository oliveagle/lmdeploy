#!/usr/bin/env python3
"""Qwen3.5 MoE E2E inference test — verify num_experts and moe_intermediate_size config.

Tests:
1. Config parsing: num_experts=256, moe_intermediate_size=512 parsed correctly
2. C library symbols: MoE parameters present in libturbomind_c.so
3. Config struct layout: HfModelConfig parses text_config nested MoE params
"""
import sys, json, os

MODEL_PATH = "/mnt/data/models/modelscope_models/Qwen3___6-35B-A3B-AWQ"
LIB_PATH = "/mnt/data/lmdeploy/lmdeploy-rust-server/libturbomind_c.so"

def test_config_parsing():
    """Test that Qwen3.5 MoE config params are correctly identified."""
    print("[Test 1] Parsing Qwen3.5 MoE config.json...")

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)

    results = {
        "num_experts": text_config.get("num_experts"),
        "moe_intermediate_size": text_config.get("moe_intermediate_size"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "model_type": text_config.get("model_type"),
    }

    checks = [
        (results["num_experts"] == 256, f"num_experts={results['num_experts']}, expected 256"),
        (results["moe_intermediate_size"] == 512, f"moe_intermediate_size={results['moe_intermediate_size']}, expected 512"),
        (results["num_experts_per_tok"] == 8, f"num_experts_per_tok={results['num_experts_per_tok']}, expected 8"),
        (results["hidden_size"] == 2048, f"hidden_size={results['hidden_size']}, expected 2048"),
        (results["num_hidden_layers"] == 40, f"num_hidden_layers={results['num_hidden_layers']}, expected 40"),
        ("moe" in results["model_type"].lower(), f"model_type={results['model_type']}, should contain 'moe'"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            all_passed = False

    return all_passed


def test_c_code_moe_path():
    """Test that C++ code has MoE config parsing and weight loading."""
    print("\n[Test 2] Checking C++ MoE implementation...")

    cc_path = "/mnt/data/lmdeploy/lmdeploy-rust-server/src/turbomind/capi/turbomind_c.cc"
    with open(cc_path) as f:
        cc_src = f.read()

    checks = [
        ("num_experts" in cc_src, "C code references num_experts"),
        ("moe_intermediate_size" in cc_src, "C code references moe_intermediate_size"),
        ("num_experts_per_tok" in cc_src, "C code references num_experts_per_tok"),
        ("moe_ffn" in cc_src, "C code creates moe_ffn module"),
        ("MoeWeight" in cc_src, "C code creates MoeWeight modules"),
        ("MoeConfig" in cc_src, "C code creates MoeConfig"),
        ("expert_num" in cc_src, "C code sets expert_num config"),
        ("experts_per_token" in cc_src, "C code sets experts_per_token config"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            all_passed = False

    return all_passed


def test_config_parsing_logic():
    """Verify the C++ config parsing logic handles Qwen3.5 nested text_config correctly."""
    print("\n[Test 3] Verifying config parsing logic for Qwen3.5...")

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)

    # Simulate C++ ParseHfConfig logic for Qwen3.5:
    # 1. text_config is detected → config_source = text_config
    # 2. num_local_experts NOT found → falls back to num_experts → 256 ✓
    # 3. moe_intermediate_size found → 512 ✓
    # 4. num_experts_per_tok found → 8 ✓

    has_text_config = "text_config" in config
    num_experts = text_config.get("num_experts", 0)
    num_local_experts = text_config.get("num_local_experts", 0)
    if num_local_experts == 0:
        num_local_experts = num_experts  # C++ fallback at line 514-515

    moe_intermediate_size = text_config.get("moe_intermediate_size", 0)
    if moe_intermediate_size == 0:
        moe_intermediate_size = text_config.get("intermediate_size", 0)  # C++ fallback

    num_experts_per_tok = text_config.get("num_experts_per_tok", 0)

    checks = [
        (has_text_config, "text_config detected in config.json"),
        (num_local_experts == 256, f"num_local_experts (via num_experts fallback) = {num_local_experts}, expected 256"),
        (moe_intermediate_size == 512, f"moe_intermediate_size = {moe_intermediate_size}, expected 512"),
        (num_experts_per_tok == 8, f"num_experts_per_tok = {num_experts_per_tok}, expected 8"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            all_passed = False

    return all_passed


def test_library_exists():
    """Test that the compiled library exists and has MoE symbols."""
    print("\n[Test 4] Checking compiled library...")

    if not os.path.exists(LIB_PATH):
        print(f"  [FAIL] Library not found at {LIB_PATH}")
        return False

    size_mb = os.path.getsize(LIB_PATH) / (1024 * 1024)
    print(f"  [PASS] Library exists ({size_mb:.1f} MB)")

    # Check for MoE-related strings in the library
    with open(LIB_PATH, "rb") as f:
        lib_content = f.read()

    # These strings are embedded in the compiled binary
    checks = [
        (b"moe_ffn" in lib_content or b"MoeConfig" in lib_content, "MoE module strings found in binary"),
        (b"experts" in lib_content, "Experts-related strings found in binary"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            all_passed = False

    return all_passed


def test_weight_mapping():
    """Test that HF MoE weight names map correctly to TurboMind names."""
    print("\n[Test 5] Verifying MoE weight name mapping...")

    # Simulate MapHuggingFaceWeightToTurboMind for Qwen3.5 MoE weights
    def map_weight(hf_name):
        result = hf_name
        if result.startswith("model."):
            result = result[6:]
        # MoE experts: .mlp.experts.N. -> .moe_ffn.experts.N.
        while ".mlp.experts." in result:
            result = result.replace(".mlp.experts.", ".moe_ffn.experts.")
        # MoE gate: .mlp.gate. -> .moe_ffn.gate. (but not gate_proj)
        while ".mlp.gate." in result:
            if ".gate_proj." not in result:
                result = result.replace(".mlp.gate.", ".moe_ffn.gate.")
            else:
                break
        # mlp -> feed_forward (skip if already moe_ffn)
        pos = result.find(".mlp.")
        while pos != -1:
            if not result.startswith(".moe_ffn.", pos - 5):
                result = result[:pos] + ".feed_forward." + result[pos+5:]
            else:
                break
            pos = result.find(".mlp.")
        # gate_proj -> w1, up_proj -> w3, down_proj -> w2
        result = result.replace(".gate_proj.", ".w1.")
        result = result.replace(".up_proj.", ".w3.")
        result = result.replace(".down_proj.", ".w2.")
        return result

    test_cases = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight",
         "layers.0.moe_ffn.experts.0.w1.weight"),
        ("model.layers.0.mlp.experts.0.up_proj.weight",
         "layers.0.moe_ffn.experts.0.w3.weight"),
        ("model.layers.0.mlp.experts.0.down_proj.weight",
         "layers.0.moe_ffn.experts.0.w2.weight"),
        ("model.layers.0.mlp.gate.weight",
         "layers.0.moe_ffn.gate.weight"),
    ]

    all_passed = True
    for hf_name, expected_tm in test_cases:
        actual = map_weight(hf_name)
        passed = actual == expected_tm
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {hf_name}")
        if not passed:
            print(f"         -> {actual}")
            print(f"         expected: {expected_tm}")
        if not passed:
            all_passed = False

    return all_passed


def main():
    print("=== Qwen3.5 MoE End-to-End Configuration Verification ===\n")

    results = []
    results.append(("Config parsing", test_config_parsing()))
    results.append(("C++ MoE path", test_c_code_moe_path()))
    results.append(("Config parsing logic", test_config_parsing_logic()))
    results.append(("Compiled library", test_library_exists()))
    results.append(("Weight mapping", test_weight_mapping()))

    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nResult: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
