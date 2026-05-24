#!/usr/bin/env python3
"""Simple test to verify Python Bridge works."""
import subprocess
import json
import sys
import time
from pathlib import Path

# Bridge script
BRIDGE_SCRIPT = Path("/mnt/data/lmdeploy/lmdeploy/turbomind/python_bridge.py")
MODEL_PATH = "/mnt/eaget-4tb/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

def test_bridge():
    print("Testing Python Bridge...")

    # Start bridge
    proc = subprocess.Popen(
        ["python3", str(BRIDGE_SCRIPT), "--model-path", MODEL_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for ready
    print("Waiting for model load...")
    ready_line = proc.stdout.readline()
    print(f"Got: {ready_line.strip()}")

    ready = json.loads(ready_line)
    if ready.get("status") != "ok":
        print(f"ERROR: Bridge failed to load: {ready}")
        proc.kill()
        return False

    print("Model loaded! Testing ping...")
    proc.stdin.write(json.dumps({"cmd": "ping"}) + "\n")
    proc.stdin.flush()

    pong = proc.stdout.readline()
    print(f"Ping response: {pong.strip()}")

    # Test metrics
    print("\nTesting metrics...")
    proc.stdin.write(json.dumps({"cmd": "metrics"}) + "\n")
    proc.stdin.flush()

    metrics = json.loads(proc.stdout.readline())
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Shutdown
    print("\nShutting down...")
    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=10)

    print("Bridge test PASSED!")
    return True

if __name__ == "__main__":
    try:
        success = test_bridge()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
