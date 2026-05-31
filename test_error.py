
#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile

os.environ['LD_LIBRARY_PATH'] = f'{os.getcwd()}/build/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

print("Running benchmark with stderr capture...")
print("="*80)

cmd = [sys.executable, "benchmark_dflash.py"]
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print("="*80)
print(result.stdout)
print("\nSTDERR:")
print("="*80)
print(result.stderr)
print("\nReturn code:", result.returncode)
