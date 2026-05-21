#!/usr/bin/env python3
import subprocess
import sys
import time
import os

# Set up environment
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:" + \
                         os.getcwd() + "/build/lib:" + \
                         "/usr/local/cuda-12.5/lib64:/usr/lib/x86_64-linux-gnu"

# Start diagnostic in background
proc = subprocess.Popen(
    ["./lmdeploy-rust-server/target/release/diagnostic"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait 20 seconds then interrupt
time.sleep(20)

print(f"--- Sending SIGINT to PID {proc.pid} ---", file=sys.stderr)
proc.send_signal(subprocess.signal.SIGINT)

# Read output
try:
    output = proc.communicate(timeout=5)[0]
    print(output)
except subprocess.TimeoutExpired:
    print(f"Timeout, sending SIGKILL to PID {proc.pid}", file=sys.stderr)
    proc.kill()
    output = proc.communicate()[0]
    print(output)

# Get stack trace with gdb
if proc.returncode in (-2, 143):  # SIGINT or SIGTERM
    print("\n--- Running gdb to get stack trace ---", file=sys.stderr)
    gdb_cmd = [
        "gdb", "./lmdeploy-rust-server/target/release/diagnostic", "-ex", "run",
        "-ex", "sleep 15", "-ex", "interrupt", "-ex", "thread apply all bt", "-ex", "quit"
    ]
    try:
        gdb_out = subprocess.run(gdb_cmd, env=env, capture_output=True, text=True, timeout=30)
        print(gdb_out.stdout)
        if gdb_out.stderr:
            print("\n--- GDB stderr ---", file=sys.stderr)
            print(gdb_out.stderr)
    except subprocess.TimeoutExpired:
        print("GDB timed out", file=sys.stderr)