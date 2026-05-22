#!/bin/bash
# Automated InitFromPath Hang Diagnosis
# Attaches to a running process and analyzes all threads for hang patterns

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <PID> [output_file]"
    echo ""
    echo "Example:"
    echo "  $0 12345"
    echo ""
    echo "To find the PID of the hanging process:"
    echo "  ps aux | grep -E 'lmdeploy|diagnostic|file_diag'"
    exit 1
fi

PID=$1
OUTPUT=${2:-gdb_diagnosis_$(date +%Y%m%d_%H%M%S).txt}

echo "=== InitFromPath Automated Hang Diagnosis ==="
echo "PID: $PID"
echo "Output: $OUTPUT"
echo ""

# Step 1: Verify process is running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: Process $PID is not running"
    exit 1
fi

echo "✓ Process $PID is running"

# Step 2: Get basic process info
echo ""
echo "=== Process Info ==="
ps -p "$PID" -o pid,vsz,rss,stat,etime,command | head -5
echo ""

# Step 3: Run gdb analysis
echo "=== GDB Analysis ==="

GDB_SCRIPT=$(mktemp)
cat > "$GDB_SCRIPT" << 'GDBEOF'
# === Thread Analysis ===
info threads

# === Per-thread backtrace ===
thread apply all bt 5

# === Look for threads in futex/wait state ===
printf "\n=== THREAD STATE ANALYSIS ===\n"
set $total_threads = 0
set $in_futex = 0
set $in_cuda = 0
set $in_init = 0
set $in_queue = 0

# Count threads in different states
thread apply all
GDBEOF

echo "Running gdb analysis..."
echo ""

# Run gdb and capture output
gdb -batch -p "$PID" -x "$GDB_SCRIPT" > "$OUTPUT" 2>&1 || {
    echo "⚠ gdb failed (permissions issue). Try with sudo:"
    echo "  sudo $0 $PID"
    echo ""
    echo "Alternative: Use strace for syscall analysis:"
    echo "  strace -p $PID -f -e trace=futex,sched_yield,nanosleep -c"
    rm -f "$GDB_SCRIPT" "$OUTPUT"
    exit 1
}

rm -f "$GDB_SCRIPT"

# Step 4: Analyze the output
echo ""
echo "=== Analysis Results ==="

# Check for futex waits (normal blocking)
FUTEX_COUNT=$(grep -c "__lll_lock_wait\|futex\|pthread_cond_wait" "$OUTPUT" 2>/dev/null || echo "0")
echo "Threads in futex/cond_wait: $FUTEX_COUNT"

# Check for threads stuck in InitFromPath
INIT_COUNT=$(grep -c "InitFromPath\|ProcessWeights\|CreateEngine" "$OUTPUT" 2>/dev/null || echo "0")
echo "Threads in InitFromPath path: $INIT_COUNT"

# Check for CUDA operations
CUDA_COUNT=$(grep -c "cuda\|CUstream\|cuLaunch" "$OUTPUT" 2>/dev/null || echo "0")
echo "Threads in CUDA operations: $CUDA_COUNT"

# Check for threads stuck in gateway pop
QUEUE_COUNT=$(grep -c "gateway_.*pop\|RequestQueue::pop" "$OUTPUT" 2>/dev/null || echo "0")
echo "Threads in RequestQueue::pop: $QUEUE_COUNT"

# Check for InternalThreadEntry
ENGINE_COUNT=$(grep -c "InternalThreadEntry" "$OUTPUT" 2>/dev/null || echo "0")
echo "Threads in InternalThreadEntry: $ENGINE_COUNT"

echo ""
echo "=== Interpretation ==="

if [ "$ENGINE_COUNT" -gt 0 ] && [ "$QUEUE_COUNT" -gt 0 ]; then
    echo "✓ Engine threads are running and waiting for requests"
    echo "  This is NORMAL - the engine is idle, waiting for inference requests."
    echo "  The hang is likely elsewhere (not in CreateEngine)."
fi

if [ "$INIT_COUNT" -gt 0 ]; then
    echo "⚠ Thread(s) stuck in InitFromPath/ProcessWeights/CreateEngine"
    echo "  This indicates the initialization did not complete."
    echo "  Check the backtrace for the exact function."
fi

if [ "$CUDA_COUNT" -gt "$ENGINE_COUNT" ] 2>/dev/null; then
    echo "⚠ Thread(s) stuck in CUDA operations"
    echo "  Possible CUDA hang or GPU resource issue."
fi

if [ "$FUTEX_COUNT" -gt "$((ENGINE_COUNT + QUEUE_COUNT + 5))" ] 2>/dev/null; then
    echo "⚠ Many threads in futex wait - possible deadlock"
    echo "  Check if threads are waiting on different locks."
fi

echo ""
echo "=== Full Backtrace ==="
echo "See $OUTPUT for complete thread backtraces."
echo ""
echo "=== Suggested Next Steps ==="
echo "1. Check if the hang is in ProcessWeights:"
echo "   gdb -batch -p $PID -ex 'thread apply all bt' | grep -A5 -B5 ProcessWeights"
echo ""
echo "2. Check if the hang is in CreateEngine:"
echo "   gdb -batch -p $PID -ex 'thread apply all bt' | grep -A5 -B5 CreateEngine"
echo ""
echo "3. Run with strace for syscall analysis:"
echo "   strace -p $PID -f -tt -o strace.log"
echo ""
echo "4. Check CUDA operations with nsight:"
echo "   nsys profile --stats=true ./your_binary"
echo ""
echo "=== End of Diagnosis ==="
