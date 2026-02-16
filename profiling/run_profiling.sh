#!/bin/bash
# Run all CVXRO profiling scripts and consolidate output.
#
# Usage:
#   bash profiling/run_profiling.sh           # Run all
#   bash profiling/run_profiling.sh hotspots  # Run specific script

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_FILE="$RESULTS_DIR/profiling_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RESULTS_DIR"

cd "$PROJECT_DIR"

run_script() {
    local name="$1"
    local script="$2"
    echo ""
    echo "=================================================================="
    echo "  Running: $name"
    echo "  Script:  $script"
    echo "  Time:    $(date)"
    echo "=================================================================="
    echo ""
    uv run python "$script"
}

# If a specific script is requested
if [ -n "$1" ]; then
    case "$1" in
        canon*)
            run_script "Canonicalization Profile" "$SCRIPT_DIR/profile_canonicalization.py" 2>&1 | tee -a "$LOG_FILE"
            ;;
        memory|mem*)
            run_script "Memory Profile" "$SCRIPT_DIR/profile_memory.py" 2>&1 | tee -a "$LOG_FILE"
            ;;
        scal*)
            run_script "Scaling Analysis" "$SCRIPT_DIR/profile_scaling.py" 2>&1 | tee -a "$LOG_FILE"
            ;;
        hot*)
            run_script "Hotspot Micro-benchmarks" "$SCRIPT_DIR/profile_hotspots.py" 2>&1 | tee -a "$LOG_FILE"
            ;;
        *)
            echo "Unknown script: $1"
            echo "Options: canonicalization, memory, scaling, hotspots"
            exit 1
            ;;
    esac
    echo ""
    echo "Log saved to: $LOG_FILE"
    exit 0
fi

# Run all scripts: hotspots first (quickest, most targeted), then scaling,
# memory, and finally cProfile (most detailed, slowest).
echo "CVXRO Profiling Suite"
echo "Log: $LOG_FILE"
echo ""

{
    run_script "Hotspot Micro-benchmarks" "$SCRIPT_DIR/profile_hotspots.py"
    run_script "Scaling Analysis" "$SCRIPT_DIR/profile_scaling.py"
    run_script "Memory Profile" "$SCRIPT_DIR/profile_memory.py"
    run_script "Canonicalization Profile" "$SCRIPT_DIR/profile_canonicalization.py"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================================="
echo "  All profiling complete"
echo "  Log: $LOG_FILE"
echo "  Results: $RESULTS_DIR/"
echo "=================================================================="
