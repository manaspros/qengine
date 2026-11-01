#!/usr/bin/env bash
# Run a command while holding the exclusive qengine GPU lock.
# Every GPU job (tests, benchmarks, baseline engines) goes through this so that
# two jobs never share the 5090 (VRAM: two 27B models do not fit; timings get disturbed).
# Usage: scripts/gpu_lock.sh <command> [args...]
exec flock "$HOME/.cache/qengine/gpu.lock" "$@"
