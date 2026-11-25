#!/usr/bin/env bash
# One launcher for any world size:
#
#   ./scripts/launch.sh runs/130m_muon.toml --wandb optbench --wandb-run-name 130m-muon-1x
#
# Picks up every visible GPU via torchrun; falls back to a plain single-process
# run on 1 GPU or CPU (detect_dist makes that the degenerate case of N).
# Relaunching the same config resumes from the latest checkpoint.
set -euo pipefail

CONFIG=${1:?usage: launch.sh <config.toml> [train.py args...]}
shift

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "${NGPU}" -gt 1 ]; then
    exec uv run torchrun --standalone --nproc_per_node="${NGPU}" \
        -m optbench.train --config "${CONFIG}" "$@"
else
    exec uv run python -m optbench.train --config "${CONFIG}" "$@"
fi
