#!/usr/bin/env bash
# Fresh Lambda instance -> ready to train. Run once, from the repo root:
#
#   git clone https://github.com/mhhuang95/matrix-optimizer-scaling.git optbench
#   cd optbench && ./scripts/lambda_setup.sh
#
# Lambda Stack preinstalls CUDA drivers; everything Python lives in the uv venv.
set -euo pipefail

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --extra wandb --extra data

# Fail fast if the venv's torch can't see the GPUs.
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.device_count(), 'GPU(s):', torch.cuda.get_device_name(0))"

uv run pytest -q

# W&B credentials: skipped when WANDB_API_KEY is already exported.
if [ -z "${WANDB_API_KEY:-}" ]; then
    uv run wandb login
fi

# Tokenize FineWeb-Edu once (~3B tokens covers the 1x-Chinchilla 130M runs;
# the 8x overtraining runs need ~21B — rerun into a fresh dir with
# --subset sample-100BT --max-tokens 21e9).
if [ ! -d data/fineweb_edu ]; then
    uv run python -m optbench.tokenize_data --out data/fineweb_edu --max-tokens 3e9
fi

echo "setup complete — now: ./scripts/launch.sh runs/smoke.toml"
