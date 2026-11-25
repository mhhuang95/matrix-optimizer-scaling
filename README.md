# optbench — do matrix optimizers keep their edge at scale?

A fairly-tuned comparison of **Muon, Distributed Shampoo, and AdamW** on
Llama-style language models (130M/300M, FineWeb-Edu, 1–8× Chinchilla), built on
a from-scratch training stack. The question the study adds over prior
benchmarks (Wen et al., ICLR 2026): can **per-layer spectral statistics** of
the gradients — effective rank, nuclear/Frobenius ratio, update spectra —
explain and predict *where* the matrix-optimizer speedup shrinks?

## The stack

Everything is built here, not forked: Llama-style model (RoPE, RMSNorm,
SwiGLU), deterministic resumable data loader, bf16 train step with grad
accumulation, Muon (Newton–Schulz orthogonalization) implemented from scratch
and tested against a reference, Meta's Distributed Shampoo vendored with
torch-2.2 compat shims, cosine/WSD schedules, atomic bit-exact-resume
checkpoints, DDP with one-writer/global-mean-logging discipline, MFU metering,
per-layer spectral diagnostics, TOML configs, JSONL + W&B logging.

```
src/optbench/     model, data, optimizers, loop, ddp, train (CLI), ...
tests/            423 tests; `uv run pytest`
runs/             one TOML per run
scripts/          Lambda setup + launcher
third_party/      vendored Meta Distributed Shampoo (see VENDORED.txt)
```

## Reproduce

On a Lambda H100 instance (Lambda Stack):

```bash
git clone https://github.com/mhhuang95/matrix-optimizer-scaling.git optbench
cd optbench
./scripts/lambda_setup.sh        # uv sync, tests, wandb login, tokenize FineWeb-Edu
./scripts/launch.sh runs/smoke.toml                      # ~2 min end-to-end check
./scripts/launch.sh runs/130m_muon.toml --wandb optbench # the real thing
```

`launch.sh` uses every visible GPU via torchrun; the same TOML runs unchanged
on CPU, 1 GPU, or 8 (world size is a pure throughput knob — tokens/step, data
order, and the loss curve are identical by construction). Relaunching a config
resumes from its latest checkpoint.

Local development (macOS works, CPU-only):

```bash
uv sync && uv run pytest
```
