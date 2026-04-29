# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch research experiment comparing two optimization algorithms on CIFAR-10 with ResNet-18:
- **Baseline**: Standard Adam optimizer
- **Novel**: Weighted Backpropagation (WBP) — layer-wise gradient allocation via softmax-weighted multinomial sampling

The original specification lives in `instructions.md`. Read it entirely before writing any code.

## Running the Experiment

```bash
/Users/howard/pytorch_env/bin/python experiment.py
```

The system `python3` is 3.14 (no torch). Use the virtualenv Python above, which has PyTorch 2.8.0 + torchvision + MPS support.

Output is written to `results.md` at the end of a full run. Do not stop mid-run or ask clarifying questions — make the simplest reasonable choice and document it in the "Problems Encountered" section of results.md.

## Dependencies

- Python, PyTorch, torchvision: already installed (Homebrew). Do not upgrade.
- fvcore: check importability first; if missing: `python3 -m pip install --break-system-packages fvcore`
- **Never use bare `pip` or `pip3`** — always `python3 -m pip`
- No other third-party packages allowed (standard library + torch + torchvision + fvcore only)

## Key Implementation Rules

**Correctness**
- Fix random seed to 42 (torch, numpy, random) before anything else
- All tensors stay in float32 — MPS does not support float64; if float64 is needed, move that op to CPU
- Device priority: MPS → CUDA → CPU (using OpenMP if necessary)
- ResNet-18 from torchvision, standard architecture, no modifications
- One Adam optimizer instance per layer for WBP (preserves per-layer momentum state across outer steps)

**Code style**
- No unused imports, unused variables, shadowed names, or bare `except`
- Fix ALL PyTorch warnings (not just errors)
- Use `torch.no_grad()` wherever gradients are not needed
- Keep it simple — no unnecessary abstractions, no classes where functions suffice

## WBP Algorithm (one outer step)

1. Full forward pass → loss → full backward pass → record gradient norm per named parameter group
2. Compute softmax distribution over layers from gradient norms (temperature τ = 1.0)
3. Sample B = 2 × num_layers slots via multinomial sampling with replacement
4. Zero gradients. For each layer with ≥1 slot, in reverse order (output → input):
   - For each allocated slot:
     a. Zero gradients
     b. Partial forward from cached input activation of this layer → output → loss
     c. `.backward()` to this layer only (`retain_graph=True` if needed)
     d. Adam update for this layer's parameters only
     e. Re-run this layer's forward on cached input to refresh its output activation
   - After all slots: cache this layer's updated output activation

Cache management: before the outer step, cache input activations for every layer using forward hooks (`.detach().clone()`).

## Instrumentation (collect identically for both methods)

**Per epoch**: training loss (mean over batches), validation accuracy, cumulative wall-clock time, cumulative FLOPs

**Per layer per epoch**: mean gradient norm, allocation count (WBP: mean slots/step; Adam: constant 1.0), parameter L2 norm at epoch end

**End of training**: Hessian trace via Hutchinson estimator (10 Rademacher vectors, 256 validation samples) — implement directly, do not use pyhessian

**FLOP counting**: use fvcore; baseline = 2 × forward FLOPs per step; WBP partial passes scaled by fraction of layers traversed

## results.md Structure

Sections (as a research paper Methodology and Results section):
1. Algorithm description (describe what the code *actually* does)
2. Hyperparameters and setup (every hyperparameter, hardware, library versions)
3. Instrumentation methodology
4. Results tables (one table per metric, both methods side by side)
5. Layer-wise allocation heatmap (rows = layers, columns = epochs, values = mean slots/step for WBP)
6. Training log (epoch-by-epoch: epoch, train loss, val accuracy, wall-clock time, cumulative FLOPs)
7. Problems encountered (numbered, with resolution for each)
8. Conclusions (one short paragraph, no speculation)

Do not truncate tables. If a metric fails, log it, set value to N/A, and continue.

## Sandbox

Work only within `/Users/howard/src/WBP/` and its subdirectories.
