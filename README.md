# Weighted Backpropagation vs Adam on CIFAR-10

Research experiment comparing standard Adam against **Weighted Backpropagation (WBP)**, a novel layer-wise gradient allocation algorithm, on CIFAR-10 with ResNet-18.

## What WBP does

Standard Adam updates every layer once per step. WBP instead allocates a fixed total budget of update slots across layers, proportionally to each layer's current gradient norm:

1. Full forward + backward pass → record per-layer gradient norms
2. Compute softmax distribution over layers (temperature τ = 1.0)
3. Sample B = 2 × n_layers slots via multinomial sampling with replacement
4. For each layer with ≥ 1 slot, run a partial forward from that layer's cached input activation → loss → backward → Adam step (output-to-input order)

A simplified variant, **WBP-K=1** (`wbp_minimal.py`), replaces the stochastic allocation with a single deterministic bonus update on the highest-gradient layer on top of a normal floor Adam step. This variant outperformed both standard Adam and full WBP in the K-sweep.

Full results and analysis are in [`results.md`](results.md).

## Repository layout

```
experiment.py         Adam baseline vs full WBP (B = 104 slots)
wbp_minimal.py        WBP-K=1: floor Adam + one targeted bonus update
sweep.py              WBP-Minimal K-sweep (K = 2 … 20 bonus slots)
robustness_seed.py    Adam + WBP-K=1 on an arbitrary seed → robustness_<N>.json
instructions.md       Original experiment specification
results.md            All results (methodology, tables, robustness analysis)
sweep_plot.png        Val accuracy and sharpness vs K (from sweep.py)
robustness_*.json     Per-seed results for the robustness study
CLAUDE.md             Claude Code project instructions
```

## Requirements

### Hardware

- **Recommended:** Apple Silicon Mac (M1/M2/M3/M4) — code uses MPS acceleration
- Falls back to CUDA, then CPU
- ~340 MB disk for the CIFAR-10 dataset (downloaded automatically on first run)

### Software (exact versions used for all reported results)

| Package | Version |
|---|---|
| Python | **3.9.6** |
| PyTorch | **2.8.0** |
| torchvision | **0.23.0** |
| fvcore | **0.1.5.post20221221** |

PyTorch version matters significantly for MPS reproducibility. Results on other versions or other hardware may differ.

## Installation

Create a fresh virtual environment with Python 3.9 and install the exact package versions:

```bash
python3.9 -m venv wbp_env
source wbp_env/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0
python -m pip install fvcore==0.1.5.post20221221
```

> **Apple Silicon note:** PyTorch 2.8.0 ships with MPS support built in. No additional steps are needed. Verify with:
> ```bash
> python -c "import torch; print(torch.backends.mps.is_available())"
> # should print: True
> ```

All scripts use `python` from whichever environment is active. Substitute the full path to your venv's Python if needed (e.g. `wbp_env/bin/python experiment.py`).

## Reproducing the results

Run all scripts from the repo root. CIFAR-10 downloads automatically into `data/` on first run.

### Step 1 — Main experiment: Adam vs full WBP

```bash
python experiment.py
```

Trains ResNet-18 on CIFAR-10 for 30 epochs with two optimizers: standard Adam (baseline) and full WBP (B = 104 stochastic slots). Writes `results.md`.

**Seed:** 42  
**Runtime:** ~20 min (Adam) + ~22 min (WBP) on Apple M-series ≈ **42 min total**

### Step 2 — WBP-K=1 (WBP-Minimal)

```bash
python wbp_minimal.py
```

Trains with WBP-K=1: a floor Adam step over all parameters plus one bonus Adam step on the layer with the largest gradient norm. Reads the Adam training log from `results.md` for the side-by-side comparison table, then appends results.

**Seed:** 42  
**Runtime:** ~22 min

### Step 3 — K-sweep (WBP-Minimal with K = 2 … 20 bonus slots)

```bash
python sweep.py
```

Runs WBP-Minimal for K ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20} bonus slots. Appends a summary table and saves `sweep_plot.png`.

**Seed:** 42 (reset before each K)  
**Runtime:** ~22 min × 14 values ≈ **5 hours**

### Step 4 — Robustness study (multiple seeds)

Run Adam + WBP-K=1 on any seed; outputs `robustness_<SEED>.json`:

```bash
python robustness_seed.py --seed 0
python robustness_seed.py --seed 7
python robustness_seed.py --seed 123
# … etc.
```

**Seeds used in the paper:** 0, 1, 2, 3, 4, 5, 6, 7, 123 (plus the seed-42 original)  
**Runtime:** ~40 min per seed

To run all robustness seeds sequentially (as done for the reported results):

```bash
for SEED in 0 7 123 1 2 3 4 5 6; do
    python robustness_seed.py --seed $SEED
done
```

## Hyperparameters

All hyperparameters are hardcoded in each script. The values are identical across all scripts unless noted.

### Shared (all scripts)

| Parameter | Value |
|---|---|
| Random seed | 42 (main runs); varies (robustness) |
| Batch size | 128 |
| Epochs | 30 |
| Learning rate | 1e-3 |
| Adam β₁, β₂ | 0.9, 0.999 |
| Adam ε | 1e-8 |
| Dataset | CIFAR-10 (50 000 train / 10 000 val) |
| Model | ResNet-18 (torchvision, `weights=None`, 10-class head) |
| Train augmentation | RandomHorizontalFlip, RandomCrop(32, padding=4) |
| Val augmentation | None |
| Normalization mean | (0.4914, 0.4822, 0.4465) |
| Normalization std | (0.2023, 0.1994, 0.2010) |
| DataLoader workers | 2 |
| Device priority | MPS → CUDA → CPU |

### Full WBP (`experiment.py`)

| Parameter | Value |
|---|---|
| Temperature τ | 1.0 |
| Slot budget B | 2 × 52 = 104 per outer step |
| Leaf modules n | 52 (all `named_modules()` with no children) |
| Per-layer optimizer | One Adam instance per leaf module (momentum persists across steps) |

### WBP-K=1 (`wbp_minimal.py`, `robustness_seed.py`)

| Parameter | Value |
|---|---|
| Floor update | One Adam step over all parameters (identical to baseline) |
| Bonus update | One Adam step on the leaf module with the largest mean gradient norm |
| Per-layer bonus optimizer | One Adam instance per leaf module (momentum persists across steps) |

### Sharpness estimator (all scripts)

| Parameter | Value |
|---|---|
| Method | Hutchinson trace estimator (Rademacher vectors) |
| Rademacher vectors | 10 |
| Validation samples | 256 |
| MaxPool2d handling | Temporarily replaced with AvgPool2d (same kernel/stride/padding) for double-backward compatibility |

### FLOP accounting (`experiment.py`)

| Operation | FLOPs counted |
|---|---|
| Adam baseline step | 2 × forward FLOPs (forward + backward approximation) |
| WBP outer step | 2 × forward FLOPs (full pass) + per slot: 2 × (n_layers − i) / n_layers × forward FLOPs |
| Forward FLOPs | Measured via `fvcore.nn.FlopCountAnalysis` on a single (1, 3, 32, 32) input |

## Random seeds

| Result | Script | Seed(s) |
|---|---|---|
| Main experiment (Adam vs full WBP) | `experiment.py` | 42 |
| WBP-K=1 main run | `wbp_minimal.py` | 42 |
| K-sweep (K = 2 … 20) | `sweep.py` | 42 (reset per K) |
| Robustness batch 1 | `robustness_seed.py` | 0, 7, 123 |
| Robustness batch 2 | `robustness_seed.py` | 1, 2, 3, 4, 5, 6 |

Seeds are applied to `torch`, `numpy.random`, and `random` at the start of each run. MPS does not guarantee bit-exact reproducibility across PyTorch versions or OS versions even with a fixed seed.

## Key results summary

Full tables are in `results.md`. Brief summary across 10 seeds (42, 0, 7, 123, 1–6):

| Metric | Adam | WBP-K=1 | WBP wins |
|---|---|---|---|
| Val accuracy (mean ± std) | 0.8304 ± 0.0033 | 0.8332 ± 0.0049 | 7 / 10 seeds |
| Sharpness / Hessian trace (mean ± std) | 978 ± 157 | 615 ± 72 | 10 / 10 seeds |
| Bayesian posterior z (val acc, N=10) | — | **1.73σ** | — |
| Bayesian posterior z (sharpness, N=10) | — | **4.00σ** | — |

The sharpness advantage (WBP consistently finds flatter minima) is the more robust finding.
The val accuracy advantage is small (+0.28 pp mean) and high-variance; reaching 4σ confidence
on that metric is projected to require ~47 total seeds at the observed SNR.

## Notes on reproducibility

- Results were produced on an Apple Silicon Mac with MPS. CPU or CUDA runs will produce different numbers due to floating-point non-associativity in parallel reductions.
- PyTorch MPS does not guarantee bit-exact reproducibility across OS versions even with a fixed seed.
- The CIFAR-10 `data/` directory is excluded from the repository. It is created automatically by torchvision on first run.
- All per-seed numerical results are archived in `robustness_<SEED>.json` files tracked in this repository.
