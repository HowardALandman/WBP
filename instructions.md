These are your complete instructions. Read the entire file before writing any code.

You are implementing a neural network training experiment comparing standard Adam optimization against a novel algorithm called Weighted Backpropagation (WBP) on CIFAR-10.

Task Overview

Train ResNet-18 on CIFAR-10 with two optimizers: Adam (baseline) and WBP (novel). Collect identical instrumentation for both. Write results to a Markdown file formatted as a Methodology and Results section of a research paper. Log every problem encountered and how it was fixed.

Algorithm Description

Baseline: Standard Adam
Standard PyTorch Adam optimizer, one gradient update per parameter per step, learning rate 1e-3, default betas. No modifications.
Novel Algorithm: Weighted Backpropagation (WBP)
WBP allocates a fixed total budget of gradient update slots across layers, proportionally to each layer's current error contribution, instead of updating every layer exactly once. The atomic unit of allocation is one named parameter group (layer) as returned by model.named_parameters(). This is a deliberate simplifying choice — no finer-grained allocation within a layer is attempted.
Precise algorithm for one outer training step:

Zero all gradients. Do a full forward pass. Compute the loss. Do a full backward pass (.backward()). Compute and record the gradient norm of each named parameter group. These norms are the allocation weights used in step 2. Do not zero gradients yet.
Compute a probability distribution over layers from their gradient norms using softmax with temperature τ = 1.0: p_k = softmax(grad_norm_k / τ).
Sample a total budget of B = 2 × num_layers update slots from this distribution (multinomial sampling with replacement). This gives each layer an integer count of how many gradient steps it will receive this outer step. Some layers may receive zero; some may receive several.
Zero all gradients. Then, for each layer that received at least one slot, in order from output layer to input layer (reverse layer order):
a. For each allocated slot for this layer:

Zero all gradients.
Run a partial forward pass: starting from the current layer's cached input activation (from the full forward pass in step 1), run the network forward from this layer to the output. Compute the loss against the stored targets.
Run .backward() from this loss back to this layer only (do not propagate further back). Use retain_graph=True if needed.
Apply a single Adam update step to this layer's parameters only.
Re-run just this layer's forward on the cached input to refresh this layer's output activation, so subsequent slots see updated weights.
b. After all slots for this layer are processed, recompute and cache the output activation of this layer for use by shallower layers in subsequent iterations.


Cache management: before the outer step begins, cache the input activation to every layer by running a forward pass with hooks. These cached inputs are the "frozen upstream" activations used during inner updates. All cached activations should be detached from the computation graph (.detach().clone()).

Key parameters:

Temperature τ = 1.0 (fixed)
Budget B = 2 × num_layers per outer step
Optimizer for each per-layer step: Adam, lr = 1e-3, same hyperparameters as baseline
One Adam optimizer instance per layer, so momentum states are maintained per-layer across outer steps

Instrumentation

Collect ALL of the following identically for both Adam and WBP runs:
Per epoch:

Training loss (mean over batches)
Validation accuracy
Wall-clock time elapsed (seconds, cumulative from training start)
Total FLOPs consumed (cumulative; estimate: one forward+backward pass = 2 × forward FLOPs; for WBP count each partial forward pass proportionally by the fraction of layers it traverses)

Per layer, per epoch (for every named layer in the model):

Mean gradient norm across all steps in the epoch
Allocation count (for WBP: mean slots assigned per step; for Adam: constant 1.0)
Parameter norm (L2 norm of weights at end of epoch)

At end of training only:

Loss surface sharpness: approximate trace of the Hessian using the Hutchinson estimator (10 random Rademacher vectors, on 256 samples from the validation set). Implement the Hutchinson estimator directly; do not depend on pyhessian.

Compute tracking:

For WBP: record the actual number of partial forward passes executed per epoch
For both: record total wall-clock time per epoch


Software Requirements

Python, PyTorch, torchvision must already be installed; do not attempt to install or upgrade them. The base Python environment is in Homebrew, so use "brew install" whenever possible.
fvcore is needed for FLOP counting. Check if it is importable (via brew) first. If not, install it with "python3 -m pip install --break-system-packages fvcore". Never use bare "pip" or "pip3" anywhere in this script or in any shell commands. Always use "python3 -m pip" instead. Treat bare pip as deprecated and forbidden.
No other third-party packages beyond torch, torchvision, fvcore, and the Python standard library are permitted. Implement the Hutchinson estimator for Hessian trace directly.
ResNet-18 from torchvision, standard architecture, no modifications.
CIFAR-10: download if not present, standard train/val split (50k train, 10k val), standard normalization (mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010]).
Data augmentation for training only: random horizontal flip, random crop 32×32 with padding 4. No augmentation for validation.
Batch size: 128. Train for 30 epochs each. Use MPS (Apple Silicon Metal library) if available, then CUDA if an Nvidia GPU is present, then CPU (with OpenMP if paralellism is needed).
Random seed: fix to 42 for all sources of randomness (torch, numpy, random) at the start of the script before anything else.
Write squeaky clean code. No unused imports. No unused variables. No shadowed names. No bare except. Fix ALL warnings from PyTorch, not just errors. Use torch.no_grad() wherever gradients are not needed.
Keep it as simple as possible. Do not add abstractions that aren't needed. Do not use classes where functions suffice. Do not change things that don't need to be changed.
Handle MPS (Apple Silicon) backend correctly: MPS does not support float64; keep all tensors in float32. If any operation requires float64, perform it on CPU.


Output

Write a single Markdown file results.md at the end of the run. Structure it as the Methodology and Results section of a research paper. Include:

Algorithm description — formal description of both methods as implemented (not as intended; describe what the code actually does).
Hyperparameters and setup — every hyperparameter, hardware detected, library versions.
Instrumentation methodology — exactly how each metric was computed.
Results tables — one table per metric with values for both methods side by side.
Layer-wise allocation heatmap — Markdown table showing per-layer mean allocation count for WBP across epochs (rows = layers, columns = epochs, values = mean slots assigned per step).
Training log — epoch-by-epoch table for both methods: epoch, train loss, val accuracy, wall-clock time, cumulative FLOPs.
Problems encountered — a numbered list of every error, warning, or unexpected behavior encountered during implementation and execution, and exactly how each was resolved.
Conclusions — one short paragraph summarizing what the numbers show, without interpretation or speculation.

Do not truncate any tables. Do not omit any metrics. If a metric fails to compute, log the error, set the value to N/A, and continue; do not abort.

Execution

Run the complete experiment end to end. Do not stop to ask questions. If something is ambiguous, make the simplest reasonable choice and document it in the Problems Encountered section. When the script finishes, confirm the path to results.md.

Safety and Privacy

This directory (/Users/howard/src/WBP) is your sandbox. Stay inside it.
You may freely create subdirectories, but do not cd to any directory other than this one and its subdirectries.
Do not write or modify any files outside your sandbox (except as needed to install required packages and libraries).
Do not read any files on this computer outside your sandbox that are not directly required by your instructions. Reading and downloading files, libraries, packages, repositories etc. from the web as needed for this task is OK.
