#!/usr/bin/env python3
"""Experiment: Adam vs Weighted Backpropagation on CIFAR-10 with ResNet-18."""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fvcore.nn import FlopCountAnalysis

# ── Seed (must be first) ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")
PIN = DEVICE.type == "cuda"

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR                 = 1e-3
BATCH              = 128
EPOCHS             = 30
TAU                = 1.0
N_RADEMACHER       = 10
HUTCHINSON_SAMPLES = 256

# ── Data ──────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
val_ds   = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=val_tf)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=PIN)
val_loader = torch.utils.data.DataLoader(
    val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=PIN)

# ── Model helpers ─────────────────────────────────────────────────────────────
def make_model():
    return torchvision.models.resnet18(weights=None, num_classes=10).to(DEVICE)

def get_leaf_modules(model):
    """Ordered list of (name, module) for all leaf modules (no children)."""
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]

# ── FLOP counting ─────────────────────────────────────────────────────────────
def count_forward_flops(model):
    # fvcore JIT tracing requires CPU; use eval mode to avoid BN single-sample error.
    cpu_model = model.cpu().eval()
    dummy = torch.zeros(1, 3, 32, 32)
    fa = FlopCountAnalysis(cpu_model, dummy)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    total = fa.total()
    model.to(DEVICE).train()
    return total

# ── Validation ────────────────────────────────────────────────────────────────
def validate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            preds = model(inputs).argmax(1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
    model.train()
    return correct / total

# ── Hutchinson trace estimator ────────────────────────────────────────────────
def hutchinson_trace(model):
    """Approximate Hessian trace via Hutchinson estimator (Rademacher vectors).

    MaxPool2d is not twice-differentiable, so we temporarily replace it with
    AvgPool2d of identical spatial hyperparameters for this computation.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Swap MaxPool2d → AvgPool2d so double-backward succeeds.
    swapped = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.MaxPool2d):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            avg = nn.AvgPool2d(
                kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding
            ).to(DEVICE)
            setattr(parent, parts[-1], avg)
            swapped.append((parent, parts[-1], mod))

    inputs_list, targets_list, count = [], [], 0
    for inp, tgt in val_loader:
        inputs_list.append(inp)
        targets_list.append(tgt)
        count += inp.size(0)
        if count >= HUTCHINSON_SAMPLES:
            break
    inputs  = torch.cat(inputs_list)[:HUTCHINSON_SAMPLES].to(DEVICE)
    targets = torch.cat(targets_list)[:HUTCHINSON_SAMPLES].to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    traces = []
    err_msg = None

    try:
        for _ in range(N_RADEMACHER):
            vs = [torch.randint(0, 2, p.shape, device=DEVICE, dtype=torch.float32) * 2 - 1
                  for p in params]
            out   = model(inputs)
            loss  = criterion(out, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv    = sum((g * v).sum() for g, v in zip(grads, vs))
            Hvs   = torch.autograd.grad(gv, params, retain_graph=False)
            traces.append(sum((v * hv).sum().item() for v, hv in zip(vs, Hvs)))
    except Exception as exc:
        err_msg = str(exc)
    finally:
        for parent, attr, orig in swapped:
            setattr(parent, attr, orig)
        model.train()

    if err_msg:
        return None, err_msg
    return float(np.mean(traces)), None

# ── Bypass hook ───────────────────────────────────────────────────────────────
def make_bypass_hook(cached_out):
    """Forward hook that replaces module output with a cached detached tensor."""
    def hook(module, inp, out):
        return cached_out
    return hook

# ── Adam baseline ─────────────────────────────────────────────────────────────
def run_adam():
    problems = []
    model     = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    forward_flops = count_forward_flops(model)
    step_flops    = 2 * forward_flops  # forward + backward

    leaf_modules = get_leaf_modules(model)
    n_layers     = len(leaf_modules)

    epoch_logs  = []
    layer_logs  = {n: {"grad_norms": [], "alloc": [], "param_norms": []}
                   for n, _ in leaf_modules}

    train_start      = time.time()
    cumulative_flops = 0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss     = 0.0
        step_gnorms    = {n: [] for n, _ in leaf_modules}

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()

            for name, mod in leaf_modules:
                norms = [p.grad.norm().item() for p in mod.parameters()
                         if p.grad is not None]
                step_gnorms[name].append(np.mean(norms) if norms else 0.0)

            optimizer.step()
            total_loss       += loss.item()
            cumulative_flops += step_flops

        val_acc   = validate(model)
        elapsed   = time.time() - train_start
        mean_loss = total_loss / len(train_loader)

        for name, mod in leaf_modules:
            layer_logs[name]["grad_norms"].append(np.mean(step_gnorms[name]))
            layer_logs[name]["alloc"].append(1.0)
            pnorms = [p.data.norm().item() for p in mod.parameters()]
            layer_logs[name]["param_norms"].append(np.mean(pnorms) if pnorms else 0.0)

        epoch_logs.append({
            "epoch": epoch, "train_loss": mean_loss,
            "val_acc": val_acc, "time": elapsed, "flops": cumulative_flops,
        })
        print(f"[Adam] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}  t={elapsed:.1f}s")

    sharpness, err = hutchinson_trace(model)
    if err:
        problems.append(f"Adam Hutchinson estimator failed: {err}")

    return epoch_logs, layer_logs, sharpness, problems, n_layers

# ── WBP ───────────────────────────────────────────────────────────────────────
def wbp_outer_step(model, leaf_modules, optimizers, criterion, inputs, targets, B):
    """
    One WBP outer step.
    Returns (step_loss, grad_norms_per_layer, slot_counts_per_layer).
    """
    n = len(leaf_modules)

    # Cache inputs/outputs of every leaf module during the full forward pass.
    cache_inputs  = {}
    cache_outputs = {}
    handles       = []

    def make_cache_hook(nm):
        def hook(mod, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                cache_inputs[nm]  = inp[0].detach().clone()
            cache_outputs[nm] = out.detach().clone()
        return hook

    for name, mod in leaf_modules:
        handles.append(mod.register_forward_hook(make_cache_hook(name)))

    # Step 1: full forward + backward
    model.zero_grad()
    out       = model(inputs)
    loss      = criterion(out, targets)
    step_loss = loss.item()
    loss.backward()

    for h in handles:
        h.remove()

    # Gradient norms per layer
    grad_norms = []
    for _, mod in leaf_modules:
        norms = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        grad_norms.append(np.mean(norms) if norms else 0.0)

    # Step 2: softmax distribution
    gn    = torch.tensor(grad_norms, dtype=torch.float32)
    probs = torch.softmax(gn / TAU, dim=0)

    # Step 3: sample slots
    slot_idx    = torch.multinomial(probs, num_samples=B, replacement=True)
    slot_counts = torch.bincount(slot_idx, minlength=n).tolist()

    # Step 4: process layers in reverse order (output → input)
    model.zero_grad()

    for i in reversed(range(n)):
        if slot_counts[i] == 0:
            continue

        name_i, mod_i = leaf_modules[i]
        opt_i         = optimizers[i]

        for _ in range(slot_counts[i]):
            model.zero_grad()

            # Bypass modules 0..i-1: replace their outputs with cached detached tensors.
            bypass = []
            for j in range(i):
                co = cache_outputs[leaf_modules[j][0]]
                bypass.append(leaf_modules[j][1].register_forward_hook(make_bypass_hook(co)))

            out  = model(inputs)
            loss = criterion(out, targets)
            loss.backward()

            for h in bypass:
                h.remove()

            if opt_i is not None:
                opt_i.step()

            # Refresh module i's cached output with its updated weights.
            ci = cache_inputs.get(name_i)
            if ci is not None and opt_i is not None:
                with torch.no_grad():
                    cache_outputs[name_i] = mod_i(ci).detach().clone()

    return step_loss, grad_norms, slot_counts


def run_wbp():
    problems = []
    model     = make_model()
    criterion = nn.CrossEntropyLoss()

    forward_flops = count_forward_flops(model)
    leaf_modules  = get_leaf_modules(model)
    n_layers      = len(leaf_modules)
    B             = 2 * n_layers

    # One Adam optimizer per leaf module (None for parameter-free modules).
    optimizers = []
    for _, mod in leaf_modules:
        params = list(mod.parameters())
        optimizers.append(torch.optim.Adam(params, lr=LR) if params else None)

    epoch_logs  = []
    layer_logs  = {n: {"grad_norms": [], "alloc": [], "param_norms": []}
                   for n, _ in leaf_modules}

    train_start      = time.time()
    cumulative_flops = 0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss      = 0.0
        epoch_gnorms    = {n: [] for n, _ in leaf_modules}
        epoch_slots     = {n: [] for n, _ in leaf_modules}
        epoch_partial   = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            step_loss, grad_norms, slot_counts = wbp_outer_step(
                model, leaf_modules, optimizers, criterion, inputs, targets, B
            )
            total_loss += step_loss

            # FLOPs: outer full forward+backward + per-slot partial forward+backward.
            cumulative_flops += 2 * forward_flops
            for i, cnt in enumerate(slot_counts):
                if cnt > 0:
                    fraction = (n_layers - i) / n_layers
                    cumulative_flops += int(cnt * fraction * forward_flops * 2)
                    epoch_partial    += cnt

            for idx, (name, _) in enumerate(leaf_modules):
                epoch_gnorms[name].append(grad_norms[idx])
                epoch_slots[name].append(slot_counts[idx])

        val_acc   = validate(model)
        elapsed   = time.time() - train_start
        mean_loss = total_loss / len(train_loader)

        for name, mod in leaf_modules:
            layer_logs[name]["grad_norms"].append(np.mean(epoch_gnorms[name]))
            layer_logs[name]["alloc"].append(np.mean(epoch_slots[name]))
            pnorms = [p.data.norm().item() for p in mod.parameters()]
            layer_logs[name]["param_norms"].append(np.mean(pnorms) if pnorms else 0.0)

        epoch_logs.append({
            "epoch": epoch, "train_loss": mean_loss,
            "val_acc": val_acc, "time": elapsed,
            "flops": cumulative_flops, "partial_passes": epoch_partial,
        })
        print(f"[WBP]  Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}"
              f"  t={elapsed:.1f}s  partial={epoch_partial}")

    sharpness, err = hutchinson_trace(model)
    if err:
        problems.append(f"WBP Hutchinson estimator failed: {err}")

    return epoch_logs, layer_logs, sharpness, problems, n_layers

# ── Results writing ───────────────────────────────────────────────────────────
def write_results(adam_logs, adam_layers, adam_sharp, adam_problems,
                  wbp_logs,  wbp_layers,  wbp_sharp,  wbp_problems,
                  n_layers, leaf_names):
    problems = [f"Adam: {p}" for p in adam_problems] + \
               [f"WBP: {p}"  for p in wbp_problems]

    def fmt_sharp(v):
        return f"{v:.4f}" if v is not None else "N/A"

    lines = ["# Methodology and Results\n\n"]

    # 1. Algorithm description
    lines += [
        "## 1. Algorithm Description\n\n",
        "### Baseline: Adam\n\n",
        "Standard `torch.optim.Adam` with default betas (0.9, 0.999) and ε=1e-8. "
        "One gradient update per parameter per step via a single forward pass, cross-entropy loss, "
        "and full backward pass.\n\n",
        "### Novel: Weighted Backpropagation (WBP)\n\n",
        "The atomic unit of allocation is one leaf module as returned by `model.named_modules()` "
        "(modules with no children). For each outer training step:\n\n",
        "1. Register forward hooks on all leaf modules to cache their input and output activations "
        "(detached clones). Run a full forward pass, compute loss, run full backward pass. "
        "Record the mean gradient L2-norm across each module's parameters.\n",
        "2. Compute a softmax probability distribution over modules from their gradient norms "
        "with temperature τ = 1.0.\n",
        "3. Sample B = 2 × n_layers slots via `torch.multinomial` with replacement, giving each "
        "module an integer slot count (may be zero).\n",
        "4. For each module with ≥1 slot, in reverse index order (output → input): for each slot, "
        "zero all gradients; run the full model forward while bypassing modules 0..i-1 via forward "
        "hooks that return their cached (detached) outputs — this causes the computation graph to "
        "start at module i, so backward does not propagate past it; compute loss; call `backward()`; "
        "call `optimizer.step()` for module i only; recompute and cache module i's output "
        "using its updated weights on the cached input activation.\n\n",
        "Modules with no parameters (ReLU, MaxPool, etc.) have no associated optimizer; "
        "if sampled they consume slots without performing an update.\n\n",
    ]

    # 2. Hyperparameters
    lines += [
        "## 2. Hyperparameters and Setup\n\n",
        "| Parameter | Value |\n|---|---|\n",
        f"| Learning rate | {LR} |\n",
        f"| Adam betas | (0.9, 0.999) |\n",
        f"| Adam ε | 1e-8 |\n",
        f"| Batch size | {BATCH} |\n",
        f"| Epochs | {EPOCHS} |\n",
        f"| WBP temperature τ | {TAU} |\n",
        f"| WBP budget B | 2 × {n_layers} = {2*n_layers} |\n",
        f"| Leaf modules (n_layers) | {n_layers} |\n",
        f"| Random seed | {SEED} |\n",
        f"| Device | {DEVICE} |\n",
        f"| PyTorch | {torch.__version__} |\n",
        f"| torchvision | {torchvision.__version__} |\n",
        f"| Python | {__import__('sys').version.split()[0]} |\n",
        "| CIFAR-10 train augmentation | RandomHorizontalFlip, RandomCrop(32, padding=4) |\n",
        f"| CIFAR-10 normalization mean | {MEAN} |\n",
        f"| CIFAR-10 normalization std | {STD} |\n\n",
    ]

    # 3. Instrumentation methodology
    lines += [
        "## 3. Instrumentation Methodology\n\n",
        "- **Training loss**: mean `CrossEntropyLoss` over all batches per epoch.\n",
        "- **Validation accuracy**: fraction of correct predictions on the 10k CIFAR-10 test set "
        "(model in eval mode, `torch.no_grad()`).\n",
        "- **Wall-clock time**: cumulative seconds via `time.time()` measured from the start of "
        "training (data loading and model construction excluded).\n",
        "- **FLOPs**: `fvcore.nn.FlopCountAnalysis` with a single (1, 3, 32, 32) input for one "
        "forward pass. Baseline: 2× forward FLOPs per step (forward + backward approximation). "
        "WBP outer step: 2× forward FLOPs; each slot at layer index i: "
        "2 × (n_layers − i) / n_layers × forward FLOPs.\n",
        "- **Gradient norm per layer**: after `backward()` in the outer step, mean L2 norm across "
        "all parameter tensors of the leaf module.\n",
        "- **Allocation count**: Adam = 1.0 (constant); WBP = mean sampled slots per step over "
        "the epoch.\n",
        "- **Parameter norm**: mean L2 norm across each leaf module's parameter tensors at "
        "epoch end.\n",
        "- **Sharpness**: Hutchinson trace estimator — 10 Rademacher vectors, 256 validation "
        "samples; for each vector v, compute Hv = ∇(∇L·v) via `create_graph=True` double "
        "backward; trace estimate = v^T Hv; final estimate = mean over 10 vectors. "
        "MaxPool2d is temporarily replaced with AvgPool2d (same kernel/stride/padding) "
        "for this computation because MaxPool2d does not support second-order gradients.\n\n",
    ]

    # 4. Results tables
    lines += ["## 4. Results Tables\n\n"]

    lines += ["### Training Loss\n\n",
              "| Epoch | Adam | WBP |\n|---|---|---|\n"]
    for a, w in zip(adam_logs, wbp_logs):
        lines.append(f"| {a['epoch']} | {a['train_loss']:.4f} | {w['train_loss']:.4f} |\n")

    lines += ["\n### Validation Accuracy\n\n",
              "| Epoch | Adam | WBP |\n|---|---|---|\n"]
    for a, w in zip(adam_logs, wbp_logs):
        lines.append(f"| {a['epoch']} | {a['val_acc']:.4f} | {w['val_acc']:.4f} |\n")

    lines += ["\n### Cumulative FLOPs\n\n",
              "| Epoch | Adam | WBP |\n|---|---|---|\n"]
    for a, w in zip(adam_logs, wbp_logs):
        lines.append(f"| {a['epoch']} | {a['flops']:.3e} | {w['flops']:.3e} |\n")

    lines += ["\n### Loss Surface Sharpness (Hutchinson Trace)\n\n",
              "| Method | Trace Estimate |\n|---|---|\n",
              f"| Adam | {fmt_sharp(adam_sharp)} |\n",
              f"| WBP  | {fmt_sharp(wbp_sharp)} |\n\n"]

    # 5. Layer-wise allocation heatmap (WBP)
    lines += ["## 5. Layer-wise Allocation Heatmap (WBP mean slots/step)\n\n"]
    header = "| Layer | " + " | ".join(f"E{e+1}" for e in range(EPOCHS)) + " |\n"
    sep    = "|---|" + "---|" * EPOCHS + "\n"
    lines += [header, sep]
    for name in leaf_names:
        vals = wbp_layers[name]["alloc"]
        lines.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in vals) + " |\n")

    # 6. Training log
    lines += ["\n## 6. Training Log\n\n",
              "### Adam\n\n",
              "| Epoch | Train Loss | Val Acc | Time (s) | Cumul. FLOPs |\n|---|---|---|---|---|\n"]
    for a in adam_logs:
        lines.append(f"| {a['epoch']} | {a['train_loss']:.4f} | {a['val_acc']:.4f}"
                     f" | {a['time']:.1f} | {a['flops']:.3e} |\n")

    lines += ["\n### WBP\n\n",
              "| Epoch | Train Loss | Val Acc | Time (s) | Cumul. FLOPs | Partial Passes |\n"
              "|---|---|---|---|---|---|\n"]
    for w in wbp_logs:
        lines.append(f"| {w['epoch']} | {w['train_loss']:.4f} | {w['val_acc']:.4f}"
                     f" | {w['time']:.1f} | {w['flops']:.3e} | {w['partial_passes']} |\n")

    # 7. Problems
    lines += ["\n## 7. Problems Encountered\n\n"]
    if problems:
        for i, p in enumerate(problems, 1):
            lines.append(f"{i}. {p}\n")
    else:
        lines.append("None.\n")

    # 8. Conclusions
    al = adam_logs[-1]
    wl = wbp_logs[-1]
    lines += [
        "\n## 8. Conclusions\n\n",
        f"After {EPOCHS} epochs, Adam achieved {al['val_acc']:.4f} validation accuracy "
        f"(final training loss {al['train_loss']:.4f}) while WBP achieved "
        f"{wl['val_acc']:.4f} validation accuracy "
        f"(final training loss {wl['train_loss']:.4f}). "
        f"WBP consumed approximately {wl['flops']/al['flops']:.1f}× more FLOPs than Adam "
        f"due to the {2*n_layers}-slot budget of partial forward passes per outer step. "
        f"Loss surface sharpness: Adam = {fmt_sharp(adam_sharp)}, "
        f"WBP = {fmt_sharp(wbp_sharp)}.\n",
    ]

    with open("results.md", "w") as f:
        f.writelines(lines)

    print(f"Results written to: {os.path.abspath('results.md')}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Adam baseline ===")
    adam_logs, adam_layers, adam_sharp, adam_problems, n_layers = run_adam()

    print("\n=== WBP ===")
    wbp_logs, wbp_layers, wbp_sharp, wbp_problems, _ = run_wbp()

    leaf_names = [n for n, _ in get_leaf_modules(make_model())]

    write_results(
        adam_logs, adam_layers, adam_sharp, adam_problems,
        wbp_logs,  wbp_layers,  wbp_sharp,  wbp_problems,
        n_layers, leaf_names,
    )

    print(f"\nDone. Results at: {os.path.abspath('results.md')}")
