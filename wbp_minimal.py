#!/usr/bin/env python3
"""WBP-Minimal: one bonus-layer update per outer step. Appends results to results.md."""

import os
import re
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fvcore.nn import FlopCountAnalysis

# ── Seed (must match Adam run exactly) ───────────────────────────────────────
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
    train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=PIN)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=PIN)

# ── Model ─────────────────────────────────────────────────────────────────────
def make_model():
    return torchvision.models.resnet18(weights=None, num_classes=10).to(DEVICE)

def get_leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]

# ── FLOPs ─────────────────────────────────────────────────────────────────────
def count_forward_flops(model):
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
            correct += (model(inputs).argmax(1) == targets).sum().item()
            total   += targets.size(0)
    model.train()
    return correct / total

# ── Hutchinson trace ──────────────────────────────────────────────────────────
def hutchinson_trace(model):
    """Approximate Hessian trace. MaxPool2d swapped to AvgPool2d for double-backward."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    swapped = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.MaxPool2d):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            avg = nn.AvgPool2d(mod.kernel_size, stride=mod.stride, padding=mod.padding).to(DEVICE)
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

    params  = [p for p in model.parameters() if p.requires_grad]
    traces  = []
    err_msg = None
    try:
        for _ in range(N_RADEMACHER):
            vs    = [torch.randint(0, 2, p.shape, device=DEVICE, dtype=torch.float32) * 2 - 1
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
    def hook(module, inp, out):
        return cached_out
    return hook

# ── One outer WBP-Minimal step ────────────────────────────────────────────────
def wbp_minimal_step(model, leaf_modules, n_layers, floor_opt, bonus_opts,
                     criterion, inputs, targets):
    """
    Returns (step_loss, grad_norms, bonus_idx).

    Step 1: cache all leaf-module outputs during full forward+backward.
    Step 2: pick bonus layer = argmax(grad_norms).
    Step 3: floor Adam step over all parameters.
    Step 4: partial forward from bonus layer (pre-bonus outputs bypassed with
            cached values) → backward → Adam step for bonus layer only.
    """
    # ── Step 1 ───────────────────────────────────────────────────────────────
    cache_out = {}
    handles   = []

    def make_cache_hook(nm):
        def hook(mod, inp, out):
            cache_out[nm] = out.detach().clone()
        return hook

    for name, mod in leaf_modules:
        handles.append(mod.register_forward_hook(make_cache_hook(name)))

    floor_opt.zero_grad()
    out       = model(inputs)
    loss      = criterion(out, targets)
    step_loss = loss.item()
    loss.backward()

    for h in handles:
        h.remove()

    # ── Gradient norms ────────────────────────────────────────────────────────
    grad_norms = []
    for _, mod in leaf_modules:
        norms = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        grad_norms.append(np.mean(norms) if norms else 0.0)

    # ── Step 2: bonus layer ───────────────────────────────────────────────────
    bonus_idx  = int(np.argmax(grad_norms))
    bonus_name = leaf_modules[bonus_idx][0]

    # ── Step 3: floor update ──────────────────────────────────────────────────
    floor_opt.step()

    # ── Step 4: bonus update ──────────────────────────────────────────────────
    bonus_opt = bonus_opts.get(bonus_name)
    if bonus_opt is not None:
        model.zero_grad()

        # Bypass modules 0..bonus_idx-1 with their cached (pre-floor-update) outputs.
        bypass = []
        for j in range(bonus_idx):
            name_j, mod_j = leaf_modules[j]
            bypass.append(mod_j.register_forward_hook(make_bypass_hook(cache_out[name_j])))

        out  = model(inputs)
        loss = criterion(out, targets)
        loss.backward()

        for h in bypass:
            h.remove()

        bonus_opt.step()

    return step_loss, grad_norms, bonus_idx

# ── Training run ──────────────────────────────────────────────────────────────
def run_wbp_minimal():
    problems     = []
    model        = make_model()
    leaf_modules = get_leaf_modules(model)
    n_layers     = len(leaf_modules)
    leaf_names   = [n for n, _ in leaf_modules]

    forward_flops = count_forward_flops(model)

    # One Adam over all parameters for the floor update (identical to Adam baseline)
    floor_opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Per-layer bonus optimizers; persisted across steps for momentum continuity.
    bonus_opts = {}
    for name, mod in leaf_modules:
        params = list(mod.parameters())
        if params:
            bonus_opts[name] = torch.optim.Adam(params, lr=LR)

    criterion = nn.CrossEntropyLoss()

    epoch_logs = []
    layer_logs = {n: {"grad_norms": [], "alloc": [], "param_norms": []}
                  for n in leaf_names}
    # Cumulative bonus-selection count per layer across all epochs
    bonus_hist = {n: 0 for n in leaf_names}

    train_start      = time.time()
    cumulative_flops = 0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss  = 0.0
        step_gnorms = {n: [] for n in leaf_names}
        step_bonus  = {n: 0  for n in leaf_names}
        n_steps     = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            step_loss, grad_norms, bonus_idx = wbp_minimal_step(
                model, leaf_modules, n_layers,
                floor_opt, bonus_opts,
                criterion, inputs, targets,
            )
            total_loss += step_loss
            n_steps    += 1

            bn = leaf_names[bonus_idx]
            step_bonus[bn] += 1
            bonus_hist[bn] += 1

            # FLOPs: full fwd+bwd + partial bonus fwd+bwd
            fraction = (n_layers - bonus_idx) / n_layers
            cumulative_flops += int(2 * forward_flops * (1.0 + fraction))

            for idx, name in enumerate(leaf_names):
                step_gnorms[name].append(grad_norms[idx])

        val_acc   = validate(model)
        elapsed   = time.time() - train_start
        mean_loss = total_loss / n_steps

        # Most frequently selected bonus layer this epoch
        epoch_bonus = max(step_bonus, key=step_bonus.get)

        for name, mod in leaf_modules:
            layer_logs[name]["grad_norms"].append(np.mean(step_gnorms[name]))
            # alloc = floor(1) + fraction of steps this layer was the bonus
            layer_logs[name]["alloc"].append(1.0 + step_bonus[name] / n_steps)
            pnorms = [p.data.norm().item() for p in mod.parameters()]
            layer_logs[name]["param_norms"].append(np.mean(pnorms) if pnorms else 0.0)

        epoch_logs.append({
            "epoch": epoch, "train_loss": mean_loss,
            "val_acc": val_acc, "time": elapsed,
            "flops": cumulative_flops, "epoch_bonus": epoch_bonus,
        })
        print(f"[WBP-Min] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}"
              f"  t={elapsed:.1f}s  bonus={epoch_bonus}")

    sharpness, err = hutchinson_trace(model)
    if err:
        problems.append(f"Hutchinson estimator failed: {err}")

    return epoch_logs, layer_logs, bonus_hist, sharpness, problems, leaf_names, forward_flops

# ── Parse Adam results from results.md ───────────────────────────────────────
def parse_adam_results(path):
    with open(path) as f:
        content = f.read()

    log_match = re.search(
        r'### Adam\n\n\| Epoch.*?\n\|---.*?\n((?:\|[^\n]+\n)+)',
        content,
    )
    adam_logs = []
    if log_match:
        for line in log_match.group(1).strip().split('\n'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if parts and parts[0].isdigit():
                adam_logs.append({
                    'epoch':      int(parts[0]),
                    'train_loss': float(parts[1]),
                    'val_acc':    float(parts[2]),
                    'time':       float(parts[3]),
                    'flops':      float(parts[4]),
                })

    sharp_match = re.search(r'\| Adam \| ([\d.e+\-]+) \|', content)
    adam_sharp = float(sharp_match.group(1)) if sharp_match else None

    return adam_logs, adam_sharp

# ── Append WBP-Minimal section to results.md ──────────────────────────────────
def append_results(results_path, wbp_logs, layer_logs, bonus_hist, sharpness,
                   problems, leaf_names, adam_logs, adam_sharp):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    lines = ["\n## WBP-Minimal Results\n\n"]

    # Algorithm description
    lines += [
        "### Algorithm Description\n\n",
        "Each outer step:\n\n",
        "1. Forward hooks cache all leaf-module output activations (detached clones). "
        "Full forward pass, cross-entropy loss, full backward pass. "
        "Record mean gradient L2-norm per leaf module.\n",
        "2. Identify the **bonus layer**: the leaf module with the largest gradient norm.\n",
        "3. **Floor update**: one Adam step over all model parameters "
        "(identical to the Adam baseline).\n",
        "4. **Bonus update**: zero gradients; run the full model forward with modules "
        "0..i-1 bypassed via forward hooks returning their cached pre-update outputs, "
        "so the computation graph starts at the bonus layer; compute loss; backward; "
        "one Adam step on the bonus layer's parameters only.\n\n",
        "The floor optimizer is one Adam instance over all parameters. "
        "Each leaf module has its own separate Adam optimizer for bonus updates, "
        "persisted across steps so momentum accumulates correctly.\n\n",
    ]

    # Training log
    lines += [
        "### Training Log\n\n",
        "| Epoch | Train Loss | Val Acc | Time (s) | Cumul. FLOPs | Epoch Bonus Layer |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for w in wbp_logs:
        lines.append(
            f"| {w['epoch']} | {w['train_loss']:.4f} | {w['val_acc']:.4f}"
            f" | {w['time']:.1f} | {w['flops']:.3e} | {w['epoch_bonus']} |\n"
        )

    # Sharpness
    lines += [f"\n**Hutchinson Hessian Trace**: {fmt(sharpness)} "
              "(MaxPool2d replaced with AvgPool2d for double-backward compatibility)\n\n"]

    # Layer-wise allocation
    lines += ["### Layer-wise Mean Allocation per Epoch (floor=1 + bonus fraction)\n\n"]
    header = "| Layer | " + " | ".join(f"E{e+1}" for e in range(EPOCHS)) + " |\n"
    sep    = "|---|" + "---|" * EPOCHS + "\n"
    lines += [header, sep]
    for name in leaf_names:
        vals = layer_logs[name]["alloc"]
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n")

    # Bonus layer histogram
    total_steps = sum(bonus_hist.values())
    lines += ["\n### Bonus Layer Histogram (total steps selected, all epochs)\n\n",
              "| Layer | Steps | % |\n|---|---|---|\n"]
    for name in leaf_names:
        cnt = bonus_hist[name]
        if cnt > 0:
            lines.append(f"| {name} | {cnt} | {100*cnt/total_steps:.1f}% |\n")

    if problems:
        lines += ["\n### Problems Encountered\n\n"]
        for i, p in enumerate(problems, 1):
            lines.append(f"{i}. {p}\n")

    # Side-by-side comparison
    lines += ["\n## Adam vs WBP-Minimal Comparison\n\n"]

    lines += [
        "### Per-Epoch Metrics\n\n",
        "| Epoch | Adam Loss | WBP-Min Loss | Adam Acc | WBP-Min Acc |"
        " Adam FLOPs | WBP-Min FLOPs |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for a, w in zip(adam_logs, wbp_logs):
        lines.append(
            f"| {a['epoch']}"
            f" | {a['train_loss']:.4f} | {w['train_loss']:.4f}"
            f" | {a['val_acc']:.4f} | {w['val_acc']:.4f}"
            f" | {a['flops']:.3e} | {w['flops']:.3e} |\n"
        )

    al = adam_logs[-1]
    wl = wbp_logs[-1]
    flops_ratio = wl['flops'] / al['flops'] if al['flops'] else float('nan')
    lines += [
        "\n### Scalar Summary\n\n",
        "| Metric | Adam | WBP-Minimal |\n|---|---|---|\n",
        f"| Final val accuracy | {al['val_acc']:.4f} | {wl['val_acc']:.4f} |\n",
        f"| Final train loss | {al['train_loss']:.4f} | {wl['train_loss']:.4f} |\n",
        f"| Sharpness (Hutchinson trace) | {fmt(adam_sharp)} | {fmt(sharpness)} |\n",
        f"| Total FLOPs | {al['flops']:.3e} | {wl['flops']:.3e} |\n",
        f"| Training time (s) | {al['time']:.1f} | {wl['time']:.1f} |\n",
        f"| FLOPs ratio (WBP-Min / Adam) | 1.00 | {flops_ratio:.2f} |\n",
    ]

    with open(results_path, 'a') as f:
        f.writelines(lines)

    print(f"Appended results to {os.path.abspath(results_path)}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    (wbp_logs, layer_logs, bonus_hist, sharpness,
     problems, leaf_names, forward_flops) = run_wbp_minimal()

    adam_logs, adam_sharp = parse_adam_results("results.md")
    if not adam_logs:
        print("WARNING: could not parse Adam training log from results.md")

    append_results(
        "results.md",
        wbp_logs, layer_logs, bonus_hist,
        sharpness, problems, leaf_names,
        adam_logs, adam_sharp,
    )

    print(f"\nDone. Results at: {os.path.abspath('results.md')}")
