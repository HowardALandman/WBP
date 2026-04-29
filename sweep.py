#!/usr/bin/env python3
"""WBP bonus-slot sweep K=2..20. Appends ## WBP Bonus Slot Sweep to results.md."""

import os
import re
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

# ── Constants ─────────────────────────────────────────────────────────────────
SEED               = 42
LR                 = 1e-3
BATCH              = 128
EPOCHS             = 30
N_RADEMACHER       = 10
HUTCHINSON_SAMPLES = 256
K_VALUES           = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
PIN = DEVICE.type == "cuda"
print(f"Device: {DEVICE}")

# ── Seed / Data (created once; shuffle re-determined by reset_seed per run) ───
def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

reset_seed()

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

# ── Model helpers ─────────────────────────────────────────────────────────────
def make_model():
    return torchvision.models.resnet18(weights=None, num_classes=10).to(DEVICE)

def get_leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]

# ── FLOPs (computed once, cached) ─────────────────────────────────────────────
_FORWARD_FLOPS = None

def get_forward_flops():
    global _FORWARD_FLOPS
    if _FORWARD_FLOPS is None:
        tmp = make_model().cpu().eval()
        fa  = FlopCountAnalysis(tmp, torch.zeros(1, 3, 32, 32))
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        _FORWARD_FLOPS = fa.total()
        del tmp
    return _FORWARD_FLOPS

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
        inputs_list.append(inp); targets_list.append(tgt)
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

# ── One outer step ────────────────────────────────────────────────────────────
def outer_step(model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets):
    """
    Returns (step_loss, bonus_idx).
    Bonus optimizer already has lr = LR * K; one partial forward, one Adam step.
    """
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

    grad_norms = []
    for _, mod in leaf_modules:
        norms = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        grad_norms.append(np.mean(norms) if norms else 0.0)

    bonus_idx  = int(np.argmax(grad_norms))
    bonus_name = leaf_modules[bonus_idx][0]

    floor_opt.step()

    bonus_opt = bonus_opts.get(bonus_name)
    if bonus_opt is not None:
        model.zero_grad()

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

    return step_loss, bonus_idx

# ── Run one K value ───────────────────────────────────────────────────────────
def run_k(k):
    """
    Train for EPOCHS epochs with bonus lr = LR * k.
    Returns dict with final metrics.
    """
    problems = []

    reset_seed()  # identical model init and data order for every K
    model        = make_model()
    leaf_modules = get_leaf_modules(model)
    n_layers     = len(leaf_modules)
    forward_flops = get_forward_flops()

    floor_opt = torch.optim.Adam(model.parameters(), lr=LR)

    bonus_opts = {}
    for name, mod in leaf_modules:
        params = list(mod.parameters())
        if params:
            bonus_opts[name] = torch.optim.Adam(params, lr=LR * k)

    criterion = nn.CrossEntropyLoss()

    train_start      = time.time()
    cumulative_flops = 0
    final_loss       = 0.0
    final_acc        = 0.0

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_steps    = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            step_loss, bonus_idx = outer_step(
                model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets
            )
            total_loss += step_loss
            n_steps    += 1

            fraction = (n_layers - bonus_idx) / n_layers
            cumulative_flops += int(2 * forward_flops * (1.0 + fraction))

        val_acc   = validate(model)
        elapsed   = time.time() - train_start
        mean_loss = total_loss / n_steps
        final_loss = mean_loss
        final_acc  = val_acc

        print(f"  [K={k}] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}"
              f"  t={elapsed:.1f}s")

    sharpness, err = hutchinson_trace(model)
    if err:
        problems.append(f"K={k} Hutchinson failed: {err}")

    wall_time = time.time() - train_start

    return {
        'k':          k,
        'val_acc':    final_acc,
        'train_loss': final_loss,
        'sharpness':  sharpness,
        'flops':      cumulative_flops,
        'time':       wall_time,
        'problems':   problems,
    }

# ── Parse Adam + K=1 reference values ────────────────────────────────────────
def parse_reference_results(path):
    with open(path) as f:
        content = f.read()

    match = re.search(
        r'### Scalar Summary\n\n\|.*?\n\|---.*?\n((?:\|[^\n]+\n)+)',
        content,
    )
    adam = {}
    k1   = {}
    if match:
        for line in match.group(1).strip().split('\n'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) < 3:
                continue
            metric, a_val, w_val = parts[0].lower(), parts[1], parts[2]
            try:
                if 'accuracy' in metric:
                    adam['val_acc']    = float(a_val)
                    k1['val_acc']      = float(w_val)
                elif 'train loss' in metric:
                    adam['train_loss'] = float(a_val)
                    k1['train_loss']   = float(w_val)
                elif 'sharpness' in metric:
                    adam['sharpness']  = float(a_val)
                    k1['sharpness']    = float(w_val)
                elif 'total flops' in metric:
                    adam['flops']      = float(a_val)
                    k1['flops']        = float(w_val)
                elif 'training time' in metric:
                    adam['time']       = float(a_val)
                    k1['time']         = float(w_val)
            except ValueError:
                pass

    return adam, k1

# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(k_values, sweep_results, adam_ref, k1_ref):
    ks       = k_values
    accs     = [sweep_results[k]['val_acc']   for k in ks]
    sharps   = [sweep_results[k]['sharpness'] for k in ks]

    # Include K=1 in plot series
    all_ks    = [1] + ks
    all_accs  = [k1_ref.get('val_acc', float('nan'))]   + accs
    all_sharps = [k1_ref.get('sharpness', float('nan'))] + sharps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Val accuracy
    ax1.plot(all_ks, all_accs, 'o-', color='steelblue', label='WBP-Min (K)')
    ax1.axhline(adam_ref.get('val_acc', float('nan')),
                linestyle='--', color='tomato', label='Adam baseline')
    ax1.set_xlabel('K (bonus slots)')
    ax1.set_ylabel('Final validation accuracy')
    ax1.set_title('Val Accuracy vs K')
    ax1.legend()
    ax1.set_xticks(all_ks)
    ax1.grid(True, alpha=0.3)

    # Sharpness
    valid_sharps = [(k, s) for k, s in zip(all_ks, all_sharps) if s is not None]
    if valid_sharps:
        vks, vss = zip(*valid_sharps)
        ax2.plot(vks, vss, 'o-', color='darkorange', label='WBP-Min (K)')
    ax2.axhline(adam_ref.get('sharpness', float('nan')),
                linestyle='--', color='tomato', label='Adam baseline')
    ax2.set_xlabel('K (bonus slots)')
    ax2.set_ylabel('Sharpness (Hutchinson trace)')
    ax2.set_title('Sharpness vs K')
    ax2.legend()
    ax2.set_xticks(all_ks)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_plot.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ── Append section to results.md ──────────────────────────────────────────────
def append_sweep_section(results_path, k_values, sweep_results, adam_ref, k1_ref,
                         plot_path, all_problems):
    def fv(v, fmt='.4f'):
        return format(v, fmt) if v is not None else 'N/A'

    lines = ["\n## WBP Bonus Slot Sweep\n\n"]

    # Summary table (Adam + K=1 + sweep)
    lines += [
        "### Summary Table\n\n",
        "Bonus update: one partial forward pass, one Adam step with lr = base_lr × K. "
        "All runs use seed 42, identical model initialization.\n\n",
        "| K | Final Val Acc | Final Train Loss | Sharpness | Total FLOPs | Wall-clock (s) |\n",
        "|---|---|---|---|---|---|\n",
        f"| Adam (baseline) | {fv(adam_ref.get('val_acc'))} |"
        f" {fv(adam_ref.get('train_loss'))} |"
        f" {fv(adam_ref.get('sharpness'))} |"
        f" {adam_ref.get('flops', float('nan')):.3e} |"
        f" {fv(adam_ref.get('time'), '.1f')} |\n",
        f"| 1 (WBP-Minimal) | {fv(k1_ref.get('val_acc'))} |"
        f" {fv(k1_ref.get('train_loss'))} |"
        f" {fv(k1_ref.get('sharpness'))} |"
        f" {k1_ref.get('flops', float('nan')):.3e} |"
        f" {fv(k1_ref.get('time'), '.1f')} |\n",
    ]
    for k in k_values:
        r = sweep_results[k]
        lines.append(
            f"| {k} | {fv(r['val_acc'])} | {fv(r['train_loss'])} |"
            f" {fv(r['sharpness'])} | {r['flops']:.3e} | {r['time']:.1f} |\n"
        )

    # Best K values
    valid_acc   = {k: sweep_results[k]['val_acc'] for k in k_values
                   if sweep_results[k]['val_acc'] is not None}
    valid_sharp = {k: sweep_results[k]['sharpness'] for k in k_values
                   if sweep_results[k]['sharpness'] is not None}

    if valid_acc:
        best_acc_k = max(valid_acc, key=valid_acc.get)
        # Also compare K=1
        if k1_ref.get('val_acc', 0) > valid_acc[best_acc_k]:
            best_acc_k = 1
            best_acc_v = k1_ref['val_acc']
        else:
            best_acc_v = valid_acc[best_acc_k]
        lines.append(f"\n**Best val accuracy**: K={best_acc_k} ({best_acc_v:.4f})\n")

    if valid_sharp:
        min_sharp_k = min(valid_sharp, key=valid_sharp.get)
        # Also compare K=1
        k1_sharp = k1_ref.get('sharpness')
        if k1_sharp is not None and k1_sharp < valid_sharp[min_sharp_k]:
            min_sharp_k = 1
            min_sharp_v = k1_sharp
        else:
            min_sharp_v = valid_sharp[min_sharp_k]
        lines.append(f"\n**Flattest model (min sharpness)**: K={min_sharp_k} ({min_sharp_v:.4f})\n")

    # Plot reference
    lines += [
        f"\n![Val accuracy and sharpness vs K](sweep_plot.png)\n\n",
    ]

    # Problems
    if all_problems:
        lines += ["\n### Problems Encountered\n\n"]
        for i, p in enumerate(all_problems, 1):
            lines.append(f"{i}. {p}\n")
    else:
        lines.append("\n**Problems encountered**: None.\n")

    with open(results_path, 'a') as f:
        f.writelines(lines)

    print(f"Appended sweep results to {os.path.abspath(results_path)}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sweep_results = {}
    all_problems  = []

    # Pre-compute forward FLOPs (needs one model creation)
    reset_seed()
    _ = get_forward_flops()
    print(f"Forward FLOPs: {_FORWARD_FLOPS:.3e}")

    for k in K_VALUES:
        print(f"\n{'='*50}\n=== K={k} ({K_VALUES.index(k)+1}/{len(K_VALUES)}) ===\n{'='*50}")
        result = run_k(k)
        sweep_results[k] = result
        all_problems.extend(result['problems'])
        print(f"K={k} done: val_acc={result['val_acc']:.4f}  "
              f"sharp={result['sharpness']:.1f}  "
              f"time={result['time']:.0f}s")

    adam_ref, k1_ref = parse_reference_results("results.md")
    if not adam_ref:
        print("WARNING: could not parse reference results from results.md")

    plot_path = make_plot(K_VALUES, sweep_results, adam_ref, k1_ref)
    print(f"Plot saved to {plot_path}")

    append_sweep_section(
        "results.md", K_VALUES, sweep_results,
        adam_ref, k1_ref, plot_path, all_problems,
    )

    print(f"\nDone. Results at: {os.path.abspath('results.md')}")
