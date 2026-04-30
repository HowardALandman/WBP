#!/usr/bin/env python3
"""Run Adam and WBP-K=1 on one seed, measuring sharpness after every epoch.

Usage: python sharpness_trajectory.py --seed N
Writes sharpness_trajectory_<N>.json and sharpness_trajectory_<N>.png
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
SEED = args.seed

LR                 = 1e-3
BATCH              = 128
EPOCHS             = 30
N_RADEMACHER       = 10
HUTCHINSON_SAMPLES = 256

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def make_loaders(device):
    pin = device.type == "cuda"
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
        train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, val_loader

def make_model(device):
    return torchvision.models.resnet18(weights=None, num_classes=10).to(device)

def get_leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]

def validate(model, val_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += (model(inputs).argmax(1) == targets).sum().item()
            total   += targets.size(0)
    model.train()
    return correct / total

def hutchinson_trace(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    swapped = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.MaxPool2d):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            avg = nn.AvgPool2d(mod.kernel_size, stride=mod.stride, padding=mod.padding).to(device)
            setattr(parent, parts[-1], avg)
            swapped.append((parent, parts[-1], mod))
    inputs_list, targets_list, count = [], [], 0
    for inp, tgt in val_loader:
        inputs_list.append(inp); targets_list.append(tgt); count += inp.size(0)
        if count >= HUTCHINSON_SAMPLES: break
    inputs  = torch.cat(inputs_list)[:HUTCHINSON_SAMPLES].to(device)
    targets = torch.cat(targets_list)[:HUTCHINSON_SAMPLES].to(device)
    params  = [p for p in model.parameters() if p.requires_grad]
    traces  = []
    err_msg = None
    try:
        for _ in range(N_RADEMACHER):
            vs    = [torch.randint(0, 2, p.shape, device=device, dtype=torch.float32)*2-1 for p in params]
            out   = model(inputs)
            loss  = criterion(out, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv    = sum((g*v).sum() for g, v in zip(grads, vs))
            Hvs   = torch.autograd.grad(gv, params, retain_graph=False)
            traces.append(sum((v*hv).sum().item() for v, hv in zip(vs, Hvs)))
    except Exception as exc:
        err_msg = str(exc)
    finally:
        for parent, attr, orig in swapped: setattr(parent, attr, orig)
        model.train()
    if err_msg:
        print(f"  Hutchinson warning: {err_msg}")
        return None
    return float(np.mean(traces))

def make_bypass_hook(cached_out):
    def hook(module, inp, out): return cached_out
    return hook

# ── Adam training ─────────────────────────────────────────────────────────────
def run_adam(device, train_loader, val_loader):
    print(f"\n=== Adam (seed={SEED}) ===")
    model     = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    log = []
    t0  = time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc   = validate(model, val_loader, device)
        sharpness = hutchinson_trace(model, val_loader, device)
        elapsed   = time.time() - t0
        print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc:.4f}  sharp={sharpness:.1f}  t={elapsed:.1f}s")
        log.append({"epoch": epoch, "val_acc": val_acc, "sharpness": sharpness, "time": elapsed})
    return log

# ── WBP-K=1 training ─────────────────────────────────────────────────────────
def wbp_step(model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets, device):
    cache_out = {}
    handles   = []
    def make_cache_hook(nm):
        def hook(mod, inp, out): cache_out[nm] = out.detach().clone()
        return hook
    for name, mod in leaf_modules:
        handles.append(mod.register_forward_hook(make_cache_hook(name)))
    floor_opt.zero_grad()
    loss = criterion(model(inputs), targets)
    step_loss = loss.item()
    loss.backward()
    for h in handles: h.remove()
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
        loss = criterion(model(inputs), targets)
        loss.backward()
        for h in bypass: h.remove()
        bonus_opt.step()
    return step_loss

def run_wbp(device, train_loader, val_loader):
    print(f"\n=== WBP-K=1 (seed={SEED}) ===")
    model        = make_model(device)
    leaf_modules = get_leaf_modules(model)
    criterion    = nn.CrossEntropyLoss()
    floor_opt    = torch.optim.Adam(model.parameters(), lr=LR)
    bonus_opts   = {name: torch.optim.Adam(list(mod.parameters()), lr=LR)
                    for name, mod in leaf_modules if list(mod.parameters())}
    log = []
    t0  = time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_steps    = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            total_loss += wbp_step(model, leaf_modules, floor_opt, bonus_opts,
                                   criterion, inputs, targets, device)
            n_steps    += 1
        val_acc   = validate(model, val_loader, device)
        sharpness = hutchinson_trace(model, val_loader, device)
        elapsed   = time.time() - t0
        print(f"  Epoch {epoch:2d}: loss={total_loss/n_steps:.4f}  "
              f"val_acc={val_acc:.4f}  sharp={sharpness:.1f}  t={elapsed:.1f}s")
        log.append({"epoch": epoch, "val_acc": val_acc, "sharpness": sharpness, "time": elapsed})
    return log

# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot(adam_log, wbp_log, seed, out_path):
    epochs      = [e["epoch"]    for e in adam_log]
    adam_sharp  = [e["sharpness"] for e in adam_log]
    wbp_sharp   = [e["sharpness"] for e in wbp_log]
    adam_acc    = [e["val_acc"]   for e in adam_log]
    wbp_acc     = [e["val_acc"]   for e in wbp_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, adam_sharp, color="#2271b2", linewidth=2, marker="o",
             markersize=3.5, label="Adam")
    ax1.plot(epochs, wbp_sharp,  color="#e66101", linewidth=2, marker="o",
             markersize=3.5, label="WBP-K=1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sharpness (Hutchinson trace)")
    ax1.set_title(f"Loss Surface Sharpness over Training (seed {seed})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, EPOCHS)

    ax2.plot(epochs, adam_acc, color="#2271b2", linewidth=2, marker="o",
             markersize=3.5, label="Adam")
    ax2.plot(epochs, wbp_acc,  color="#e66101", linewidth=2, marker="o",
             markersize=3.5, label="WBP-K=1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title(f"Validation Accuracy over Training (seed {seed})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, EPOCHS)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {os.path.abspath(out_path)}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    print(f"Seed: {SEED}  Device: {device}")

    set_seed(SEED)
    train_loader, val_loader = make_loaders(device)
    adam_log = run_adam(device, train_loader, val_loader)

    set_seed(SEED)
    train_loader, val_loader = make_loaders(device)
    wbp_log = run_wbp(device, train_loader, val_loader)

    json_path = f"sharpness_trajectory_{SEED}.json"
    with open(json_path, "w") as f:
        json.dump({"seed": SEED, "adam": adam_log, "wbp_k1": wbp_log}, f, indent=2)
    print(f"Trajectory data saved to {os.path.abspath(json_path)}")

    plot_path = f"sharpness_trajectory_{SEED}.png"
    make_plot(adam_log, wbp_log, SEED, plot_path)

    print(f"\nFinal epoch 30:")
    print(f"  Adam   sharpness={adam_log[-1]['sharpness']:.2f}  val_acc={adam_log[-1]['val_acc']:.4f}")
    print(f"  WBP-K1 sharpness={wbp_log[-1]['sharpness']:.2f}  val_acc={wbp_log[-1]['val_acc']:.4f}")
