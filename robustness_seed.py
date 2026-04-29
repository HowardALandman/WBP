#!/usr/bin/env python3
"""Robustness check: Adam vs WBP-K=1 across seeds.

Usage: python robustness_seed.py --seed N
Writes robustness_N.json with val_acc and sharpness for both methods.
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
from fvcore.nn import FlopCountAnalysis

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
SEED = args.seed

# ── Seed ──────────────────────────────────────────────────────────────────────
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

print(f"Seed: {SEED}  Device: {DEVICE}")
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

def make_loaders():
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=PIN)
    val_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=PIN)
    return train_loader, val_loader

# ── Model helpers ─────────────────────────────────────────────────────────────
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
def validate(model, val_loader):
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
def hutchinson_trace(model, val_loader):
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

# ── Adam baseline ─────────────────────────────────────────────────────────────
def run_adam(train_loader, val_loader):
    model     = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_start = time.time()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc   = validate(model, val_loader)
        mean_loss = total_loss / len(train_loader)
        elapsed   = time.time() - train_start
        print(f"[Adam seed={SEED}] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}  t={elapsed:.1f}s")

    final_val_acc = val_acc
    sharpness, err = hutchinson_trace(model, val_loader)
    if err:
        print(f"  Adam Hutchinson error: {err}")
    return final_val_acc, sharpness

# ── WBP-K=1 (WBP-Minimal) ────────────────────────────────────────────────────
def wbp_minimal_step(model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets):
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

    return step_loss


def run_wbp_minimal(train_loader, val_loader):
    model        = make_model()
    leaf_modules = get_leaf_modules(model)
    criterion    = nn.CrossEntropyLoss()

    floor_opt = torch.optim.Adam(model.parameters(), lr=LR)
    bonus_opts = {}
    for name, mod in leaf_modules:
        params = list(mod.parameters())
        if params:
            bonus_opts[name] = torch.optim.Adam(params, lr=LR)

    train_start = time.time()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_steps    = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            step_loss = wbp_minimal_step(
                model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets)
            total_loss += step_loss
            n_steps    += 1

        val_acc   = validate(model, val_loader)
        mean_loss = total_loss / n_steps
        elapsed   = time.time() - train_start
        print(f"[WBP-K1 seed={SEED}] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}  t={elapsed:.1f}s")

    final_val_acc = val_acc
    sharpness, err = hutchinson_trace(model, val_loader)
    if err:
        print(f"  WBP-K1 Hutchinson error: {err}")
    return final_val_acc, sharpness

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = make_loaders()

    print("=== Adam ===")
    adam_val_acc, adam_sharp = run_adam(train_loader, val_loader)

    # Re-seed before WBP so both methods start from the same state
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    train_loader, val_loader = make_loaders()

    print("\n=== WBP-K=1 ===")
    wbp_val_acc, wbp_sharp = run_wbp_minimal(train_loader, val_loader)

    result = {
        "seed": SEED,
        "adam": {"val_acc": adam_val_acc, "sharpness": adam_sharp},
        "wbp_k1": {"val_acc": wbp_val_acc, "sharpness": wbp_sharp},
    }

    out_path = f"robustness_{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSeed {SEED} done.")
    print(f"  Adam:   val_acc={adam_val_acc:.4f}  sharpness={adam_sharp}")
    print(f"  WBP-K1: val_acc={wbp_val_acc:.4f}  sharpness={wbp_sharp}")
    print(f"  Saved to {os.path.abspath(out_path)}")
