#!/usr/bin/env python3
"""Hybrid training: N epochs Adam then M epochs WBP-K=1, or vice versa.

Usage:
  python hybrid_seed.py --seed 0 --mode adam_then_wbp   # 29 Adam + 1 WBP-K=1
  python hybrid_seed.py --seed 0 --mode wbp_then_adam   # 29 WBP-K=1 + 1 Adam

Writes hybrid_<mode>_<seed>.json with final val_acc and sharpness.
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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=["adam_then_wbp", "wbp_then_adam"], required=True)
parser.add_argument("--switch-epoch", type=int, default=29,
                    help="Epoch at which the optimizer switches (1-indexed; default 29 means switch after epoch 29)")
args = parser.parse_args()

SEED         = args.seed
MODE         = args.mode
SWITCH_EPOCH = args.switch_epoch   # first SWITCH_EPOCH epochs use phase-1 optimizer

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Seed: {SEED}  Mode: {MODE}  Switch after epoch: {SWITCH_EPOCH}  Device: {DEVICE}")
PIN = DEVICE.type == "cuda"

LR                 = 1e-3
BATCH              = 128
EPOCHS             = 30
N_RADEMACHER       = 10
HUTCHINSON_SAMPLES = 256

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

def make_model():
    return torchvision.models.resnet18(weights=None, num_classes=10).to(DEVICE)

def get_leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]

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
        inputs_list.append(inp); targets_list.append(tgt); count += inp.size(0)
        if count >= HUTCHINSON_SAMPLES: break
    inputs  = torch.cat(inputs_list)[:HUTCHINSON_SAMPLES].to(DEVICE)
    targets = torch.cat(targets_list)[:HUTCHINSON_SAMPLES].to(DEVICE)
    params  = [p for p in model.parameters() if p.requires_grad]
    traces  = []
    err_msg = None
    try:
        for _ in range(N_RADEMACHER):
            vs    = [torch.randint(0, 2, p.shape, device=DEVICE, dtype=torch.float32)*2-1 for p in params]
            out   = model(inputs)
            loss  = nn.CrossEntropyLoss()(out, targets)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            gv    = sum((g*v).sum() for g, v in zip(grads, vs))
            Hvs   = torch.autograd.grad(gv, params, retain_graph=False)
            traces.append(sum((v*hv).sum().item() for v, hv in zip(vs, Hvs)))
    except Exception as exc:
        err_msg = str(exc)
    finally:
        for parent, attr, orig in swapped:
            setattr(parent, attr, orig)
        model.train()
    if err_msg:
        return None, err_msg
    return float(np.mean(traces)), None

def make_bypass_hook(cached_out):
    def hook(module, inp, out):
        return cached_out
    return hook

def adam_epoch(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def wbp_step(model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets):
    cache_out = {}
    handles   = []
    def make_cache_hook(nm):
        def hook(mod, inp, out):
            cache_out[nm] = out.detach().clone()
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

def wbp_epoch(model, leaf_modules, floor_opt, bonus_opts, criterion, train_loader):
    model.train()
    total_loss = 0.0
    n_steps    = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        total_loss += wbp_step(model, leaf_modules, floor_opt, bonus_opts, criterion, inputs, targets)
        n_steps    += 1
    return total_loss / n_steps

if __name__ == "__main__":
    # ── Main training loop ────────────────────────────────────────────────────
    model        = make_model()
    leaf_modules = get_leaf_modules(model)
    criterion    = nn.CrossEntropyLoss()
    train_start  = time.time()

    # Phase-1 optimizers
    if MODE == "adam_then_wbp":
        phase1_opt = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        phase1_floor = torch.optim.Adam(model.parameters(), lr=LR)
        phase1_bonus = {name: torch.optim.Adam(list(mod.parameters()), lr=LR)
                        for name, mod in leaf_modules if list(mod.parameters())}

    model.train()
    for epoch in range(1, EPOCHS + 1):
        if MODE == "adam_then_wbp":
            if epoch <= SWITCH_EPOCH:
                mean_loss = adam_epoch(model, phase1_opt, criterion, train_loader)
                tag = "Adam"
            else:
                # Build WBP optimizers fresh at the switch point
                if epoch == SWITCH_EPOCH + 1:
                    wbp_floor = torch.optim.Adam(model.parameters(), lr=LR)
                    wbp_bonus = {name: torch.optim.Adam(list(mod.parameters()), lr=LR)
                                 for name, mod in leaf_modules if list(mod.parameters())}
                mean_loss = wbp_epoch(model, leaf_modules, wbp_floor, wbp_bonus, criterion, train_loader)
                tag = "WBP-K1"
        else:  # wbp_then_adam
            if epoch <= SWITCH_EPOCH:
                mean_loss = wbp_epoch(model, leaf_modules, phase1_floor, phase1_bonus, criterion, train_loader)
                tag = "WBP-K1"
            else:
                if epoch == SWITCH_EPOCH + 1:
                    adam_opt = torch.optim.Adam(model.parameters(), lr=LR)
                mean_loss = adam_epoch(model, adam_opt, criterion, train_loader)
                tag = "Adam"

        val_acc = validate(model)
        elapsed = time.time() - train_start
        print(f"[{tag}] Epoch {epoch:2d}: loss={mean_loss:.4f}  val_acc={val_acc:.4f}  t={elapsed:.1f}s")

    final_val_acc = val_acc
    sharpness, err = hutchinson_trace(model)
    if err:
        print(f"Hutchinson error: {err}")

    result = {
        "seed": SEED,
        "mode": MODE,
        "switch_epoch": SWITCH_EPOCH,
        "val_acc": final_val_acc,
        "sharpness": sharpness,
    }
    out_path = f"hybrid_{MODE}_{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone: val_acc={final_val_acc:.4f}  sharpness={sharpness}")
    print(f"Saved to {os.path.abspath(out_path)}")
