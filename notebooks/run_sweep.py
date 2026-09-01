"""
Orchestrates the hyperparameter sweep: launches each config as a separate
sweep_train.py process, pinned to one GPU via CUDA_VISIBLE_DEVICES, with a
fixed pool of concurrent slots so we never touch a GPU we haven't been told
is fair game.

Slot pool: GPUs 4 and 5 (completely idle, no other process on them at all)
get 2 concurrent slots each; GPUs 0,1,2,3,6,7 (already running other users'
jobs) get 1 slot each as authorized "leftover space" overflow -- one small
extra process alongside what's already there, never more.
"""
import itertools
import sys
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = ROOT.parent / "outputs_v4_sweep"
SWEEP_ROOT.mkdir(exist_ok=True)
PY = sys.executable
TRAIN_SCRIPT = str(ROOT / "sweep_train.py")

GPU_SLOTS = [4, 5, 4, 5, 0, 1, 2, 3, 6, 7]   # 10 concurrent slots total

GRID = {
    "learning_rate": [1e-3, 3e-4],
    "kappa": [0.5, 1.5],
    "n_osc_per_side": [25, 50],
    "v_hidden": [16, 32],
}
FIXED = dict(readout_hidden=32, teacher_forcing_end=0.1, batch_size=32,
             pad_pre=20, pad_post=20, epochs=250, patience=30, seed=42)

configs = []
keys = list(GRID.keys())
for combo in itertools.product(*GRID.values()):
    cfg = dict(zip(keys, combo))
    cfg.update(FIXED)
    name = f"lr{cfg['learning_rate']}_k{cfg['kappa']}_osc{cfg['n_osc_per_side']}_vh{cfg['v_hidden']}"
    cfg["name"] = name
    configs.append(cfg)

print(f"Total configs: {len(configs)}", flush=True)
manifest = {c["name"]: c for c in configs}
with open(SWEEP_ROOT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

pending = list(configs)
running = {}  # slot_idx -> (Popen, config_name, gpu)

def launch(cfg, gpu):
    out_dir = SWEEP_ROOT / cfg["name"]
    log_path = out_dir
    out_dir.mkdir(exist_ok=True)
    log_file = open(out_dir / "run.log", "w")
    cmd = [
        PY, TRAIN_SCRIPT,
        "--output_dir", str(out_dir),
        "--epochs", str(cfg["epochs"]),
        "--batch_size", str(cfg["batch_size"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--seed", str(cfg["seed"]),
        "--n_osc_per_side", str(cfg["n_osc_per_side"]),
        "--v_hidden", str(cfg["v_hidden"]),
        "--readout_hidden", str(cfg["readout_hidden"]),
        "--kappa", str(cfg["kappa"]),
        "--teacher_forcing_end", str(cfg["teacher_forcing_end"]),
        "--pad_pre", str(cfg["pad_pre"]),
        "--pad_post", str(cfg["pad_post"]),
        "--patience", str(cfg["patience"]),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": "/usr/bin:/bin"}
    import os
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    print(f"LAUNCHED {cfg['name']} on GPU {gpu} (pid {proc.pid})", flush=True)
    return proc, log_file

# prime all slots
for slot_idx, gpu in enumerate(GPU_SLOTS):
    if not pending:
        break
    cfg = pending.pop(0)
    proc, log_file = launch(cfg, gpu)
    running[slot_idx] = {"proc": proc, "name": cfg["name"], "gpu": gpu, "log_file": log_file}

while running or pending:
    time.sleep(10)
    for slot_idx in list(running.keys()):
        info = running[slot_idx]
        ret = info["proc"].poll()
        if ret is not None:
            info["log_file"].close()
            print(f"FINISHED {info['name']} on GPU {info['gpu']} (exit {ret})", flush=True)
            del running[slot_idx]
            if pending:
                cfg = pending.pop(0)
                gpu = GPU_SLOTS[slot_idx]
                proc, log_file = launch(cfg, gpu)
                running[slot_idx] = {"proc": proc, "name": cfg["name"], "gpu": gpu, "log_file": log_file}

print("ALL SWEEP CONFIGS DONE", flush=True)

# aggregate results
results = []
for cfg in configs:
    mpath = SWEEP_ROOT / cfg["name"] / "metrics.json"
    if mpath.exists():
        with open(mpath) as f:
            m = json.load(f)
        results.append((cfg["name"], m["best_val_loss"], m["best_epoch"], m["epochs_run"]))

results.sort(key=lambda r: r[1])
with open(SWEEP_ROOT / "leaderboard.json", "w") as f:
    json.dump(results, f, indent=2)

print("=== LEADERBOARD (best_val_loss ascending) ===", flush=True)
for name, val, best_ep, ep_run in results:
    print(f"  {val:.6f}  {name}  (best@{best_ep}, ran {ep_run} epochs)", flush=True)
