"""
v4 + low-frequency smoothing stage: after the mu-band (8-13Hz) oscillator
population, a second ComplexLinear+ComplexReLU projection feeds a second
HopfCell tuned to 0.1-3Hz, and the readout consumes THAT stage's output
instead of the raw mu-band z. Mirrors the 3-stage cascade pattern from the
1d_track_concat sweep winner (osc_bands=[[1,4],[4,8],[0.1,3]]), applied
here as mu-band -> low-freq only (2 stages, not 3), since the mu-band
stage is the one that needs to carry the Kuramoto L/R coupling for this
study -- coupling stays there, not on the smoothing stage.

Also: pad_pre/pad_post default to 100 (up from 20), so there's enough
rest-period length on each side to do real synchrony-vs-desynchrony
statistics against the ~100-step motion window on equal footing.

Sweep-winning hyperparameters are the CLI defaults: lr=1e-3, kappa=0.5,
n_osc_per_side=50, v_hidden=32, readout_hidden=32.
"""
import argparse
import json
import random
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from complexPyTorch.complexLayers import ComplexLinear, ComplexReLU

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HopfCell(nn.Module):
    MU = 1.0
    BETA = -1.0

    def __init__(self, units=50, min_freq=8.0, max_freq=13.0, dt=0.01, input_scaler=5.0):
        super().__init__()
        self.units = units
        self.dt = dt
        self.input_scaler = input_scaler
        freqs = torch.linspace(min_freq, max_freq, units)
        self.register_buffer("omega", 2 * np.pi * freqs)

    def init_state(self, batch_size, device):
        r = torch.ones(batch_size, self.units, device=device)
        phi = torch.zeros(batch_size, self.units, device=device)
        return r, phi

    def forward(self, drive, r, phi, phase_coupling=None):
        drive_r = self.input_scaler * drive.real * torch.cos(phi)
        drive_phi = self.input_scaler * drive.imag * torch.sin(phi)
        r_next = r + (self.MU * r + self.BETA * r**3 + drive_r) * self.dt
        coupling = phase_coupling if phase_coupling is not None else 0.0
        phi_next = phi + (self.omega.unsqueeze(0) - drive_phi - coupling) * self.dt
        z = torch.complex(r_next * torch.cos(phi_next), r_next * torch.sin(phi_next))
        return z, r_next, phi_next


class LateralizedHopfModelSmoothed(nn.Module):
    """V section -> mu-band oscillators (Kuramoto-coupled L/R) -> low-freq
    smoothing oscillators (uncoupled) -> concatenated real/imag readout."""

    def __init__(self, n_osc_per_side=50, v_hidden=32, dt=0.005, input_scaler=5.0,
                 min_freq=8.0, max_freq=13.0, kappa=0.5, readout_hidden=32,
                 n_osc2_per_side=50, low_min_freq=0.1, low_max_freq=3.0):
        super().__init__()
        self.n_osc = n_osc_per_side
        self.kappa = kappa

        self.v1_L = ComplexLinear(5, v_hidden)
        self.v1_R = ComplexLinear(5, v_hidden)
        self.v2_own_L = ComplexLinear(v_hidden, v_hidden)
        self.v2_cross_R = ComplexLinear(v_hidden, v_hidden)
        self.v2_own_R = ComplexLinear(v_hidden, v_hidden)
        self.v2_cross_L = ComplexLinear(v_hidden, v_hidden)
        self.v3_L = ComplexLinear(v_hidden, v_hidden)
        self.v3_R = ComplexLinear(v_hidden, v_hidden)
        self.crelu = ComplexReLU()

        self.to_drive_L = ComplexLinear(v_hidden, n_osc_per_side)
        self.to_drive_R = ComplexLinear(v_hidden, n_osc_per_side)

        self.hopf_L = HopfCell(units=n_osc_per_side, min_freq=min_freq, max_freq=max_freq,
                                dt=dt, input_scaler=input_scaler)
        self.hopf_R = HopfCell(units=n_osc_per_side, min_freq=min_freq, max_freq=max_freq,
                                dt=dt, input_scaler=input_scaler)

        # -- smoothing stage: mu-band z -> ComplexLinear+ComplexReLU -> low-freq oscillator --
        self.to_low_L = ComplexLinear(n_osc_per_side, n_osc2_per_side)
        self.to_low_R = ComplexLinear(n_osc_per_side, n_osc2_per_side)
        self.hopf_low_L = HopfCell(units=n_osc2_per_side, min_freq=low_min_freq, max_freq=low_max_freq,
                                    dt=dt, input_scaler=input_scaler)
        self.hopf_low_R = HopfCell(units=n_osc2_per_side, min_freq=low_min_freq, max_freq=low_max_freq,
                                    dt=dt, input_scaler=input_scaler)

        self.readout_L = nn.Sequential(
            nn.Linear(2 * n_osc2_per_side, readout_hidden), nn.LeakyReLU(),
            nn.Linear(readout_hidden, 2),
        )
        self.readout_R = nn.Sequential(
            nn.Linear(2 * n_osc2_per_side, readout_hidden), nn.LeakyReLU(),
            nn.Linear(readout_hidden, 2),
        )

    def init_state(self, batch_size, device):
        r_L, phi_L = self.hopf_L.init_state(batch_size, device)
        r_R, phi_R = self.hopf_R.init_state(batch_size, device)
        r2_L, phi2_L = self.hopf_low_L.init_state(batch_size, device)
        r2_R, phi2_R = self.hopf_low_R.init_state(batch_size, device)
        return (r_L, phi_L, r_R, phi_R, r2_L, phi2_L, r2_R, phi2_R)

    def visual_encode(self, x):
        target = x[:, 4:6]
        moving = x[:, 6:7]
        in_L = torch.cat([x[:, 0:2], target, moving], dim=1)
        in_R = torch.cat([x[:, 2:4], target, moving], dim=1)
        in_L_c = torch.complex(in_L, torch.zeros_like(in_L))
        in_R_c = torch.complex(in_R, torch.zeros_like(in_R))
        h1_L = self.crelu(self.v1_L(in_L_c))
        h1_R = self.crelu(self.v1_R(in_R_c))
        h2_L = self.crelu(self.v2_own_L(h1_L) + self.v2_cross_R(h1_R))
        h2_R = self.crelu(self.v2_own_R(h1_R) + self.v2_cross_L(h1_L))
        h3_L = self.crelu(self.v3_L(h2_L))
        h3_R = self.crelu(self.v3_R(h2_R))
        return h3_L, h3_R

    def forward_step(self, x, state, return_diagnostics=False):
        r_L, phi_L, r_R, phi_R, r2_L, phi2_L, r2_R, phi2_R = state
        enc_L, enc_R = self.visual_encode(x)
        drive_L = self.to_drive_L(enc_L)
        drive_R = self.to_drive_R(enc_R)

        z_L_now = torch.complex(r_L * torch.cos(phi_L), r_L * torch.sin(phi_L))
        z_R_now = torch.complex(r_R * torch.cos(phi_R), r_R * torch.sin(phi_R))
        psi_L = torch.angle(z_L_now.mean(dim=1, keepdim=True))
        psi_R = torch.angle(z_R_now.mean(dim=1, keepdim=True))

        coupling_into_L = self.kappa * torch.sin(psi_R - phi_L)
        coupling_into_R = self.kappa * torch.sin(psi_L - phi_R)

        z_L, r_L_next, phi_L_next = self.hopf_L(drive_L, r_L, phi_L, phase_coupling=coupling_into_L)
        z_R, r_R_next, phi_R_next = self.hopf_R(drive_R, r_R, phi_R, phase_coupling=coupling_into_R)

        # smoothing stage, uncoupled -- purely a low-pass on the mu-band output
        drive2_L = self.crelu(self.to_low_L(z_L))
        drive2_R = self.crelu(self.to_low_R(z_R))
        z2_L, r2_L_next, phi2_L_next = self.hopf_low_L(drive2_L, r2_L, phi2_L)
        z2_R, r2_R_next, phi2_R_next = self.hopf_low_R(drive2_R, r2_R, phi2_R)

        feat_L = torch.cat([z2_L.real, z2_L.imag], dim=1)
        feat_R = torch.cat([z2_R.real, z2_R.imag], dim=1)
        pred = torch.cat([self.readout_L(feat_L), self.readout_R(feat_R)], dim=1)
        new_state = (r_L_next, phi_L_next, r_R_next, phi_R_next, r2_L_next, phi2_L_next, r2_R_next, phi2_R_next)

        if return_diagnostics:
            diag = {"z_L": z_L, "z_R": z_R, "r_L": r_L_next, "r_R": r_R_next, "phi_L": phi_L_next, "phi_R": phi_R_next,
                    "z2_L": z2_L, "z2_R": z2_R, "r2_L": r2_L_next, "r2_R": r2_R_next, "phi2_L": phi2_L_next, "phi2_R": phi2_R_next}
            return pred, new_state, diag
        return pred, new_state


class TrajectoryDataset(Dataset):

    def __init__(self, root_dir, pad_pre=100, pad_post=100):
        root_dir = Path(root_dir)
        self.pad_pre = pad_pre
        self.pad_post = pad_post
        with open(root_dir / "data" / "raw" / "splined_trajectories_100.txt", "rb") as f:
            self.trajectories = pickle.load(f)
        with open(root_dir / "data" / "processed" / "targets_100.pkl", "rb") as f:
            self.targets = pickle.load(f)
        with open(root_dir / "data" / "processed" / "active_arms_100.pkl", "rb") as f:
            self.active_arms = pickle.load(f)
        assert len(self.trajectories) == len(self.targets) == len(self.active_arms)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = np.asarray(self.trajectories[idx], dtype=np.float32)
        target = np.asarray(self.targets[idx], dtype=np.float32)
        active_arm = np.int64(self.active_arms[idx])
        pre = np.repeat(traj[0:1], self.pad_pre, axis=0)
        post = np.repeat(traj[-1:], self.pad_post, axis=0)
        traj_padded = np.concatenate([pre, traj, post], axis=0)
        moving_flag = np.concatenate([
            np.zeros(self.pad_pre, dtype=np.float32),
            np.ones(traj.shape[0], dtype=np.float32),
            np.zeros(self.pad_post, dtype=np.float32),
        ])
        return (torch.from_numpy(traj_padded), torch.from_numpy(target),
                torch.from_numpy(moving_flag), torch.tensor(active_arm))


def teacher_forcing_prob(epoch, epochs, start=1.0, end=0.1):
    frac = (epoch - 1) / max(1, epochs - 1)
    return start + (end - start) * frac


def collect_oscillator_states(traj_batch, target_batch, moving_batch, model, device):
    B, T = traj_batch.shape[0], traj_batch.shape[1]
    state = model.init_state(batch_size=B, device=device)
    prev = traj_batch[:, 0]
    rec = {side: {"z_real": [], "z_imag": [], "r": [], "phi": [],
                   "z2_real": [], "z2_imag": [], "r2": [], "phi2": []} for side in ("L", "R")}
    for t in range(1, T):
        moving_t = moving_batch[:, t:t + 1]
        inp = torch.cat([prev, target_batch, moving_t], dim=1)
        pred, state, diag = model.forward_step(inp, state, return_diagnostics=True)
        for side in ("L", "R"):
            rec[side]["z_real"].append(diag[f"z_{side}"].real.detach().cpu().numpy())
            rec[side]["z_imag"].append(diag[f"z_{side}"].imag.detach().cpu().numpy())
            rec[side]["r"].append(diag[f"r_{side}"].detach().cpu().numpy())
            rec[side]["phi"].append(diag[f"phi_{side}"].detach().cpu().numpy())
            rec[side]["z2_real"].append(diag[f"z2_{side}"].real.detach().cpu().numpy())
            rec[side]["z2_imag"].append(diag[f"z2_{side}"].imag.detach().cpu().numpy())
            rec[side]["r2"].append(diag[f"r2_{side}"].detach().cpu().numpy())
            rec[side]["phi2"].append(diag[f"phi2_{side}"].detach().cpu().numpy())
        prev = pred
    out = {}
    for side in ("L", "R"):
        for k in rec[side]:
            out[f"{k}_{side}"] = np.stack(rec[side][k], axis=0).transpose(1, 0, 2)
    return out


def train(output_dir, epochs=500, batch_size=32, learning_rate=1e-3, seed=42,
          n_osc_per_side=50, v_hidden=32, dt=0.005, input_scaler=5.0,
          freq_min=8.0, freq_max=13.0, kappa=0.5, readout_hidden=32,
          n_osc2_per_side=50, low_min_freq=0.1, low_max_freq=3.0,
          teacher_forcing_start=1.0, teacher_forcing_end=0.1,
          pad_pre=100, pad_post=100, patience=50, save_full_diagnostics=True,
          device=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(PROJECT_ROOT, pad_pre=pad_pre, pad_post=pad_post)
    train_size = int(0.8 * len(dataset)); val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LateralizedHopfModelSmoothed(
        n_osc_per_side=n_osc_per_side, v_hidden=v_hidden, dt=dt, input_scaler=input_scaler,
        min_freq=freq_min, max_freq=freq_max, kappa=kappa, readout_hidden=readout_hidden,
        n_osc2_per_side=n_osc2_per_side, low_min_freq=low_min_freq, low_max_freq=low_max_freq,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "teacher_forcing": []}
    best_val_loss = float("inf"); epochs_since_improve = 0; best_epoch = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        tf_prob = teacher_forcing_prob(epoch, epochs, teacher_forcing_start, teacher_forcing_end)
        model.train(); train_loss_epoch = 0.0

        for traj_batch, target_batch, moving_batch, _ in train_loader:
            traj_batch, target_batch, moving_batch = traj_batch.to(device), target_batch.to(device), moving_batch.to(device)
            B, T = traj_batch.shape[0], traj_batch.shape[1]
            state = model.init_state(batch_size=B, device=device)
            prev = traj_batch[:, 0]; loss = 0.0
            optimizer.zero_grad()
            for t in range(1, T):
                moving_t = moving_batch[:, t:t + 1]
                inp = torch.cat([prev, target_batch, moving_t], dim=1)
                pred, state = model.forward_step(inp, state)
                gt = traj_batch[:, t]
                loss += loss_fn(pred, gt)
                prev = gt if random.random() < tf_prob else pred.detach()
            loss = loss / (T - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader)

        model.eval(); val_loss_epoch = 0.0
        with torch.no_grad():
            for traj_batch, target_batch, moving_batch, _ in val_loader:
                traj_batch, target_batch, moving_batch = traj_batch.to(device), target_batch.to(device), moving_batch.to(device)
                B, T = traj_batch.shape[0], traj_batch.shape[1]
                state = model.init_state(batch_size=B, device=device)
                prev = traj_batch[:, 0]; loss = 0.0
                for t in range(1, T):
                    moving_t = moving_batch[:, t:t + 1]
                    inp = torch.cat([prev, target_batch, moving_t], dim=1)
                    pred, state = model.forward_step(inp, state)
                    gt = traj_batch[:, t]
                    loss += loss_fn(pred, gt)
                    prev = pred
                loss = loss / (T - 1)
                val_loss_epoch += loss.item()
        val_loss_epoch /= len(val_loader)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["teacher_forcing"].append(tf_prob)

        improved = val_loss_epoch < best_val_loss
        if improved:
            best_val_loss = val_loss_epoch; best_epoch = epoch; epochs_since_improve = 0
            torch.save({"epoch": epoch, "val_loss": val_loss_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       output_dir / "best_hopf_model.pt")
            if save_full_diagnostics:
                osc_states = {"traj_gt": [], "target": [], "moving_flag": []}
                for side in ("L", "R"):
                    for k in ("z_real", "z_imag", "r", "phi", "z2_real", "z2_imag", "r2", "phi2"):
                        osc_states[f"{k}_{side}"] = []
                with torch.no_grad():
                    for traj_batch, target_batch, moving_batch, _ in val_loader:
                        traj_batch, target_batch, moving_batch = traj_batch.to(device), target_batch.to(device), moving_batch.to(device)
                        batch_states = collect_oscillator_states(traj_batch, target_batch, moving_batch, model, device)
                        for b in range(traj_batch.shape[0]):
                            for key, arr in batch_states.items():
                                osc_states[key].append(arr[b])
                            osc_states["traj_gt"].append(traj_batch[b].cpu().numpy())
                            osc_states["target"].append(target_batch[b].cpu().numpy())
                            osc_states["moving_flag"].append(moving_batch[b, 1:].cpu().numpy())
                np.savez(output_dir / "best_oscillator_states.npz",
                         **{k: np.array(v, dtype=object) for k, v in osc_states.items()})
        else:
            epochs_since_improve += 1

        print(f"[smoothed] Epoch {epoch:04d} | Train {train_loss_epoch:.6f} | Val {val_loss_epoch:.6f} | "
              f"TF {tf_prob:.4f} | best {best_val_loss:.6f}@{best_epoch}", flush=True)

        if epochs_since_improve >= patience:
            print(f"[smoothed] Early stop at epoch {epoch} (no improvement in {patience} epochs)", flush=True)
            break

    elapsed = time.time() - t_start
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"best_val_loss": best_val_loss, "best_epoch": best_epoch, "epochs_run": epoch,
                    "elapsed_sec": elapsed,
                    "config": {"learning_rate": learning_rate, "batch_size": batch_size, "kappa": kappa,
                               "n_osc_per_side": n_osc_per_side, "v_hidden": v_hidden,
                               "readout_hidden": readout_hidden, "teacher_forcing_end": teacher_forcing_end,
                               "seed": seed, "pad_pre": pad_pre, "pad_post": pad_post,
                               "n_osc2_per_side": n_osc2_per_side, "low_min_freq": low_min_freq,
                               "low_max_freq": low_max_freq}}, f, indent=2)
    print(f"[smoothed] DONE best_val_loss={best_val_loss:.6f} @epoch {best_epoch} ({elapsed:.1f}s, {epoch} epochs)", flush=True)
    return best_val_loss


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_osc_per_side", type=int, default=50)
    p.add_argument("--v_hidden", type=int, default=32)
    p.add_argument("--readout_hidden", type=int, default=32)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--n_osc2_per_side", type=int, default=50)
    p.add_argument("--low_min_freq", type=float, default=0.1)
    p.add_argument("--low_max_freq", type=float, default=3.0)
    p.add_argument("--pad_pre", type=int, default=100)
    p.add_argument("--pad_post", type=int, default=100)
    p.add_argument("--patience", type=int, default=50)
    args = p.parse_args()

    train(output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.learning_rate, seed=args.seed, n_osc_per_side=args.n_osc_per_side,
          v_hidden=args.v_hidden, readout_hidden=args.readout_hidden, kappa=args.kappa,
          n_osc2_per_side=args.n_osc2_per_side, low_min_freq=args.low_min_freq, low_max_freq=args.low_max_freq,
          pad_pre=args.pad_pre, pad_post=args.pad_post, patience=args.patience)
