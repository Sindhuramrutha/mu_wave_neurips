# Mu-Wave Desynchronization — Bimanual Reaching Model

A complex-weighted, dual-population Hopf-oscillator network that predicts bimanual
reaching trajectories and is used to study mu-band (8–13Hz) synchrony at rest vs.
desynchrony during movement — plus lateral (inter-hemisphere) coupling ablations and
an added low-frequency smoothing stage.

This repo contains only the final, working pipeline — data generation, the trained
model architecture, the hyperparameter sweep, three trained model variants, and the
synchrony/desynchrony analysis. Earlier exploratory variants (jerk-penalty, amplitude-gate,
recurrent-readout, etc.) are intentionally not included.

## Repo layout

```
data/
  raw/splined_trajectories_100.txt      498 spline-smoothed bimanual reach trajectories, (100, 4) = [x_L,y_L,x_R,y_R]
  processed/targets_100.pkl             2D reach target per trial
  processed/active_arms_100.pkl         which arm (0=left, 1=right) reached in each trial

notebooks/
  01_data_pipeline.ipynb                generates the trajectories above from 2-link-arm kinematics
  02_model_and_training_v4.ipynb        the model + training loop, as a readable notebook
  sweep_train.py                        standalone CLI version of the same model/training, used for the sweep
  train_smoothed.py                     model + training with an added low-frequency (0.1-3Hz) smoothing oscillator stage
  run_sweep.py                          orchestrates the 16-config hyperparameter sweep across GPUs
  07_v4_final_sweep_and_results.ipynb   network code, sweep leaderboard, final-run loss curve, prediction plots
  08_synchrony_kuramoto_hebbian.ipynb   Kuramoto order parameter + Hebbian synchrony detector + ERD/ERS + lateralization, on the main model
  09_synchrony_smoothed_model.ipynb     same battery, on the smoothed-architecture model
  10_synchrony_kappa0.ipynb             same battery, on the no-coupling (kappa=0) ablation
  11_predictions_kappa0.ipynb           rest->motion->rest prediction check, kappa=0 model
  12_predictions_smoothed.ipynb         rest->motion->rest prediction check, smoothed model

outputs_v4_final/       trained weights + results for the main model (winning sweep config, kappa=0.5)
outputs_v4_kappa0/      trained weights + results for the no-coupling ablation (kappa=0)
outputs_v4_smoothed/    trained weights + results for the smoothed-architecture model
outputs_v4_sweep/       all 16 sweep configs: weights, per-config metrics, and the leaderboard
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Everything assumes it's run from inside `notebooks/` — every script/notebook resolves
its own paths as `Path.cwd().parent`, so as long as the folder layout above is intact,
nothing needs to be edited.

## The model

`LateralizedHopfModelVM` (in `sweep_train.py` / `02_model_and_training_v4.ipynb`):

- **V section**: per-side (left/right arm) complex-weighted encoder (`ComplexLinear` +
  `ComplexReLU`, genuine complex weights `W_r + iW_i`), three layers deep, with exactly
  one learned crossing point mixing the two sides' hidden representations ("mix, not
  swap").
- **M section**: the V output projects (via one more `ComplexLinear`) into a complex
  drive for a bank of 50 Hopf oscillators per side, fixed in the mu band (8–13Hz,
  not learned). The two sides' populations are linked by a Kuramoto phase-coupling
  term, strength `kappa`.
- **Readout**: `[z.real, z.imag]` concatenated (not magnitude-only) into a real
  `Linear -> LeakyReLU -> Linear` per side, predicting the next `(x, y)` position.
- **Input**: `[own arm position (2), target (2), moving_flag (1)]` per side.
  `moving_flag` is a ground-truth 0/1 signal — each trajectory is padded with the
  hand held still at its own start/end position (rest), around the real 100-step
  reach (motion), so the model is explicitly told whether it's expected to hold
  still or move.
- **Training**: scheduled sampling (teacher forcing annealed 1.0→0.1), gradient
  clipping, early stopping on validation loss (which is always fully autoregressive
  — never teacher-forced — so it's an honest measure of real deployment behavior).

`train_smoothed.py`'s `LateralizedHopfModelSmoothed` adds a second
`ComplexLinear+ComplexReLU` projection and a second Hopf oscillator stage tuned to
0.1–3Hz after the mu-band stage (uncoupled), with the readout consuming *that*
stage's output instead — a low-pass on the mu-band carrier before it reaches the
position prediction.

## The three trained models

| | `outputs_v4_final` | `outputs_v4_kappa0` | `outputs_v4_smoothed` |
|---|---|---|---|
| Architecture | `LateralizedHopfModelVM` | `LateralizedHopfModelVM` | `LateralizedHopfModelSmoothed` |
| kappa (coupling) | 0.5 | **0.0** | 0.5 |
| n_osc_per_side | 50 | 50 | 50 (+ 50 low-freq) |
| v_hidden | 32 | 32 | 32 |
| pad_pre / pad_post | 20 / 20 | 20 / 20 | **100 / 100** |
| Best val loss | 0.0000118 (epoch 236) | 0.0000208 (epoch 157) | 0.0000189 (epoch 96) |
| Full config | `metrics.json` | `metrics.json` | not saved (stopped by hand) — see config below |

`outputs_v4_smoothed`'s config (training was stopped manually at epoch 96 rather
than via early-stopping, so no `metrics.json` was written):
```python
dict(learning_rate=1e-3, batch_size=32, kappa=0.5, n_osc_per_side=50, v_hidden=32,
     readout_hidden=32, n_osc2_per_side=50, low_min_freq=0.1, low_max_freq=3.0,
     pad_pre=100, pad_post=100, seed=42)
```

All three models were selected from a **16-config hyperparameter sweep**
(`run_sweep.py` + `sweep_train.py`, grid: `learning_rate` x `kappa` x
`n_osc_per_side` x `v_hidden`) — the full leaderboard is in
`outputs_v4_sweep/leaderboard.json`, and every config's own checkpoint + training
history is saved alongside it.

### Loading a trained model

```python
import torch, json
from sweep_train import LateralizedHopfModelVM   # or LateralizedHopfModelSmoothed from train_smoothed.py

with open("outputs_v4_final/metrics.json") as f:
    cfg = json.load(f)["config"]

model = LateralizedHopfModelVM(
    n_osc_per_side=cfg["n_osc_per_side"], v_hidden=cfg["v_hidden"],
    kappa=cfg["kappa"], readout_hidden=cfg["readout_hidden"],
)
ckpt = torch.load("outputs_v4_final/best_hopf_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Regenerating the full oscillator-state diagnostics

Each `outputs_v4_*/` folder holds the trained checkpoint and small artifacts
(metrics, loss curves, prediction plots) but **not** the full per-timestep
oscillator-state dump (`best_oscillator_states.npz`) that the synchrony notebooks
were built from — those files are 49–207MB each, too large for this repo. They're
fully reproducible from the checkpoint without retraining: `sweep_train.py` and
`train_smoothed.py` both define `collect_oscillator_states()`, the exact function
used to produce them. Load the checkpoint as above, then run a validation-set
rollout through that function (see the top of any `08`-`10`/`12` notebook for the
dataset/split setup) to regenerate the `.npz` used by the corresponding analysis
notebook.

## Key findings (see the analysis notebooks for full statistics)

- **Mu-band phase desynchronizes sharply during movement**: Kuramoto order
  parameter drops ~65-70% (rest→motion) within each hemisphere's oscillator
  population — confirmed independently by both the Kuramoto measure and an
  adapted Hebbian synchrony detector, which agree closely.
- **This is a phase effect, not an amplitude effect**: raw mu-band power (ERD/ERS)
  only changes ~3-9%, an order of magnitude smaller than the phase-coherence
  effect — this model's desynchronization signature lives in phase coordination,
  not signal amplitude.
- **Lateralization is real but asymmetric and coupling-dependent**: splitting
  trials by which arm was reaching shows the active hemisphere desynchronizes
  more than the idle one — strongly for left-arm reaches, weakly for right-arm
  reaches in the main (kappa=0.5) model. Removing coupling (kappa=0) sharpens the
  weaker (right-arm) lateralization ~4x, suggesting the coupling term partly masks
  hemisphere-specific dynamics on whichever side it's weaker to begin with.
- **Coupling trades off accuracy against lateralization clarity**: kappa=0.5 gives
  ~1.8x better trajectory-prediction accuracy than kappa=0, but produces a less
  clean per-hemisphere signature — there's no single kappa that's strictly best for
  both objectives at once.
- **The smoothed-architecture model shows a reversed lateralization direction**
  (active side *more* synchronized than idle, not less) — real and significant,
  but confounded with that run also using 5x longer rest windows, so which change
  is responsible isn't yet isolated.
