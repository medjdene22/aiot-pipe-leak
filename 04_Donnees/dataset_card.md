# Dataset card - Hanoi leak scenarios (04_Donnees)

## Source and generation

| Item | Value |
|---|---|
| Hydraulic network | **Hanoi benchmark** (31 demand nodes + 1 reservoir, 34 pipes; Fujiwara & Kang 1998 design problem) |
| Network file | `Hanoi.inp`, committed copy of the EPANET example network distributed with **OpenWaterAnalytics/EPyT** (`epyt/networks/asce-tf-wdst/Hanoi.inp`); byte-identical to the EPyT-Flow mirror `filedn.com/.../EPyT-Flow/Networks/Hanoi.inp`. EPANET example networks are US-EPA public domain; Hanoi originates from the Fujiwara & Kang (1998) benchmark and is freely redistributed by the EPANET/OWA projects. |
| Simulator | WNTR 1.5.0 `WNTRSimulator` (pressure-dependent demands and leaks) |
| Generator | `generate_data.py` (this folder), `SEED = 42` everywhere |
| Samples | **1,320 scenarios** x 31 nodes x 48 windows = **1,964,160 node-windows** + 63,360 network residual windows (> 1,000 required) |
| License note | Generated data are synthetic; no personal data. |

## Scenario design

- 24 h extended-period runs, 5-min hydraulic/report step (289 points), hourly
  diurnal demand pattern (documented in `generate_data.py`).
- 40 random demand patterns: diurnal curve x uniform scale [0.85, 1.15] x
  per-node lognormal jitter (sigma = 0.05).
- **1,240 leak scenarios** = 40 patterns x 31 candidate leak nodes
  (= 40 distinct leak scenarios per candidate node, brief Section 5).
- **80 leak-free scenarios** (healthy behaviour + sensor noise).
- **40 nominal runs** (leak-free, noise-free) = model-based baseline for the
  pressure residuals and the Edge `dp` feature.

## Leak model (physical orifice)

WNTR `Junction.add_leak`: **Q = Cd * A * sqrt(2 g h)** with Cd = 0.75,
h = gauge head at the junction. The leak area A is calibrated per scenario so
the resulting leak flow equals **1-10% of the network's mean total demand**
(target fraction ~ U[1%, 10%]); the realized leak flows are recorded per node
and window (column `leak_flow_node`, L/s; observed range ~1-534 L/s, mean ~145).
Leak start time ~ U[2 h, 20 h] (scheduled via WNTR leak controls).

## Sensors (3 per node)

| Signal | Unit | Source | Noise |
|---|---|---|---|
| pressure `p` | m | WNTR node pressure | Gaussian sigma = **0.15 m** |
| flow `q` | L/s | mean of abs flow of the pipes adjacent to the node (WNTR link flowrate) | sigma = 1% of q + 0.10 L/s |
| vibration `vib` | g RMS | **SYNTHETIC**: `vib = K*sqrt(leak_flow_Lps) * att + N(0, sigma_v)` | sigma_v = **0.005 g** |

- K = 0.02 g/sqrt(L/s); attenuation `att` = 1.0 at the leak node, 0.3 at
  1-hop neighbouring nodes, 0 elsewhere (graph distance on the pipe graph).
- **The vibration channel does not exist in WNTR** - it is a declared synthetic
  channel correlated with the leak flow (report "Limits of the simulation" box).

## Windows and labels

- Window = 6 steps (30 min), 48 windows per scenario per node.
- `y_global` = 1 if any leak is active during the window (primary label for the
  Edge detector; any-overlap semantics). Balance: 1,044,514 / 1,964,160 (53%).
- `y_local` = 1 if the leak is at **this node** during the window (analysis).
  33,694 positives.
- Residuals: `r = p_measured(noisy) - p_nominal` (paired nominal run with the
  same demand pattern, no leak, no noise).

## Splits (no leakage)

60/20/20 by **scenario id** (`splits.json`; 792/264/264 scenarios), stratified
by scenario kind and by leak node (every leak node occurs in every split).
`generate_data.py` and `ia/common.py` both **assert** that no scenario id
appears in two splits. Scalers/thresholds are fitted on train only.

## Files

| File | Rows | Content |
|---|---|---|
| `scenarios.csv` | 1,320 | manifest: scenario_id, kind, pattern_id, leak_node, leak_frac, start_s |
| `windows.csv.gz` | 1,964,160 | node-windows: features (p, dp, q, vib_rms, + std/max), labels, leak flow |
| `residuals.csv.gz` | 63,360 | network-windows: 31 residual columns + true leak node |
| `nominal_profiles.csv.gz` | 355,360 | nominal pressure per demand pattern (fog/edge baseline) |
| `splits.json` | - | train/val/test scenario ids |
| `sample_windows.csv`, `sample_residuals.csv` | 2,000 each | small samples shipped in the ZIP |
| `generation_meta.json` | - | parameters + row counts of the run |

The full dataset is ~150 MB compressed and is **not** shipped in the ZIP
(20 MB limit); it is reproduced bit-for-bit by running
`python generate_data.py` (SEED=42, ~3 min on 3 cores).

## Class balance (node-windows)

| Label | Positives | Share |
|---|---|---|
| y_global (leak active) | 1,044,514 | 53.2% |
| y_local (leak at this node) | 33,694 | 1.7% |

## Variables (`windows.csv.gz`)

`scenario_id, pattern_id, node, w_idx, t_start_step` (keys);
`p_mean` [m], `dp_mean` [m] (residual vs nominal), `q_mean` [L/s],
`vib_rms` [g], `p_std`, `q_std`, `vib_max` (diagnostics);
`y_global`, `y_local` (labels); `leak_flow_node` [L/s] (leak magnitude at the
node during the window; 0 if no leak).
