#!/usr/bin/env python3
"""
Hydraulic scenario generation for the Hanoi benchmark network (water-network
leak detection and localization).

Course: UMBB Boumerdes - M2 I2A - Embedded Systems and IoT, 2026-2027.
(Agent brief Section 5, data pipeline.)

WHAT THIS SCRIPT DOES
---------------------
1. Loads the Hanoi benchmark network (04_Donnees/Hanoi.inp - committed copy of the
   EPANET example network distributed with OpenWaterAnalytics/EPyT).
2. Generates 1,320 labelled 24 h / 5-min extended-period scenarios with WNTR's
   own simulator (WNTRSimulator - supports pressure-dependent leaks):
     - 40 random demand patterns (diurnal curve x per-scenario scale x per-node jitter)
     - 1,240 leak scenarios: for every pattern x every junction (31) ONE leak with
       a random magnitude and start time  ->  40 scenarios per candidate node
     -    80 leak-free scenarios (healthy behaviour + sensor noise)
     -    40 paired nominal runs (noise-free, leak-free, one per demand pattern)
       used as the model-based baseline for pressure residuals and Edge dp.
3. Leak model: physical orifice at the junction (WNTR Junction.add_leak):
       Q = Cd * A * sqrt(2*g*h)      (Cd = 0.75, h = gauge head)
   The leak area A is calibrated per scenario so the leak flow equals
   1-10% of the network's mean total demand (brief Section 5.1).
4. Sensors (E3): pressure p (m), flow q (L/s - mean of |flow| of the pipes
   adjacent to the node), vibration RMS vib (g). The vibration channel is
   SYNTHETIC and correlated with the leak flow near the leak:
       vib = K*sqrt(leak_flow_Lps) * attenuation(graph distance) + N(0, sigma_v)
   attenuation = 1.0 at the leak node, 0.3 at 1-hop neighbours, 0 elsewhere.
   K = 0.02, sigma_v = 0.005 g.
5. Sensor noise: Gaussian on pressure (sigma_p = 0.15 m) and flow
   (sigma_q = 1% of |q| + 0.10 L/s), added after the simulation; per-scenario
   independent draws derived from (SEED, scenario_id) - parallel-safe.
6. Windowing: 6 steps (30 min) per node. Labels:
     y_global = 1 if any leak is active in the network during the window
                (primary label for the Edge detector, brief Section 5.3)
     y_local  = 1 if a leak is active AT THIS NODE during the window (analysis)
7. Residuals: r = p_measured(noisy) - p_nominal(same demand pattern, no leak,
   no noise). Stored per node and per network window (fog localization input).
8. Split 60/20/20 BY SCENARIO (assert: no scenario id in two splits),
   stratified by scenario kind and by leak node. Splits saved to splits.json.

OUTPUTS (in --outdir, default 04_Donnees/generated/)
  scenarios.csv            scenario manifest (one row per scenario)
  windows.csv.gz           node-level windows (features + labels)   ~2M rows
  residuals.csv.gz         network-level residual vectors + leak node
  nominal_profiles.csv.gz  nominal pressure per demand pattern
  splits.json              scenario-id splits (train/val/test)
  sample_windows.csv       2,000-row sample (fits in the ZIP)
  sample_residuals.csv     2,000-row sample (fits in the ZIP)
  generation_meta.json     parameters + row counts

REPRODUCIBILITY: SEED=42 everywhere. See dataset_card.md.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wntr

SEED = 42
CD = 0.75                      # leak discharge coefficient
G = 9.81
WINDOW = 6                     # steps per window (6 x 5 min = 30 min)
REPORT_STEP = 300              # s
DURATION = 24 * 3600           # s
N_STEPS = DURATION // REPORT_STEP + 1     # 289 points (incl t=0)
N_PATTERNS = 40
N_LEAK_FREE = 80               # leak-free scenarios (2 per pattern)
LEAK_FRAC_RANGE = (0.01, 0.10)  # leak flow / mean total demand
START_RANGE = (2 * 3600, 20 * 3600)
SIGMA_P = 0.15                 # pressure noise std [m]
SIGMA_Q_FLOOR = 0.10           # flow noise floor [L/s]
SIGMA_Q_REL = 0.01             # flow noise relative to |q|
VIB_K = 0.02                   # vib = K * sqrt(leak_flow_Lps) * att
SIGMA_V = 0.005                # vibration noise std [g]
ATT_NEIGHBOR = 0.3

DIURNAL = np.array([0.55, 0.45, 0.42, 0.42, 0.50, 0.70, 0.95, 1.15, 1.25, 1.20,
                    1.15, 1.12, 1.10, 1.08, 1.10, 1.15, 1.22, 1.30, 1.28, 1.18,
                    1.00, 0.85, 0.70, 0.60])

# Globals initialised in the parent, inherited by forked workers
JUNCTIONS = None
PIPE_ENDS = None
ADJ = None
PIPE_FLOWS = None
NOMINAL = None


def build_wn(inp_path, pattern_scale, node_jitter, leak=None):
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = DURATION
    wn.options.time.report_timestep = REPORT_STEP
    wn.options.time.hydraulic_timestep = REPORT_STEP
    wn.options.time.pattern_timestep = 3600
    wn.add_pattern("diurnal", DIURNAL * pattern_scale)
    for _, j in wn.junctions():
        j.demand_pattern_name = "diurnal"
        if len(j.demand_timeseries_list) > 0:
            dts = j.demand_timeseries_list[0]
            dts.base_value = dts.base_value * node_jitter[j.name]
    if leak is not None:
        node_id, area, start_s = leak
        wn.get_node(node_id).add_leak(wn, area=area, discharge_coeff=CD,
                                      start_time=start_s, end_time=None)
    return wn


def node_flow(q_link):
    """mean |flow| over the pipes adjacent to each node, (steps, nodes) [L/s]"""
    q = np.zeros((q_link.shape[0], len(JUNCTIONS)))
    for i, n in enumerate(JUNCTIONS):
        cols = [PIPE_FLOWS[pl] for pl in ADJ[n]]
        q[:, i] = np.abs(q_link[:, cols]).mean(axis=1) * 1000.0
    return q


def make_manifest():
    rng = np.random.default_rng(SEED)
    patterns = []
    for pid in range(N_PATTERNS):
        scale = rng.uniform(0.85, 1.15)
        jitter = rng.lognormal(mean=0.0, sigma=0.05, size=len(JUNCTIONS))
        patterns.append(dict(pattern_id=pid, scale=scale,
                             jitter=dict(zip(JUNCTIONS, jitter))))
    scenarios = []
    sid = 0
    for pat in patterns:
        for node in JUNCTIONS:
            scenarios.append(dict(
                scenario_id=sid, kind="leak", pattern_id=pat["pattern_id"],
                leak_node=node, leak_frac=rng.uniform(*LEAK_FRAC_RANGE),
                start_s=int(rng.integers(*START_RANGE)),
                pattern_scale=round(pat["scale"], 6)))
            sid += 1
    for pat in patterns:
        for _ in range(N_LEAK_FREE // N_PATTERNS):
            scenarios.append(dict(
                scenario_id=sid, kind="free", pattern_id=pat["pattern_id"],
                leak_node="", leak_frac=0.0, start_s=-1,
                pattern_scale=round(pat["scale"], 6)))
            sid += 1
    return patterns, scenarios


def run_scenario(args):
    """Worker: one scenario -> node windows + network residual windows."""
    sc, patterns, inp_path = args
    pat = patterns[sc["pattern_id"]]
    rng = np.random.default_rng([SEED, sc["scenario_id"]])
    p_nom = NOMINAL[sc["pattern_id"]]["p_nom"]           # (289, 31)
    q_node = NOMINAL[sc["pattern_id"]]["q_nom"]          # (289, 31) L/s
    h_mean = NOMINAL[sc["pattern_id"]]["h_mean"]         # (31,)
    total_mean = NOMINAL[sc["pattern_id"]]["total_mean"]  # m3/s

    leak_dem = np.zeros((N_STEPS, len(JUNCTIONS)))
    vib = np.zeros_like(p_nom)
    if sc["kind"] == "leak":
        li = JUNCTIONS.index(sc["leak_node"])
        target_q = sc["leak_frac"] * total_mean          # m3/s
        h = max(float(h_mean[li]), 0.5)
        area = target_q / (CD * np.sqrt(2 * G * h))
        assert area > 0 and np.isfinite(area), f"bad leak area {area} (target_q={target_q})"
        wn = build_wn(inp_path, pat["scale"], pat["jitter"],
                      leak=(sc["leak_node"], area, sc["start_s"]))
        res = wntr.sim.WNTRSimulator(wn).run_sim()
        p = res.node["pressure"][JUNCTIONS].to_numpy()
        q_link = res.link["flowrate"].to_numpy()
        q_node = node_flow(q_link)
        leak_dem = res.node["leak_demand"][JUNCTIONS].to_numpy()
    else:
        # leak-free: hydraulics identical to the paired nominal run
        p = p_nom.copy()

    if sc["kind"] == "leak":
        li = JUNCTIONS.index(sc["leak_node"])
        lf = np.maximum(leak_dem[:, li] * 1000.0, 0.0)    # L/s
        vib[:, li] += VIB_K * np.sqrt(lf)
        for pl in ADJ[sc["leak_node"]]:
            for nb in PIPE_ENDS[pl]:
                if nb in JUNCTIONS and nb != sc["leak_node"]:
                    vib[:, JUNCTIONS.index(nb)] += ATT_NEIGHBOR * VIB_K * np.sqrt(lf)

    # sensor noise on measured signals
    p_meas = p + rng.normal(0, SIGMA_P, p.shape)
    q_meas = q_node + rng.normal(0, 1.0, q_node.shape) * (SIGMA_Q_REL * q_node + SIGMA_Q_FLOOR)
    vib_meas = vib + rng.normal(0, SIGMA_V, vib.shape)
    dp = p_meas - p_nom

    n_win = (N_STEPS - 1) // WINDOW
    win_rows, net_rows = [], []
    for w in range(n_win):
        s0, s1 = w * WINDOW, (w + 1) * WINDOW
        # label rule (brief Section 5.3): a window is positive if the leak is
        # active at any moment inside it (any-overlap semantics)
        leak_active = (sc["kind"] == "leak") and (s1 * REPORT_STEP > sc["start_s"])
        rvec = []
        for i, n in enumerate(JUNCTIONS):
            p_w, q_w, v_w, d_w = (p_meas[s0:s1, i], q_meas[s0:s1, i],
                                  vib_meas[s0:s1, i], dp[s0:s1, i])
            win_rows.append((
                sc["scenario_id"], sc["pattern_id"], n, w, s0,
                float(p_w.mean()), float(d_w.mean()), float(q_w.mean()),
                float(np.sqrt((v_w ** 2).mean())),
                float(p_w.std()), float(q_w.std()), float(v_w.max()),
                1 if leak_active else 0,
                1 if (leak_active and n == sc["leak_node"]) else 0,
                float(leak_dem[s0:s1, i].mean() * 1000.0)))
            rvec.append(float(d_w.mean()))
        net_rows.append((
            sc["scenario_id"], sc["pattern_id"], w, s0,
            1 if leak_active else 0,
            sc["leak_node"] if leak_active else "",
            sc["leak_frac"] * 100.0 if sc["kind"] == "leak" else 0.0,
            *rvec))
    return win_rows, net_rows


def init_worker(inp_path, nominal=None):
    global JUNCTIONS, PIPE_ENDS, ADJ, PIPE_FLOWS, NOMINAL
    if nominal is not None:
        NOMINAL = nominal
    wn = wntr.network.WaterNetworkModel(inp_path)
    JUNCTIONS = sorted([v.name for _, v in wn.junctions()], key=int)
    PIPE_ENDS = {p.name: (p.start_node_name, p.end_node_name) for _, p in wn.pipes()}
    ADJ = {n: [] for n in JUNCTIONS}
    for pl, (a, b) in PIPE_ENDS.items():
        if a in ADJ:
            ADJ[a].append(pl)
        if b in ADJ:
            ADJ[b].append(pl)
    PIPE_FLOWS = {p.name: i for i, (_, p) in enumerate(wn.pipes())}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--inp", default=os.path.join(here, "Hanoi.inp"))
    ap.add_argument("--outdir", default=os.path.join(here, "generated"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="run only N scenarios (debug)")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    init_worker(args.inp)

    patterns, scenarios = make_manifest()
    if args.limit:
        scenarios = scenarios[: args.limit]
    pd.DataFrame(scenarios).to_csv(os.path.join(args.outdir, "scenarios.csv"), index=False)

    # ---- one nominal run per demand pattern (baseline + area calibration) ----
    nominal = {}
    nom_rows = []
    for pat in patterns:
        wn = build_wn(args.inp, pat["scale"], pat["jitter"], leak=None)
        res = wntr.sim.WNTRSimulator(wn).run_sim()
        p_nom = res.node["pressure"][JUNCTIONS].to_numpy()
        q_nom = node_flow(res.link["flowrate"].to_numpy())
        # NOTE: res.node['demand'] contains the reservoir's negative balance;
        # total consumer demand = sum over junctions only
        dem = res.node["demand"][JUNCTIONS].sum(axis=1).mean()  # m3/s
        nominal[pat["pattern_id"]] = dict(
            p_nom=p_nom, q_nom=q_nom, h_mean=p_nom.mean(axis=0),
            total_mean=float(dem))
        for s in range(N_STEPS):
            for i, n in enumerate(JUNCTIONS):
                nom_rows.append((pat["pattern_id"], s, n, float(p_nom[s, i])))
    pd.DataFrame(nom_rows, columns=["pattern_id", "step", "node", "p_nom"]).to_csv(
        os.path.join(args.outdir, "nominal_profiles.csv.gz"),
        index=False, compression="gzip")
    print(f"[{time.time()-t0:.0f}s] nominal runs done ({len(patterns)} patterns)", flush=True)

    work = [(sc, patterns, args.inp) for sc in scenarios]
    win_rows, net_rows = [], []
    with Pool(args.workers, initializer=init_worker,
              initargs=(args.inp, nominal)) as pool:
        for i, (w_, n_) in enumerate(pool.imap_unordered(run_scenario, work, chunksize=4)):
            win_rows.extend(w_)
            net_rows.extend(n_)
            if (i + 1) % 50 == 0:
                print(f"[{time.time()-t0:.0f}s] {i+1}/{len(scenarios)} scenarios", flush=True)

    win_cols = ["scenario_id", "pattern_id", "node", "w_idx", "t_start_step",
                "p_mean", "dp_mean", "q_mean", "vib_rms", "p_std", "q_std",
                "vib_max", "y_global", "y_local", "leak_flow_node"]
    net_cols = ["scenario_id", "pattern_id", "w_idx", "t_start_step",
                "y_global", "leak_node", "leak_frac_pct"] + [f"r_{n}" for n in JUNCTIONS]
    dfw = pd.DataFrame(win_rows, columns=win_cols)
    dfn = pd.DataFrame(net_rows, columns=net_cols)
    dfw.to_csv(os.path.join(args.outdir, "windows.csv.gz"), index=False, compression="gzip")
    dfn.to_csv(os.path.join(args.outdir, "residuals.csv.gz"), index=False, compression="gzip")

    # ---- split by scenario id (assert no leakage), stratified by kind + leak node ----
    rng = np.random.default_rng(SEED)
    kind_of = {s["scenario_id"]: s["kind"] for s in scenarios}
    node_of = {s["scenario_id"]: s["leak_node"] for s in scenarios}
    ids = np.array(sorted(kind_of))
    train, val, test = [], [], []
    for kind in ("leak", "free"):
        g = np.array([i for i in ids if kind_of[i] == kind])
        rng.shuffle(g)
        n = len(g)
        train += g[: int(0.6 * n)].tolist()
        val += g[int(0.6 * n): int(0.8 * n)].tolist()
        test += g[int(0.8 * n):].tolist()
    leak_ids = [i for i in ids if kind_of[i] == "leak"]
    for node in JUNCTIONS:      # every leak node present in every split
        for split in (train, val, test):
            if not any(node_of[i] == node for i in split):
                donor = [i for i in leak_ids if node_of[i] == node and i in train]
                if donor:
                    train.remove(donor[0])
                    split.append(donor[0])
    assert not (set(train) & set(val)), "LEAKAGE: scenario id in train AND val"
    assert not (set(train) & set(test)), "LEAKAGE: scenario id in train AND test"
    assert not (set(val) & set(test)), "LEAKAGE: scenario id in val AND test"
    with open(os.path.join(args.outdir, "splits.json"), "w") as f:
        json.dump({"seed": SEED, "train": sorted(int(i) for i in train),
                   "val": sorted(int(i) for i in val),
                   "test": sorted(int(i) for i in test)}, f, indent=1)

    dfw.sample(min(2000, len(dfw)), random_state=SEED).to_csv(
        os.path.join(args.outdir, "sample_windows.csv"), index=False)
    dfn.sample(min(2000, len(dfn)), random_state=SEED).to_csv(
        os.path.join(args.outdir, "sample_residuals.csv"), index=False)

    meta = dict(seed=SEED, n_scenarios=len(scenarios),
                n_leak=sum(1 for s in scenarios if s["kind"] == "leak"),
                n_free=sum(1 for s in scenarios if s["kind"] == "free"),
                n_windows=len(dfw), n_net_windows=len(dfn),
                y_global_pos=int(dfw["y_global"].sum()),
                y_local_pos=int(dfw["y_local"].sum()),
                sigma_p=SIGMA_P, sigma_q_floor=SIGMA_Q_FLOOR,
                sigma_q_rel=SIGMA_Q_REL, vib_k=VIB_K, sigma_v=SIGMA_V,
                leak_frac_range=list(LEAK_FRAC_RANGE), window=WINDOW,
                runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(args.outdir, "generation_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"DONE in {time.time()-t0:.0f}s -> {args.outdir}")


if __name__ == "__main__":
    main()
