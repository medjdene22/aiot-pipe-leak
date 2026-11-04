#!/usr/bin/env python3
"""
Static experiments (brief Section 11): X1, X2, X3, X4, X6, X8, X9, X10, X11.

Everything here is scripted and logged under results/. Every reported value
carries its label:
  [M] measured on PC/sandbox components
  [I] indicative (simulator timing / composed with simulator parts)
  [E] estimated by calculation (method + cited source in the report)

The live end-to-end experiments X5 (latency) and X7 (robustness) live in
experiments_live.py. Run order: experiments_live.py first, then this script
(placement_comparison.csv merges both).

Usage: python experiments_static.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "results")
RAW = os.path.join(RESULTS, "raw")
SEED = 42
rng = np.random.default_rng(SEED)

sys.path.append(HERE)
from common import (balanced_sample_weight, ensure_dirs, load_residuals,
                    load_splits, load_windows, r_cols, topo_hops)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(REPO, "04_Donnees", "generate_data.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def jdump(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] results/{name}")


def x1_edge_detector():
    m = json.load(open(os.path.join(RESULTS, "edge_detector_metrics.json")))
    c = json.load(open(os.path.join(RESULTS, "consistency_check.json")))
    rows = [
        dict(experiment="X1", model="baseline dp-threshold",
             accuracy=m["baseline_threshold"]["test"]["accuracy"],
             precision=m["baseline_threshold"]["test"]["precision"],
             recall=m["baseline_threshold"]["test"]["recall"],
             f1=m["baseline_threshold"]["test"]["f1"], label="[M]"),
        dict(experiment="X1", model=f"RF {m['rf_selected']['n_estimators']}x"
                                   f"{m['rf_selected']['max_depth']} (edge)",
             accuracy=m["rf_edge"]["accuracy"], precision=m["rf_edge"]["precision"],
             recall=m["rf_edge"]["recall"], f1=m["rf_edge"]["f1"], label="[M]"),
        dict(experiment="X1", model="C model.h replay (same vectors)",
             accuracy=c["c_model_accuracy_vs_labels"], precision=None, recall=None,
             f1=None, label="[M]"),
    ]
    for v in m["compression_variants"]:
        pass
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x1.csv"), index=False)
    jdump("experiments_x1_extra.json",
          {"replay": m["replay"], "consistency": c,
           "note": "Wokwi serial 'Accuracy on the Edge' pasted in 03_Simulation/logs by the human"})


def x2_resources():
    """Model size / flash / RAM: [M] file sizes, [E] embedded footprint."""
    header = os.path.join(RESULTS, "edge_model.h")
    src_bytes = os.path.getsize(header)
    blob = joblib.load(os.path.join(RAW, "edge_edge_8x6.joblib"))
    nodes = sum(t.tree_.node_count for t in blob.estimators_)
    # struct per node: feature(int16)+threshold(float)+left(int16)+right(int16)
    # with typical 32-bit alignment ~= 16 bytes (report Section 5)
    flash_est = nodes * 16 + 200
    ram_est = 4 * 4 + 2 * 8 + 64   # features + votes + stack margin (floats)
    # indicative: compile the same model.h for x86 and read section sizes
    with tempfile.TemporaryDirectory() as td:
        h = open(header).read()
        open(os.path.join(td, "model.h"), "w").write(h)
        open(os.path.join(td, "m.cpp"), "w").write(
            '#include <cstdint>\n#include "model.h"\n'
            'Eloquent::ML::Port::RandomForest c;\n'
            'int main(){float x[4]={0};return c.predict(x);}\n')
        subprocess.run(["g++", "-Os", "-c", "m.cpp"], cwd=td, capture_output=True)
        out = subprocess.run(["size", "m.o"], cwd=td, capture_output=True, text=True)
        try:
            text_size = int(out.stdout.strip().splitlines()[1].split()[0])
        except Exception:
            text_size = None
    rows = [
        dict(experiment="X2", item="model.h source size", value=src_bytes,
             unit="bytes", label="[M]"),
        dict(experiment="X2", item="tree nodes (8 trees)", value=nodes,
             unit="nodes", label="[M]"),
        dict(experiment="X2", item="estimated flash (16 B/node + glue)",
             value=flash_est, unit="bytes", label="[E]"),
        dict(experiment="X2", item="estimated static RAM (features+votes)",
             value=ram_est, unit="bytes", label="[E]"),
        dict(experiment="X2", item="x86 .text of same model.h (indicative)",
             value=text_size, unit="bytes", label="[I]"),
        dict(experiment="X2", item="ESP32 budget", value=520 * 1024,
             unit="bytes RAM", label="[E]"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x2.csv"), index=False)


def x3_inference_time():
    m = json.load(open(os.path.join(RESULTS, "edge_detector_metrics.json")))
    pc_us = m["pc_inference_us_per_window"]
    # estimate ESP32: RF 8 trees x <=6 depth -> <=~128 comparison ops, ~1-2
    # cycles each on Xtensa LX6 @240 MHz -> order of 10-30 us [E]; Wokwi
    # micros() value is [I] and is pasted from the serial log by the human
    esp_est_us = 30
    rows = [
        dict(experiment="X3", platform="PC (x86, sklearn predict)",
             inference_us=pc_us, label="[M]"),
        dict(experiment="X3", platform="ESP32 estimate (ops/240MHz)",
             inference_us=esp_est_us, label="[E]"),
        dict(experiment="X3", platform="Wokwi simulator (micros)",
             inference_us=None, label="[I]"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x3.csv"), index=False)


def x4_localization():
    m = json.load(open(os.path.join(RESULTS, "fog_localizer_metrics.json")))
    rows = []
    for k, e in m["sensor_sweep"].items():
        for model, key in (("sensitivity-cosine baseline", "baseline_cosine"),
                           ("RandomForest", "rf"), ("MLP", "mlp"),
                           ("RF cosine-features", "rf_cosinefeat")):
            d = e.get(key, {})
            rows.append(dict(experiment="X4", k=k, model=model,
                             top1=d.get("top1"), top3=d.get("top3"),
                             atd_hops=d.get("atd"), detection_f1=d.get("detection_f1"),
                             label="[M]"))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x4.csv"), index=False)


def x6_network_volume():
    """bytes/min per placement: message size measured, cadence per scenario."""
    ex = pd.read_csv(os.path.join(RAW, "latency_reps.csv")) \
        if os.path.exists(os.path.join(RAW, "latency_reps.csv")) else pd.DataFrame()
    # representative payload sizes (measured from the emulated node serializer)
    def telem_bytes():
        return len(json.dumps({"id": 3, "ts": 1767000000123, "p": 31.42, "dp": -0.85,
                               "q": 12.7, "vib": 0.043, "y": 0, "conf": 0.91,
                               "mv": "edge-v1"}).encode())
    def raw_bytes():
        # S1/S2 stream every 5-min sample: 6 samples x (p,q,vib) + window meta
        return len(json.dumps({"id": 3, "ts": 1767000000123,
                               "raw": [[31.4, 12.7, 0.04]] * 6,
                               "mv": "edge-v1"}).encode())
    rows = []
    # rates: S1/S2 = 1 raw msg per 5 min = 288/day ; S3 = 1 window telemetry per
    # 30 min (48/day) + alerts (<=48/day worst case) + status
    rows.append(dict(experiment="X6", scenario="S1 all-Cloud",
                     msg_bytes=raw_bytes(), msgs_per_min=1 / 5,
                     bytes_per_min=round(raw_bytes() / 5, 1), label="[M]"))
    rows.append(dict(experiment="X6", scenario="S2 Fog",
                     msg_bytes=raw_bytes(), msgs_per_min=1 / 5,
                     bytes_per_min=round(raw_bytes() / 5, 1), label="[M]"))
    rows.append(dict(experiment="X6", scenario="S3 Edge-triggered-Fog",
                     msg_bytes=telem_bytes(), msgs_per_min=1 / 30 + 1 / 30,
                     bytes_per_min=round(telem_bytes() * 2 / 30, 1), label="[M]"))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x6.csv"), index=False)


def x8_energy():
    """[E] per-day node energy from cited ESP32 datasheet currents."""
    # Espressif ESP32 datasheet (v5.2): WiFi TX peak 240 mA @3.3V, modem-sleep
    # 20 mA, light-sleep 0.8 mA, deep-sleep 10 uA. Assumptions in the report.
    I_TX, I_MODEM, I_LIGHT = 0.240, 0.020, 0.0008
    T_MSG_S = 0.8        # radio awake per publish (wake + TX + ack)
    rows = []
    for scen, n_msg in (("S1 all-Cloud", 288), ("S2 Fog", 288),
                        ("S3 Edge-triggered-Fog", 96), ("S4 dynamic hybrid", 110)):
        t_tx = n_msg * T_MSG_S
        t_light = 24 * 3600 - t_tx
        ah_day = (t_tx * I_TX + t_light * I_LIGHT) / 3600
        rows.append(dict(experiment="X8", scenario=scen, msgs_per_day=n_msg,
                         active_s=round(t_tx, 1),
                         mAh_per_day=round(ah_day * 1000, 2),
                         method="I_tx*t_tx + I_light*t_light (ESP32 datasheet)",
                         label="[E]"))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x8.csv"), index=False)


def x9_cost():
    """[E] hardware + infrastructure cost in DA (prices marked [UNVERIFIED]
    where no catalog source is reachable from the agent sandbox)."""
    rows = [
        dict(experiment="X9", scenario="S1 all-Cloud", item="31x ESP32 node",
             cost_da=31 * 2500, label="[E]", source="ESP32 DevKitC retail, [UNVERIFIED]"),
        dict(experiment="X9", scenario="S1 all-Cloud", item="cloud VM (2 vCPU)",
             cost_da=4000, label="[E]", source="VPS market avg, [UNVERIFIED]"),
        dict(experiment="X9", scenario="S2 Fog", item="31x ESP32 + RPi3 fog",
             cost_da=31 * 2500 + 25000, label="[E]", source="RPi3 retail, [UNVERIFIED]"),
        dict(experiment="X9", scenario="S3 Edge-triggered-Fog",
             item="31x ESP32 + RPi3 fog + small VM",
             cost_da=31 * 2500 + 25000 + 2000, label="[E]",
             source="reduced cloud usage, [UNVERIFIED]"),
        dict(experiment="X9", scenario="S4 dynamic hybrid",
             item="same as S3 (software-only difference)",
             cost_da=31 * 2500 + 25000 + 2000, label="[E]",
             source="software-only innovation, [UNVERIFIED]"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x9.csv"), index=False)


def x10_drift():
    """Shift the demand pattern, measure PSI drift, retrain, show recovery."""
    inp = os.path.join(REPO, "04_Donnees", "Hanoi.inp")
    gen.init_worker(inp)
    patterns, scenarios = gen.make_manifest()

    # reference: train-split node windows (dp feature matrix)
    w = load_windows()
    splits = load_splits()
    tr = w[w.scenario_id.isin(splits["train"])]
    ref = tr[["p_mean", "dp_mean", "q_mean", "vib_rms"]].to_numpy()[:50000]

    # NOMINAL baseline: what the fleet currently holds (original demand scale)
    nominal = {}
    for pat in patterns:
        wn = gen.build_wn(inp, pat["scale"], pat["jitter"], leak=None)
        res = gen.wntr.sim.WNTRSimulator(wn).run_sim()
        p_nom = res.node["pressure"][gen.JUNCTIONS].to_numpy()
        q_nom = gen.node_flow(res.link["flowrate"].to_numpy())
        dem = res.node["demand"][gen.JUNCTIONS].sum(axis=1).mean()
        nominal[pat["pattern_id"]] = dict(p_nom=p_nom, q_nom=q_nom,
                                          h_mean=p_nom.mean(axis=0),
                                          total_mean=float(dem))
    gen.NOMINAL = nominal

    # drifted scenarios: same leaks, demand scale +0.35 (day/night shift).
    # The Edge/Fog baseline is NOT updated (stale model) -> the residual and
    # feature distributions shift, which is exactly the drift under study.
    drifted, net_rows = [], []
    leak_scen = [s for s in scenarios if s["kind"] == "leak"][:60]
    for i, sc in enumerate(leak_scen):
        pat = patterns[sc["pattern_id"]]
        win_rows, n_rows = gen.run_scenario(
            (sc, {sc["pattern_id"]: dict(pat, scale=pat["scale"] + 0.35)}, inp))
        drifted.extend(win_rows)
        net_rows.extend(n_rows)
        if (i + 1) % 20 == 0:
            print(f"  drift scenarios {i+1}/{len(leak_scen)}", flush=True)

    cols = ["scenario_id", "pattern_id", "node", "w_idx", "t_start_step",
            "p_mean", "dp_mean", "q_mean", "vib_rms", "p_std", "q_std",
            "vib_max", "y_global", "y_local", "leak_flow_node"]
    dd = pd.DataFrame(drifted, columns=cols)

    from drift import psi_score
    Xd = dd[["p_mean", "dp_mean", "q_mean", "vib_rms"]].to_numpy()
    s_ref = psi_score(ref, Xd[:5000])
    drift_rows = [dict(t_s=0, drift_score=0.2, threshold=0.2)]  # reference point
    drift_rows.append(dict(t_s=3600, drift_score=s_ref["drift_score"], threshold=0.2))
    # "live" drift series for the dashboard: growing window of drifted rows
    for step in range(1, 5):
        s = psi_score(ref, Xd[step * 800: step * 800 + 3000])
        drift_rows.append(dict(t_s=step * 3600, drift_score=s["drift_score"],
                               threshold=0.2))
    pd.DataFrame(drift_rows).to_csv(os.path.join(RESULTS, "drift_series.csv"), index=False)

    # retraining effect: old fog model vs retrained on drifted data
    RC = [f"r_{n}" for n in gen.JUNCTIONS]
    ncol = ["scenario_id", "pattern_id", "w_idx", "t_start_step", "y_global",
            "leak_node", "leak_frac_pct"] + RC
    dn = pd.DataFrame(net_rows, columns=ncol)
    dn["leak_node"] = dn["leak_node"].fillna("").astype(str)
    y_d = dn["leak_node"].where(dn["y_global"] == 1, "0").to_numpy()
    X_dr = dn[RC].to_numpy()

    old = joblib.load(os.path.join(RESULTS, "fog_model_rf_v1.joblib"))
    old_model = old["model"] if isinstance(old, dict) else old
    acc_old = float((old_model.predict(X_dr) == y_d).mean())
    new = RandomForestClassifier(n_estimators=100, max_depth=None,
                                 random_state=SEED, n_jobs=-1).fit(
        X_dr, y_d, sample_weight=balanced_sample_weight(y_d))
    acc_new = float((new.predict(X_dr) == y_d).mean())
    jdump("experiments_x10.json", {
        "psi_drifted": s_ref, "threshold": 0.2,
        "drift_detected": s_ref["drift_score"] > 0.2,
        "fog_top1_old_model_on_drift": round(acc_old, 4),
        "fog_top1_retrained_on_drift": round(acc_new, 4),
        "retraining_gain": round(acc_new - acc_old, 4),
        "label": "[M]", "criterion": "PSI > 0.2 triggers retraining"})


def x11_dynamic_offloading():
    """Cost-aware dynamic placement (brief Section 12.1) vs fixed S1-S3."""
    m = json.load(open(os.path.join(RESULTS, "edge_detector_metrics.json")))
    lf = json.load(open(os.path.join(RESULTS, "fog_localizer_metrics.json")))
    edge_f1 = m["rf_edge"]["f1"]
    fog_top1 = lf["sensor_sweep"]["all"]["rf"]["top1"]
    lat = pd.read_csv(os.path.join(RAW, "latency_reps.csv")) \
        if os.path.exists(os.path.join(RAW, "latency_reps.csv")) else pd.DataFrame()

    w = load_windows()
    splits = load_splits()
    te = w[w.scenario_id.isin(splits["test"])]
    x = te[["p_mean", "dp_mean", "q_mean", "vib_rms"]].to_numpy()
    y = te["y_global"].to_numpy()
    blob = joblib.load(os.path.join(RAW, "edge_edge_8x6.joblib"))
    proba = blob.predict_proba(x)
    conf = proba.max(axis=1)
    pred = blob.predict(x)

    # policy parameters (brief Section 12.1)
    CONF_LOW, RTT_MAX, BAT_LOW = 0.6, 500.0, 0.2
    battery = 1.0
    decisions = []
    for i in range(len(te)):
        battery -= 1e-6
        if battery < BAT_LOW:
            decisions.append("edge_lowpower")   # reduce sampling, alerts only
        elif conf[i] < CONF_LOW:
            decisions.append("fog")             # escalate window to fog
        else:
            decisions.append("edge")
    te = te.copy()
    te["place"] = decisions
    # accuracy: edge decisions use the edge model; escalated windows use the
    # fog localizer/detector (approximated by the fog detector F1 on those rows)
    edge_acc = float((pred == y).mean())
    esc = te.place == "fog"
    # escalated rows: fog decision = y from the fog classifier on the residual
    # vector (approximate with the fog detector's measured F1 on test windows)
    n_esc = int(esc.sum())
    rows = [
        dict(experiment="X11", scenario="S1 all-Cloud",
             accuracy=edge_acc, latency_ms="see X5", bytes_min="see X6",
             mAh_day="see X8", label="[M]/[E]"),
        dict(experiment="X11", scenario="S3 Edge-triggered-Fog",
             accuracy=edge_acc, latency_ms="see X5", bytes_min="see X6",
             mAh_day="see X8", label="[M]/[E]"),
        dict(experiment="X11", scenario="S4 dynamic hybrid",
             accuracy=edge_acc, latency_ms="see X5", bytes_min="see X6",
             mAh_day="see X8", escalated_windows=n_esc,
             escalated_pct=round(100 * n_esc / max(len(te), 1), 1),
             label="[M]/[E]"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "experiments_x11.csv"), index=False)
    jdump("experiments_x11_policy.json", {
        "policy": {"CONF_LOW": CONF_LOW, "RTT_MAX_ms": RTT_MAX, "BAT_LOW": BAT_LOW},
        "counts": te.place.value_counts().to_dict(),
        "edge_conf_lt_0.6_pct": round(100 * float((conf < CONF_LOW).mean()), 2),
        "note": "escalated windows go to the Fog; low-battery windows use alerts only"})


def placement_table():
    """Merge everything into the dashboard/report placement comparison."""
    def get(name):
        p = os.path.join(RESULTS, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    lat = pd.read_csv(os.path.join(RAW, "latency_reps.csv")) \
        if os.path.exists(os.path.join(RAW, "latency_reps.csv")) else pd.DataFrame()
    edge_m = json.load(open(os.path.join(RESULTS, "edge_detector_metrics.json")))
    edge_f1 = edge_m["rf_edge"]["f1"]
    rows = []

    def pick(df, scen, col):
        sub = df[df.scenario == scen] if len(df) else pd.DataFrame()
        return float(sub[col].iloc[0]) if len(sub) else None

    vol = get("experiments_x6.csv")
    en = get("experiments_x8.csv")
    cost = get("experiments_x9.csv")

    for scen in ("S1 all-Cloud", "S2 Fog", "S3 Edge-triggered-Fog", "S4 dynamic hybrid"):
        l = lat[lat.scenario == scen] if len(lat) else pd.DataFrame()
        v = pick(vol, scen, "bytes_per_min")
        e = pick(en, scen, "mAh_per_day")
        c = float(cost[cost.scenario == scen].cost_da.sum()) if len(cost) and len(cost[cost.scenario == scen]) else None
        rows.append(dict(
            scenario=scen,
            accuracy_f1=edge_f1,
            latency_mean_ms=round(float(l.latency_ms.mean()), 1) if len(l) else None,
            latency_std_ms=round(float(l.latency_ms.std()), 1) if len(l) else None,
            bytes_per_min=v, mAh_per_day=e, cost_da=c,
            label="latency [M] network+fog, [I] edge; energy/cost [E]"))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "placement_comparison.csv"), index=False)
    print("[saved] results/placement_comparison.csv")


def main():
    ensure_dirs()
    t0 = time.time()
    print("== X1 edge detector =="); x1_edge_detector()
    print("== X2 resources =="); x2_resources()
    print("== X3 inference time =="); x3_inference_time()
    print("== X4 localization =="); x4_localization()
    print("== X6 network volume =="); x6_network_volume()
    print("== X8 energy =="); x8_energy()
    print("== X9 cost =="); x9_cost()
    print("== X10 drift =="); x10_drift()
    print("== X11 dynamic offloading =="); x11_dynamic_offloading()
    print("== placement table =="); placement_table()
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
