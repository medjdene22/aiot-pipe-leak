#!/usr/bin/env python3
"""
Fog leak localizer: baselines, advanced models, sensor-count sweep, export.

Input: network-level residual windows (04_Donnees/generated/residuals.csv.gz),
one row per (scenario, 30-min window) with the 31-node pressure residual vector
r = p_measured - p_nominal. Target: leak node id (or "0" = no leak).

Models per sensor subset k in {2, 3, 5, 8, all}:
  - baseline: sensitivity-matrix matching (cosine similarity between the
    residual vector and per-node sensitivity profiles learned on train)
  - RandomForest (100 trees)
  - MLP (64-32, scaled features; scaler fit on train only)

Metrics: top-1 / top-3 accuracy over leak-active windows, average topological
distance (hops) between predicted and true leak node, detection F1 (leak vs
no-leak), per leak-size band. Everything saved to results/.

Ablation: unit-norm (cosine) residual features on/off (brief Section 9).
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import (MODELS, RAW, RESULTS, SEED, assert_no_leakage,
                    balanced_sample_weight, ensure_dirs, load_residuals,
                    load_splits, r_cols, topo_hops)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
NOLEAK = "0"


def build_sensitivity(x, y, nodes):
    """S[sensor_i, leak_node_j] = mean |residual| at sensor i for leaks at j."""
    S = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for j in nodes:
        rows = x[y == j]
        if len(rows):
            S[j] = np.abs(rows).mean(axis=0)
    return S


def greedy_sensors(S, k):
    """Greedy coverage: sensors whose residual profiles cover the most leak nodes."""
    nodes = list(S.index)
    selected = []
    cover = pd.Series(0.0, index=S.columns)
    for _ in range(min(k, len(nodes))):
        best, best_score = None, -1
        for s in nodes:
            if s in selected:
                continue
            new_cover = np.maximum(cover, S.loc[s])
            score = float(new_cover.sum())
            if score > best_score:
                best, best_score = s, score
        selected.append(best)
        cover = np.maximum(cover, S.loc[best])
    return selected


def topk_acc(proba, classes, y_true, nodes, k):
    hits = 0
    n = 0
    for row, yt in zip(proba, y_true):
        if yt == NOLEAK:
            continue
        order = np.argsort(-row)[:k]
        top = [classes[i] for i in order]
        hits += int(yt in top)
        n += 1
    return round(hits / max(n, 1), 4)


def atd_hops(proba_pred_nodes, y_true, hops):
    ds = []
    for pred, yt in zip(proba_pred_nodes, y_true):
        if yt == NOLEAK or pred == NOLEAK or pred not in hops.columns:
            continue
        ds.append(float(hops.loc[pred, yt]))
    return round(float(np.mean(ds)), 3) if ds else None


def main():
    ensure_dirs()
    t0 = time.time()
    df = load_residuals()
    splits = load_splits()
    assert_no_leakage(splits, df)
    RC = r_cols(df)
    nodes = [c[2:] for c in RC]
    hops, _ = topo_hops()
    print(f"residual windows: {len(df):,} | leak-active: {int(df.y_global.sum()):,}")

    tr = df[df.scenario_id.isin(splits["train"])]
    va = df[df.scenario_id.isin(splits["val"])]
    te = df[df.scenario_id.isin(splits["test"])]
    y_tr = tr["leak_node"].where(tr["y_global"] == 1, NOLEAK).to_numpy()
    y_va = va["leak_node"].where(va["y_global"] == 1, NOLEAK).to_numpy()
    y_te = te["leak_node"].where(te["y_global"] == 1, NOLEAK).to_numpy()

    results = {"seed": SEED, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "sensor_sweep": {}, "hops_note": "pipe-hop shortest paths on Hanoi"}

    # sensor selection on TRAIN only (greedy sensitivity coverage)
    S = build_sensitivity(tr[RC].to_numpy(), y_tr, nodes)
    S.to_csv(os.path.join(RESULTS, "sensitivity_matrix.csv"))
    all_k = {"2": 2, "3": 3, "5": 5, "8": 8, "all": len(nodes)}
    selected_map = {kk: greedy_sensors(S, v) for kk, v in all_k.items()}
    with open(os.path.join(RESULTS, "sensor_subsets.json"), "w") as f:
        json.dump(selected_map, f, indent=1)

    for kk, sensors in selected_map.items():
        cols = [f"r_{s}" for s in sensors]
        xtr, xva, xte = tr[cols].to_numpy(), va[cols].to_numpy(), te[cols].to_numpy()
        entry = {"sensors": sensors, "k": len(sensors)}

        # --- baseline: sensitivity-profile cosine matching ---
        # NOTE: profiles and query must live in the same sign space. Pressure
        # residuals are negative at the leak; cosine on signed vectors is the
        # classical sensitivity-matrix matching (Perez/Puig style).
        prof = {}
        for j in nodes:
            rows = xtr[y_tr == j]
            if len(rows):
                prof[j] = rows.mean(axis=0)
        pj = np.array([prof[j] for j in prof])
        pj_names = list(prof)

        xn = xte / (np.linalg.norm(xte, axis=1, keepdims=True) + 1e-9)
        pn = pj / (np.linalg.norm(pj, axis=1, keepdims=True) + 1e-9)
        sim = xn @ pn.T
        top1_idx = sim.argmax(axis=1)
        top3_idx = np.argsort(-sim, axis=1)[:, :3]
        pred_bl = np.array([pj_names[i] for i in top1_idx])
        leak_mask = y_te != NOLEAK
        bl_top1 = round(float((pred_bl[leak_mask] == y_te[leak_mask]).mean()), 4)
        bl_top3 = round(float(np.mean([
            yt in [pj_names[i] for i in row]
            for row, yt in zip(top3_idx, y_te) if yt != NOLEAK])), 4)
        # no-leak rejection for the baseline: residual norm below train noise level
        norm_neg = np.linalg.norm(xtr[y_tr == NOLEAK], axis=1)
        thr = float(np.quantile(norm_neg, 0.99))
        pred_bl_rej = np.where(np.linalg.norm(xte, axis=1) < thr, NOLEAK, pred_bl)

        entry["baseline_cosine"] = dict(
            top1=bl_top1, top3=bl_top3,
            atd=atd_hops(pred_bl_rej, y_te, hops),
            detection_f1=round(float(f1_score(
                (y_te != NOLEAK).astype(int), (pred_bl_rej != NOLEAK).astype(int),
                zero_division=0)), 4))

        # --- random forest ---
        for unit_norm in (False, True):
            def norm(a):
                return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9) if unit_norm else a
            rf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                        random_state=SEED,
                                        n_jobs=-1).fit(norm(xtr), y_tr,
                                                       sample_weight=balanced_sample_weight(y_tr))
            proba = rf.predict_proba(norm(xte))
            classes = list(rf.classes_)
            pred1 = np.array([classes[i] for i in proba.argmax(axis=1)])
            key = "rf_cosinefeat" if unit_norm else "rf"
            entry[key] = dict(
                top1=topk_acc(proba, classes, y_te, nodes, 1),
                top3=topk_acc(proba, classes, y_te, nodes, 3),
                atd=atd_hops(pred1, y_te, hops),
                detection_f1=round(float(f1_score(
                    (y_te != NOLEAK).astype(int), (pred1 != NOLEAK).astype(int),
                    zero_division=0)), 4))
            if kk == "all" and not unit_norm:
                import joblib
                joblib.dump(rf, os.path.join(RESULTS, "fog_model_rf_v1.joblib"))
            if not unit_norm:
                # per-k deployment artifacts for the Fog service
                import joblib
                joblib.dump({"model": rf, "sensor_nodes": sensors,
                             "version": "fog-v1"},
                            os.path.join(MODELS, f"fog_model_rf_k{kk}.joblib"))

        # --- MLP ---
        sc = StandardScaler().fit(xtr)          # fit on train only!
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                            random_state=SEED).fit(sc.transform(xtr), y_tr)
        proba = mlp.predict_proba(sc.transform(xte))
        classes = list(mlp.classes_)
        pred1 = np.array([classes[i] for i in proba.argmax(axis=1)])
        entry["mlp"] = dict(
            top1=topk_acc(proba, classes, y_te, nodes, 1),
            top3=topk_acc(proba, classes, y_te, nodes, 3),
            atd=atd_hops(pred1, y_te, hops),
            detection_f1=round(float(f1_score(
                (y_te != NOLEAK).astype(int), (pred1 != NOLEAK).astype(int),
                zero_division=0)), 4))
        if kk == "all":
            import joblib
            joblib.dump({"scaler": sc, "model": mlp},
                        os.path.join(RESULTS, "fog_model_mlp_v1.joblib"))
        import joblib
        joblib.dump({"scaler": sc, "model": mlp, "sensor_nodes": sensors,
                     "version": "fog-v1"},
                    os.path.join(MODELS, f"fog_model_mlp_k{kk}.joblib"))

        results["sensor_sweep"][kk] = entry
        print(f"k={kk:>3} | RF top1={entry['rf']['top1']} top3={entry['rf']['top3']} "
              f"ATD={entry['rf']['atd']} | MLP top1={entry['mlp']['top1']} "
              f"| base top1={entry['baseline_cosine']['top1']}")

    # leak-size ablation (small <3.3%, medium, large) on the best model (RF, all k)
    cols = [f"r_{s}" for s in nodes]
    rf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                random_state=SEED,
                                n_jobs=-1).fit(tr[cols].to_numpy(), y_tr,
                                               sample_weight=balanced_sample_weight(y_tr))
    proba = rf.predict_proba(te[cols].to_numpy())
    classes = list(rf.classes_)
    pred1 = np.array([classes[i] for i in proba.argmax(axis=1)])
    te_m = te.copy()
    te_m["pred"] = pred1
    bands = {"small_1_3.3pct": (0, 3.3), "medium_3.3_6.7pct": (3.3, 6.7),
             "large_6.7_10pct": (6.7, 10.1)}
    abl = {}
    for b, (lo, hi) in bands.items():
        m = te_m[(te_m.y_global == 1) & (te_m.leak_frac_pct >= lo) & (te_m.leak_frac_pct < hi)]
        abl[b] = dict(n=int(len(m)),
                      top1=round(float((m.pred == m.leak_node).mean()), 4) if len(m) else None)
    results["leak_size_ablation"] = abl

    with open(os.path.join(RESULTS, "fog_localizer_metrics.json"), "w") as f:
        json.dump(results, f, indent=1)

    # flat table for the report
    rows = []
    for kk, e in results["sensor_sweep"].items():
        for model in ("baseline_cosine", "rf", "rf_cosinefeat", "mlp"):
            rows.append(dict(k=kk, model=model, **e[model]))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fog_localizer_table.csv"), index=False)
    print(f"[{time.time()-t0:.0f}s] saved -> results/fog_localizer_metrics.json")


if __name__ == "__main__":
    main()
