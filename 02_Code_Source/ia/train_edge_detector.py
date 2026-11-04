#!/usr/bin/env python3
"""
Edge leak detector: training, baselines, compression study, export to C.

Data: node-level windows (04_Donnees/generated/windows.csv.gz), features
X = [p_mean, dp_mean, q_mean, vib_rms] (pressure m, residual m, flow L/s,
vibration RMS g), label y_global = "leak active in the network during the
window" (brief Section 5.3).

No data leakage: scenarios are assigned to splits by generate_data.py and this
script only ever fits scalers/models on the train split.

Outputs (results/):
  edge_detector_metrics.json    all metrics + export info
  edge_detector_table.csv       model comparison table (for the report)
  edge_model.h                  exported TinyML model (micromlgen/emlearn C)
  edge_test_data.h              replay dataset for the ESP32 simulator
  raw/edge_model_predictions.csv  per-sample test predictions (evidence)
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from common import (EDGE_FEATURES, RAW, RESULTS, SEED, assert_no_leakage,
                    clf_metrics, ensure_dirs, load_splits, load_windows)

NS_REPLAY = 500          # samples in test_data.h
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
rng = np.random.default_rng(SEED)


def fit_rf(x_tr, y_tr, n_est, depth, seed=SEED):
    return RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, class_weight="balanced",
        random_state=seed, n_jobs=-1).fit(x_tr, y_tr)


def threshold_baseline(x_tr, y_tr, x_va, y_va):
    """Baseline 1 (brief Section 9): fixed threshold on dp. Tune on train."""
    dp_tr = x_tr[:, 1]
    best = (None, -1)
    for theta in np.arange(-2.0, 0.05, 0.05):
        pred = (dp_tr < theta).astype(int)
        f1 = clf_metrics(y_tr, pred)["f1"]
        if f1 > best[1]:
            best = (round(float(theta), 3), f1)
    theta = best[0]
    return theta, (x_va[:, 1] < theta).astype(int)


def export_c(model, n_features, path_h, class_name="RandomForest"):
    """Export with micromlgen (fallback: emlearn). Adds a confidence() method
    exposing the tree-vote fraction (== predict_proba for an RF) because the
    stock micromlgen export only returns the class label."""
    try:
        from micromlgen import port
        code = port(model)
        tool = "micromlgen"
    except Exception as e:  # noqa: BLE001
        import emlearn
        emlearn.convert(model, method="inline").save_file(path_h)
        code = open(path_h).read()
        tool = f"emlearn (micromlgen failed: {e})"
    if tool == "micromlgen":
        n_trees = int(getattr(model, "n_estimators", 1))
        # patch 1: member + accessor for the vote fraction
        code = code.replace(
            "class RandomForest {\n                public:",
            "class RandomForest {\n                public:\n"
            "                    float _lastConf = 0;\n"
            "                    float confidence() { return _lastConf; }")
        # patch 2: store the vote fraction right before returning
        code = code.replace(
            "                        return classIdx;",
            f"                        _lastConf = votes[classIdx] / {float(n_trees):.1f}f;\n"
            "                        return classIdx;")
    with open(path_h, "w") as f:
        f.write(code)
    return tool, os.path.getsize(path_h)


def main():
    ensure_dirs()
    t0 = time.time()
    df = load_windows()
    splits = load_splits()
    assert_no_leakage(splits, df)
    print(f"windows: {len(df):,} | y pos: {int(df.y_global.sum()):,}")

    tr = df[df.scenario_id.isin(splits["train"])]
    va = df[df.scenario_id.isin(splits["val"])]
    te = df[df.scenario_id.isin(splits["test"])]
    x_tr, y_tr = tr[EDGE_FEATURES].to_numpy(), tr["y_global"].to_numpy()
    x_va, y_va = va[EDGE_FEATURES].to_numpy(), va["y_global"].to_numpy()
    x_te, y_te = te[EDGE_FEATURES].to_numpy(), te["y_global"].to_numpy()

    results = {"seed": SEED, "features": EDGE_FEATURES,
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "label": "y_global (leak active in window)"}

    # ---- 1. baseline: threshold on dp (brief Section 9) ----
    theta, pred_va = threshold_baseline(x_tr, y_tr, x_va, y_va)
    pred_te = (x_te[:, 1] < theta).astype(int)
    results["baseline_threshold"] = dict(
        theta=theta, val=clf_metrics(y_va, pred_va), test=clf_metrics(y_te, pred_te))
    print(f"baseline dp<{theta}: test F1={results['baseline_threshold']['test']['f1']}")

    # ---- 2. random forest sweep (<=8 trees, <=6 depth for the edge budget) ----
    sweep = []
    for n_est in (4, 8):
        for depth in (3, 4, 6):
            m = fit_rf(x_tr, y_tr, n_est, depth)
            mv = clf_metrics(y_va, m.predict(x_va))
            sweep.append(dict(n_estimators=n_est, max_depth=depth,
                              val_f1=mv["f1"], val_accuracy=mv["accuracy"]))
    best = max(sweep, key=lambda d: d["val_f1"])
    edge_clf = fit_rf(x_tr, y_tr, best["n_estimators"], best["max_depth"])
    results["rf_sweep"] = sweep
    results["rf_selected"] = best
    results["rf_edge"] = clf_metrics(y_te, edge_clf.predict(x_te))
    print(f"RF edge ({best}): test F1={results['rf_edge']['f1']}")

    # ---- 3. compression study (brief Section 9) ----
    variants = {}
    specs = {"full_50x12": (50, 12), "edge_8x6": (best["n_estimators"], best["max_depth"]),
             "pruned_4x3": (4, 3)}
    comp_rows = []
    for name, (ne, dp_) in specs.items():
        m = fit_rf(x_tr, y_tr, ne, dp_)
        mt = clf_metrics(y_te, m.predict(x_te))
        tmp = os.path.join(RAW, f"edge_{name}.joblib")
        import joblib
        joblib.dump(m, tmp)
        size_kb = os.path.getsize(tmp) / 1024
        # export size for the edge variant only (C array)
        comp_rows.append(dict(variant=name, n_estimators=ne, max_depth=dp_,
                              test_f1=mt["f1"], test_accuracy=mt["accuracy"],
                              model_kb=round(size_kb, 1)))
        variants[name] = mt
    results["compression"] = comp_rows
    results["compression_variants"] = variants

    # ---- 4. export the edge model to C + replay test data ----
    exp_tool, exp_bytes = export_c(edge_clf, len(EDGE_FEATURES),
                                   os.path.join(RESULTS, "edge_model.h"))
    results["export"] = dict(tool=exp_tool, header_bytes=exp_bytes,
                             header_kb=round(exp_bytes / 1024, 1))

    # balanced replay sample from the TEST split (evidence for X1)
    pos = te[te.y_global == 1]
    neg = te[te.y_global == 0]
    n_pos = min(len(pos), NS_REPLAY // 2)
    n_neg = min(len(neg), NS_REPLAY - n_pos)
    sample = pd.concat([pos.sample(n_pos, random_state=SEED),
                        neg.sample(n_neg, random_state=SEED)]).sample(frac=1, random_state=SEED)
    xs = sample[EDGE_FEATURES].to_numpy()
    ys = sample["y_global"].to_numpy().astype(int)
    with open(os.path.join(RESULTS, "edge_test_data.h"), "w") as f:
        f.write("// Auto-generated replay test set (TEST split, balanced) - do not edit\n")
        f.write(f"#define NS {len(sample)}\n#define NF {len(EDGE_FEATURES)}\n")
        f.write("const float X_TEST[NS][NF] PROGMEM = {\n")
        for row in xs:
            f.write("  {" + ", ".join(f"{v:.5f}" for v in row) + "},\n")
        f.write("};\n")
        f.write("const uint8_t Y_TEST[NS] PROGMEM = {" +
                ", ".join(str(int(v)) for v in ys) + "};\n")
    results["replay"] = dict(n_samples=int(len(sample)), n_pos=int(n_pos), n_neg=int(n_neg),
                             replay_accuracy=float((edge_clf.predict(xs) == ys).mean()))

    # ---- 5. PC inference time (basis for the [I] estimate on ESP32) ----
    one = x_te[:1]
    edge_clf.predict(np.repeat(one, 200, axis=0))  # warm-up
    t1 = time.perf_counter()
    reps = 2000
    for _ in range(reps):
        edge_clf.predict(one)
    inf_us = (time.perf_counter() - t1) / reps * 1e6
    results["pc_inference_us_per_window"] = round(float(inf_us), 2)

    # per-sample predictions as raw evidence
    pred = edge_clf.predict(x_te)
    pd.DataFrame({"y_true": y_te, "y_pred": pred,
                  "dp_mean": x_te[:, 1]}).to_csv(
        os.path.join(RAW, "edge_model_predictions.csv"), index=False)

    with open(os.path.join(RESULTS, "edge_detector_metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(RESULTS, "edge_detector_table.csv"), index=False)
    print(f"[{time.time()-t0:.0f}s] saved -> results/edge_detector_metrics.json")
    print(json.dumps({k: results[k] for k in
                      ("baseline_threshold", "rf_selected", "rf_edge", "export")}, indent=1))


if __name__ == "__main__":
    main()
