#!/usr/bin/env python3
"""
Cloud retraining (brief Section 8): regenerate/augment -> retrain the Fog
localization classifier -> evaluate on the fixed test set -> save versioned
model + register metrics.

Can also be run on Google Colab / offline: pass --from-db cloud.db to retrain
from an exported database of ingested measurements instead of the generated
dataset (Colab compatibility, brief Section 8).

Usage:
  python train.py --register --db cloud.db
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "ia"))
from common import (RESULTS, RAW, balanced_sample_weight, ensure_dirs,  # noqa: E402
                    load_residuals, load_splits, r_cols)

MODELS_DIR = os.path.join(HERE, "..", "ia", "models")
NOLEAK = "0"
SEED = 42


def next_version():
    existing = glob.glob(os.path.join(MODELS_DIR, "fog_model_rf_v*.joblib"))
    ns = [int(os.path.basename(p).split("_v")[1].split(".")[0]) for p in existing]
    return (max(ns) + 1) if ns else 1


def data_from_db(db_path):
    """Fallback training data source: exported measurements database."""
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT payload FROM measures WHERE kind='telemetry'").fetchall()
    con.close()
    recs = [json.loads(r[0]) for r in rows]
    if not recs:
        raise SystemExit("DB has no telemetry rows")
    df = pd.DataFrame(recs)
    df["leak_node"] = df.get("leak_node", "")
    df["y_global"] = df.get("y_global", 0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true", help="register version in cloud DB")
    ap.add_argument("--db", default="cloud.db")
    ap.add_argument("--from-db", default=None, help="retrain from exported DB instead")
    args = ap.parse_args()

    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.from_db:
        df = data_from_db(args.from_db)
        RC = [c for c in df.columns if c.startswith("r_")]
        splits = {"train": sorted(df.scenario_id.unique()),
                  "val": sorted(df.scenario_id.unique()),
                  "test": sorted(df.scenario_id.unique())}
    else:
        df = load_residuals()
        RC = r_cols(df)
        splits = load_splits()

    tr = df[df.scenario_id.isin(splits["train"])]
    te = df[df.scenario_id.isin(splits["test"])]
    y_tr = tr["leak_node"].where(tr["y_global"] == 1, NOLEAK).to_numpy()
    y_te = te["leak_node"].where(te["y_global"] == 1, NOLEAK).to_numpy()
    x_tr, x_te = tr[RC].to_numpy(), te[RC].to_numpy()

    clf = RandomForestClassifier(n_estimators=100, max_depth=12,
                                 random_state=SEED,
                                 n_jobs=-1).fit(x_tr, y_tr,
                                                sample_weight=balanced_sample_weight(y_tr))
    acc = float((clf.predict(x_te) == y_te).mean())
    leak_mask = y_te != NOLEAK
    top1 = float((clf.predict(x_te)[leak_mask] == y_te[leak_mask]).mean()) if leak_mask.any() else 0.0
    metrics = {"test_accuracy": round(acc, 4), "top1_leak_windows": round(top1, 4),
               "n_train": len(tr), "n_test": len(te),
               "features": [c[2:] for c in RC]}

    v = next_version()
    version = f"fog-v{v}"
    path = os.path.join(MODELS_DIR, f"fog_model_rf_v{v}.joblib")
    joblib.dump({"model": clf, "sensor_nodes": metrics["features"],
                 "version": version}, path)
    joblib.dump({"model": clf, "sensor_nodes": metrics["features"],
                 "version": version},
                os.path.join(MODELS_DIR, "fog_model_latest.joblib"))
    # also refresh the per-k exports used by the fog service
    for k in ("2", "3", "5", "8", "all"):
        sub = os.path.join(RESULTS, "sensor_subsets.json")
        if os.path.exists(sub):
            nodes = json.load(open(sub))[k]
            cols = [f"r_{n}" for n in nodes]
            if len(cols) <= len(RC):
                m = RandomForestClassifier(n_estimators=100, max_depth=12,
                                           random_state=SEED, n_jobs=-1).fit(
                    tr[cols].to_numpy(), y_tr,
                    sample_weight=balanced_sample_weight(y_tr))
                joblib.dump({"model": m, "sensor_nodes": nodes, "version": version},
                            os.path.join(MODELS_DIR, f"fog_model_rf_k{k}.joblib"))

    if args.register:
        con = sqlite3.connect(args.db)
        con.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?)",
                    (version, int(time.time() * 1000), json.dumps(metrics), path))
        con.commit()
        con.close()

    with open(os.path.join(RESULTS, "cloud_train_log.json"), "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "version": version, **metrics}) + "\n")
    print(json.dumps({"version": version, **metrics}, indent=1))


if __name__ == "__main__":
    main()
