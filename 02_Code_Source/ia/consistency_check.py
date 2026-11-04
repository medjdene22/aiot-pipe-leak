#!/usr/bin/env python3
"""
Consistency check (brief Section 6, mandatory): the SAME inputs through the
model on the PC (scikit-learn) and in the embedded C code (model.h, the exact
binary that runs in the simulator) must give the SAME outputs.

Method: compile model.h with a small C++ harness (g++), feed every vector of
edge_test_data.h (the file uploaded to Wokwi), compare labels and confidences
with the sklearn model. Result: results/consistency_check.json + log.

Usage: python consistency_check.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "results")

HARNESS = r"""
#include <cstdio>
#include <cstdint>
#include "model.h"
Eloquent::ML::Port::RandomForest clf;
int main() {
    float x[4];
    int y;
    while (scanf("%f %f %f %f %d", &x[0], &x[1], &x[2], &x[3], &y) == 5) {
        int pred = clf.predict(x);
        printf("%d %.6f\n", pred, clf.confidence());
    }
    return 0;
}
"""


def parse_test_data(path):
    """Parse the generated edge_test_data.h (X_TEST + Y_TEST)."""
    src = open(path).read()
    nf = int(re.search(r"#define NF (\d+)", src).group(1))
    ns = int(re.search(r"#define NS (\d+)", src).group(1))
    body = src.split("X_TEST[NS][NF] PROGMEM = {")[1].split("};")[0]
    rows = re.findall(r"\{([^}]*)\}", body)
    X = np.array([[float(v) for v in r.split(",")] for r in rows[:ns]])
    ybody = src.split("Y_TEST[NS] PROGMEM = {")[1].split("};")[0]
    y = np.array([int(v) for v in ybody.split(",")])
    assert X.shape == (ns, nf) and y.shape == (ns,)
    return X, y


def main():
    import joblib
    X, y = parse_test_data(os.path.join(RESULTS, "edge_test_data.h"))

    # sklearn side: the trained edge model (same one that produced model.h)
    blob = joblib.load(os.path.join(RESULTS, "raw", "edge_edge_8x6.joblib"))
    sk_pred = blob.predict(X).astype(int)

    # C side: compile the shipped model.h with g++
    with tempfile.TemporaryDirectory() as td:
        src = open(os.path.join(RESULTS, "edge_model.h")).read()
        hpath = os.path.join(td, "model.h")
        with open(hpath, "w") as f:
            f.write(src)
        cpath = os.path.join(td, "harness.cpp")
        with open(cpath, "w") as f:
            f.write(HARNESS)
        binpath = os.path.join(td, "harness")
        out = subprocess.run(["g++", "-O2", "-o", binpath, cpath],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr[-2000:])
            raise SystemExit("C harness compilation failed")
        inp = "\n".join(
            f"{a:.5f} {b:.5f} {c:.5f} {d:.5f} {int(l)}"
            for (a, b, c, d), l in zip(X, y))
        run = subprocess.run([binpath], input=inp, capture_output=True, text=True)
        lines = [l.split() for l in run.stdout.strip().splitlines()]
        c_pred = np.array([int(l[0]) for l in lines])
        c_conf = np.array([float(l[1]) for l in lines])

    mismatches = int((c_pred != sk_pred).sum())
    # sklearn's predict_proba averages per-tree LEAF fractions (soft votes),
    # while the C model.h counts per-tree hard votes. Compare like with like:
    hard_vote = np.zeros((len(X), len(blob.classes_)))
    for t in blob.estimators_:
        hard_vote[np.arange(len(X)), t.predict(X).astype(int)] += 1
    hard_vote /= len(blob.estimators_)
    sk_conf_hard = hard_vote.max(axis=1)
    result = {
        "n_samples": int(len(y)),
        "sklearn_accuracy_vs_labels": round(float((sk_pred == y).mean()), 4),
        "c_model_accuracy_vs_labels": round(float((c_pred == y).mean()), 4),
        "pc_vs_c_mismatches": mismatches,
        "consistent": mismatches == 0,
        "max_conf_diff_hard_votes": round(float(np.abs(c_conf - sk_conf_hard).max()), 6),
        "note": "C confidence = tree hard-vote fraction; sklearn predict_proba = "
                "soft leaf fraction (documented difference)",
    }
    with open(os.path.join(RESULTS, "consistency_check.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    if mismatches:
        sys.exit(f"CONSISTENCY FAILURE: {mismatches} mismatches")
    print("CONSISTENCY OK: C model.h == sklearn model on all replay vectors")


if __name__ == "__main__":
    main()
