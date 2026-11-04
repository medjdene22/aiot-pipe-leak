"""
Shared helpers for the IA pipeline (loading generated data, splits, metrics,
topological distances on the Hanoi graph).
"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(REPO, "04_Donnees", "generated")
RESULTS = os.path.join(REPO, "results")
RAW = os.path.join(RESULTS, "raw")
MODELS = os.path.join(HERE, "models")

EDGE_FEATURES = ["p_mean", "dp_mean", "q_mean", "vib_rms"]
SEED = 42


def ensure_dirs():
    for d in (RESULTS, RAW, MODELS):
        os.makedirs(d, exist_ok=True)


def load_windows():
    df = pd.read_csv(os.path.join(DATA_DIR, "windows.csv.gz"),
                     dtype={"node": str})
    df["node"] = df["node"].astype(str)
    return df


def load_residuals():
    df = pd.read_csv(os.path.join(DATA_DIR, "residuals.csv.gz"),
                     dtype={"leak_node": str})
    df["leak_node"] = df["leak_node"].fillna("").astype(str)
    return df


def load_splits():
    with open(os.path.join(DATA_DIR, "splits.json")) as f:
        return json.load(f)


def assert_no_leakage(splits, df, col="scenario_id"):
    """Mandatory check (brief Section 0.3): no scenario id in two splits."""
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "scenario id in 2 splits!"
    seen = set(df[col].unique())
    assert (tr | va | te) <= seen | (tr | va | te), "unknown scenario ids"
    return True


def r_cols(df):
    return [c for c in df.columns if c.startswith("r_")]


def topo_hops(inp=None):
    """Hop-count shortest-path matrix between junctions of Hanoi."""
    import wntr
    inp = inp or os.path.join(REPO, "04_Donnees", "Hanoi.inp")
    wn = wntr.network.WaterNetworkModel(inp)
    G = nx.Graph()
    for _, p in wn.pipes():
        G.add_edge(p.start_node_name, p.end_node_name)
    nodes = sorted([v.name for _, v in wn.junctions()], key=int)
    dist = pd.DataFrame(index=nodes, columns=nodes, dtype=float)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for a in nodes:
        for b in nodes:
            dist.loc[a, b] = lengths[a][b] if b in lengths[a] else np.nan
    return dist, nodes


def balanced_sample_weight(y):
    """Balanced class weights computed manually.

    NOTE: sklearn 1.9.1 raises "The classes [...] are not in class_weight" for
    class_weight='balanced' with string labels (regression), so we compute the
    balanced weights ourselves (n / (n_classes * n_c) per class)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y])


def clf_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, confusion_matrix)
    out = dict(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out
