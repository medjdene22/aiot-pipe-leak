"""
Drift detection (brief Section 8): Population Stability Index (PSI) between a
reference distribution (training data) and recent observations.

Criterion used in the Cloud service:
  - PSI per feature (10 quantile bins from the reference)
  - drift score = max over features
  - retraining recommended when drift_score > DRIFT_PSI_THRESHOLD (default 0.2),
    or when a rolling accuracy estimate (when labels exist) drops by more than
    RETRAIN_ACC_DROP_POINTS (default 3.0 points) below the registered value.
"""
import numpy as np

NBINS = 10
EPS = 1e-6


def psi_score(ref: np.ndarray, recent: np.ndarray, nbins: int = NBINS) -> dict:
    """Per-feature PSI. ref/recent: (n, n_features) arrays. Returns summary dict."""
    ref = np.asarray(ref, dtype=float)
    recent = np.asarray(recent, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if recent.ndim == 1:
        recent = recent.reshape(-1, 1)
    scores = []
    for j in range(ref.shape[1]):
        r = ref[:, j]
        o = recent[:, j]
        edges = np.unique(np.quantile(r, np.linspace(0, 1, nbins + 1)))
        if len(edges) < 3:
            scores.append(0.0)
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        pr, _ = np.histogram(r, bins=edges)
        po, _ = np.histogram(o, bins=edges)
        pr = pr / max(pr.sum(), 1) + EPS
        po = po / max(po.sum(), 1) + EPS
        scores.append(float(np.sum((po - pr) * np.log(po / pr))))
    return {"psi_per_feature": [round(s, 4) for s in scores],
            "drift_score": round(float(max(scores)), 4),
            "n_recent": int(recent.shape[0])}
