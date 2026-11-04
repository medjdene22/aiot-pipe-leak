#!/usr/bin/env python3
"""
Cloud service (brief Section 8): FastAPI + SQLite.

REST API (brief Section 4.3):
  POST /api/mesures           ingest measurement/event from the Fog
  GET  /api/modele/latest     download latest Fog model (joblib) [+ x-model-version]
  GET  /api/modele/versions   list versions + metrics
  POST /api/retrain           trigger retraining (idempotent, logs to results/)
  GET  /api/drift             current drift score + retraining recommendation
  GET  /api/health            liveness

Tables: measures, localizations, model_versions, drift_log.

Run: uvicorn cloud_api:app --host 0.0.0.0 --port 8000
"""
import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ia"))
from drift import psi_score  # noqa: E402

CLOUD_DB = os.environ.get("CLOUD_DB", "cloud.db")
MODELS_DIR = os.environ.get("MODELS_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ia", "models"))
RESULTS = os.environ.get("RESULTS_DIR",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
REF_FILE = os.environ.get("DRIFT_REF",
                          os.path.join(RESULTS, "raw", "drift_reference.npy"))
DRIFT_PSI_THRESHOLD = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.2"))
RETRAIN_ACC_DROP = float(os.environ.get("RETRAIN_ACC_DROP_POINTS", "3.0"))
TEAM_ID = os.environ.get("TEAM_ID", "XX")

app = FastAPI(title="cloud-leak-service")
_lock = threading.Lock()
_retraining = False
_ingest_q = queue.Queue()


def _ingest_worker():
    con = db()
    while True:
        item = _ingest_q.get()
        if item is None:
            break
        ts, kind, node, payload, m = item
        try:
            con.execute("INSERT INTO measures(ts, kind, node, payload) VALUES (?,?,?,?)",
                        (ts, kind, node, payload))
            if kind == "localization":
                con.execute("INSERT INTO localizations(ts, topk, probs, mv) VALUES (?,?,?,?)",
                            (ts, json.dumps(m.get("topk", [])),
                             json.dumps(m.get("probs", [])),
                             m.get("model_version", "")))
            con.commit()
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] {e}", flush=True)


threading.Thread(target=_ingest_worker, daemon=True).start()


def db():
    con = sqlite3.connect(CLOUD_DB, check_same_thread=False, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")  # demo buffer: avoid sandbox fsync artifact (documented)
    return con


def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS measures(
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, kind TEXT, node TEXT, payload TEXT);
    CREATE TABLE IF NOT EXISTS localizations(
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, topk TEXT, probs TEXT, mv TEXT);
    CREATE TABLE IF NOT EXISTS model_versions(
      version TEXT PRIMARY KEY, created_at INTEGER, metrics_json TEXT, path TEXT);
    CREATE TABLE IF NOT EXISTS drift_log(
      ts INTEGER, drift_score REAL, threshold REAL, recommend INTEGER);
    """)
    con.commit()
    con.close()


init_db()


@app.post("/api/mesures")
def recevoir(m: dict):
    ts = int(m.get("ts", time.time() * 1000))
    kind = m.get("kind", "telemetry")
    node = str(m.get("node", m.get("id", "")))
    # async ingest: storage is off the request path (decision latency),
    # same design as the fog service writer thread
    _ingest_q.put((ts, kind, node, json.dumps(m), m))
    return {"ok": True}


@app.get("/api/modele/latest")
def modele_latest():
    path = os.path.join(MODELS_DIR, "fog_model_latest.joblib")
    if not os.path.exists(path):
        raise HTTPException(404, "no model registered")
    con = db()
    row = con.execute("SELECT version FROM model_versions ORDER BY created_at DESC LIMIT 1").fetchone()
    con.close()
    version = row[0] if row else "unversioned"
    return FileResponse(path, media_type="application/octet-stream",
                        filename="model_fog_latest.joblib",
                        headers={"x-model-version": version})


@app.get("/api/modele/versions")
def modele_versions():
    con = db()
    rows = con.execute("SELECT version, created_at, metrics_json FROM model_versions"
                       " ORDER BY created_at DESC").fetchall()
    con.close()
    return [dict(version=v, created_at=c, metrics=json.loads(m or "{}")) for v, c, m in rows]


@app.post("/api/retrain")
def retrain():
    """Idempotent retraining trigger (brief Section 8). Logs to results/."""
    global _retraining
    with _lock:
        if _retraining:
            return {"status": "already_running"}
        _retraining = True

    def run():
        global _retraining
        log = os.path.join(RESULTS, "raw", "retrain_log.txt")
        try:
            out = subprocess.run(
                [sys.executable,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
                 "--register", "--db", CLOUD_DB],
                capture_output=True, text=True, timeout=1800)
            with open(log, "a") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(out.stdout + out.stderr)
        finally:
            _retraining = False

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/drift")
def drift():
    """Drift score: PSI between training reference and recent residuals."""
    ref = None
    if os.path.exists(REF_FILE):
        ref = np.load(REF_FILE)
    con = db()
    rows = con.execute("SELECT payload FROM measures WHERE kind='telemetry' "
                       "ORDER BY id DESC LIMIT 2000").fetchall()
    con.close()
    if ref is None or not rows:
        return {"drift_score": None, "note": "not enough data (need reference + recent)"}
    recent = []
    for (pl,) in rows:
        d = json.loads(pl)
        if "dp" in d:
            recent.append([float(d.get("p", 0)), float(d["dp"]), float(d.get("q", 0)),
                           float(d.get("vib", 0))])
    if len(recent) < 30:
        return {"drift_score": None, "note": "not enough recent rows"}
    res = psi_score(ref, np.array(recent))
    recommend = int(res["drift_score"] > DRIFT_PSI_THRESHOLD)
    con = db()
    con.execute("INSERT INTO drift_log VALUES (?,?,?,?)",
                (int(time.time() * 1000), res["drift_score"], DRIFT_PSI_THRESHOLD, recommend))
    con.commit()
    con.close()
    return {"drift_score": res["drift_score"], "threshold": DRIFT_PSI_THRESHOLD,
            "retrain_recommended": bool(recommend), "n_recent": len(recent),
            "criterion": f"PSI>{DRIFT_PSI_THRESHOLD} or accuracy drop > {RETRAIN_ACC_DROP} pts"}


@app.get("/api/health")
def health():
    con = db()
    n_meas = con.execute("SELECT COUNT(*) FROM measures").fetchone()[0]
    con.close()
    return {"status": "online", "team": TEAM_ID, "measures": n_meas}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
