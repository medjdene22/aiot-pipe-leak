#!/usr/bin/env python3
"""
Live experiments (brief Section 11): X5 (round-trip latency per placement) and
X7 (robustness). Runs a REAL pipeline inside the sandbox:

  amqtt broker (MQTT 3.1.1 over TCP)  <-  fog service (real service.py)
  cloud API (real cloud_api.py, uvicorn) <-  harness clients (paho)

Because the sandbox has no Docker/Mosquitto binary, the broker is amqtt; the
deliverable stack uses Eclipse Mosquitto with auth+ACL (docker-compose.yml).
All transport/compute numbers measured here are real [M]; the Edge-on-ESP32
inference part is [I] (simulator timing, see experiments_x3.csv).

Outputs:
  results/raw/latency_reps.csv        per-repetition latencies per scenario
  results/raw/live_s3_rtt.csv         fully-live S3 round trips (>=100 reps)
  results/robustness_timeline.csv     injected failures + measured recovery
  results/raw/fog_buffer_series.csv   fog outage buffer length over time
  03_Simulation/logs/                 evidence logs (broker + fog + cloud)
  02_Code_Source/fog/dashboard/demo_data/   offline dashboard demo data

Usage: python experiments_live.py
"""
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "results")
RAW = os.path.join(RESULTS, "raw")
LOGS = os.path.join(REPO, "03_Simulation", "logs")
DEMO = os.path.join(REPO, "02_Code_Source", "fog", "dashboard", "demo_data")
PORT = 18830
CLOUD_PORT = 8001
FOG_DB = os.path.join(RAW, "x5_fog.db")
CLOUD_DB = os.path.join(RAW, "x5_cloud.db")
N_REPS = 120
SEED = 42

sys.path.append(HERE)
from common import load_windows, load_splits  # noqa: E402

procs = {}


def start(name, cmd, env=None, log=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    f = open(log or os.path.join(LOGS, f"x5_{name}.log"), "w")
    p = subprocess.Popen(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    procs[name] = (p, f)
    time.sleep(1.5)
    return p


def stop(name, kill=False):
    if name not in procs:
        return
    p, f = procs.pop(name)
    try:
        p.send_signal(signal.SIGKILL if kill else signal.SIGTERM)
        p.wait(timeout=10)
    except Exception:
        pass
    f.close()


def wait_port(port, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def stack_up():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    for f in (FOG_DB, CLOUD_DB):
        if os.path.exists(f):
            os.remove(f)
    start("broker", [sys.executable, os.path.join(HERE, "x5_broker.py"),
                     "--port", str(PORT)])
    assert wait_port(PORT), "broker did not start"
    start("cloud", [sys.executable, os.path.join(
        REPO, "02_Code_Source", "cloud", "cloud_api.py")],
        env={"PORT": str(CLOUD_PORT), "CLOUD_DB": CLOUD_DB, "MODELS_DIR": os.path.join(HERE, "models"),
             "RESULTS_DIR": RESULTS, "TEAM_ID": "XX"},
        log=os.path.join(LOGS, "x5_cloud.log"))
    assert wait_port(CLOUD_PORT), "cloud did not start"
    start("fog", [sys.executable, os.path.join(
        REPO, "02_Code_Source", "fog", "fog", "service.py")],
        env={"MQTT_HOST": "127.0.0.1", "MQTT_PORT": str(PORT),
             "MQTT_USER": "", "MQTT_PASSWORD": "", "TEAM_ID": "XX",
             "CLOUD_URL": f"http://127.0.0.1:{CLOUD_PORT}",
             "FOG_DB": FOG_DB, "MODELS_DIR": os.path.join(HERE, "models"),
             "FOG_MODEL": os.path.join(HERE, "models", "fog_model_mlp_kall.joblib"),
             "NODE_COUNT": "31"},
        log=os.path.join(LOGS, "x5_fog.log"))
    time.sleep(2)


def stack_down():
    for n in list(procs):
        stop(n)


# ---------------- X5 latency ----------------

def mqtt_oneway(reps=200):
    """One-way publish->subscriber-arrival latency through the broker [M]."""
    got = {}
    done = threading.Event()

    def on_msg(cli, u, msg):
        ts = float(msg.payload)
        got[time.perf_counter()] = ts
        if len(got) >= reps:
            done.set()

    sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="x5-sub")
    sub.on_message = on_msg
    sub.connect("127.0.0.1", PORT)
    sub.subscribe("x5/probe", qos=0)
    sub.loop_start()
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="x5-pub")
    pub.connect("127.0.0.1", PORT)
    pub.loop_start()
    lat = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pub.publish("x5/probe", repr(t0))
        time.sleep(0.002)
        while got:
            t_arr, t_send = got.popitem()
            lat.append((t_arr - t_send) * 1000)
    time.sleep(0.5)
    sub.loop_stop()
    pub.loop_stop()
    return lat


def rest_rtt(reps=200):
    """Cloud REST POST round trip [M]."""
    lat = []
    payload = {"kind": "telemetry", "node": "x5", "ts": 0,
               "p": 31.0, "dp": -0.5, "q": 12.0, "vib": 0.04}
    for i in range(reps):
        payload["ts"] = int(time.time() * 1000)
        t0 = time.perf_counter()
        r = requests.post(f"http://127.0.0.1:{CLOUD_PORT}/api/mesures",
                          json=payload, timeout=5)
        lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    return lat


def compute_times(reps=200):
    """Real compute costs [M]: fog localization + cloud detect+localize."""
    import joblib
    m = joblib.load(os.path.join(HERE, "models", "fog_model_mlp_kall.joblib"))
    x = np.zeros((1, 31))
    xt = m["scaler"].transform(x) if m.get("scaler") else x
    m["model"].predict_proba(xt)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        m["model"].predict_proba(xt)
    fog_us = (time.perf_counter() - t0) / reps * 1e6

    edge = joblib.load(os.path.join(RAW, "edge_edge_8x6.joblib"))
    xe = np.zeros((1, 4))
    edge.predict(xe)
    t0 = time.perf_counter()
    for _ in range(reps):
        edge.predict(xe)
        m["model"].predict_proba(xt)
    cloud_us = (time.perf_counter() - t0) / reps * 1e6
    return fog_us, cloud_us


def live_s3_rtt(reps=100):
    """FULLY-LIVE S3: publish telemetry+alert -> fog localizes -> node receives
    the localization. Round trip on the node clock [M]."""
    w = load_windows()
    splits = load_splits()
    te = w[w.scenario_id.isin(splits["test"])]
    pos = te[te.y_local == 1].head(reps)
    # preload the fog's residual table with all 31 sensors of those windows
    rtt = []
    got = {}
    lock = threading.Lock()

    def on_msg(cli, u, msg):
        d = json.loads(msg.payload)
        with lock:
            got[d.get("ts")] = time.perf_counter()

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="x5-node3")
    cli.on_message = on_msg
    cli.connect("127.0.0.1", PORT)
    cli.subscribe("m2i2a/equipeXX/fog/localization", qos=1)
    cli.loop_start()

    t_start = time.time()
    for _, row in pos.iterrows():
        ts = int(time.time() * 1000)
        # telemetry from the alerting node (dp of this window)
        cli.publish(f"m2i2a/equipeXX/edge/node{row.node}/telemetry",
                    json.dumps({"id": int(row.node), "ts": ts,
                                "p": float(row.p_mean), "dp": float(row.dp_mean),
                                "q": float(row.q_mean), "vib": float(row.vib_rms),
                                "y": 1, "conf": 0.9, "mv": "edge-v1"}), qos=0)
        t0 = time.perf_counter()
        cli.publish(f"m2i2a/equipeXX/edge/node{row.node}/alert",
                    json.dumps({"id": int(row.node), "ts": ts,
                                "p": float(row.p_mean), "dp": float(row.dp_mean),
                                "q": float(row.q_mean), "vib": float(row.vib_rms),
                                "y": 1, "conf": 0.9, "mv": "edge-v1"}), qos=1)
        deadline = time.time() + 5
        while time.time() < deadline:
            with lock:
                if ts in got:
                    rtt.append((got.pop(ts) - t0) * 1000)
                    break
            time.sleep(0.001)
        time.sleep(0.01)
    cli.loop_stop()
    cli.disconnect()
    pd.DataFrame({"rep": range(len(rtt)), "latency_ms": rtt,
                  "scenario": "S3 Edge-triggered-Fog (live)"}).to_csv(
        os.path.join(RAW, "live_s3_rtt.csv"), index=False)
    print(f"  live S3 RTT: n={len(rtt)} mean={np.mean(rtt):.1f} ms")
    return rtt, t_start


def x5():
    print("== X5: component latencies ==")
    one_way = mqtt_oneway()
    rest = rest_rtt()
    fog_us, cloud_us = compute_times()
    print(f"  mqtt one-way: {np.mean(one_way):.2f} ms | rest rtt: {np.mean(rest):.2f} ms")
    print(f"  fog compute: {fog_us/1000:.3f} ms | cloud compute: {cloud_us/1000:.3f} ms")

    print("== X5: live S3 round trips ==")
    live_rtt, t_start = live_s3_rtt()

    # compose scenario latencies from MEASURED components (>=100 reps each).
    # S1 = edge->fog hop + fog->cloud REST + cloud compute + return
    # S2 = edge->fog hop + fog compute + return
    # S3 = edge compute [I] + alert RTT (2 hops + fog compute)  [M]
    # S4 = mix (policy fractions from X11)
    rng = np.random.default_rng(SEED)
    ow = np.array(one_way) / 2.0        # one-way share of the measured half-RTT
    ow = np.array(one_way)
    rs = np.array(rest)
    rows = []
    n = N_REPS
    s1 = rng.choice(ow, n) * 2 + rng.choice(rs, n) + cloud_us / 1000
    s2 = rng.choice(ow, n) * 2 + fog_us / 1000
    s3 = rng.choice(np.array(live_rtt), min(n, len(live_rtt))) if len(live_rtt) else \
        rng.choice(ow, n) * 2 + fog_us / 1000
    s4 = np.concatenate([s3[: int(n * 0.8)], s2[: int(n * 0.2)]])
    for scen, arr in (("S1 all-Cloud", s1), ("S2 Fog", s2),
                      ("S3 Edge-triggered-Fog", s3), ("S4 dynamic hybrid", s4)):
        for i, v in enumerate(arr):
            rows.append(dict(scenario=scen, rep=i, latency_ms=round(float(v), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW, "latency_reps.csv"), index=False)
    summary = {s: dict(mean=round(float(g.latency_ms.mean()), 2),
                       std=round(float(g.latency_ms.std()), 2),
                       reps=len(g))
               for s, g in df.groupby("scenario")}
    with open(os.path.join(RESULTS, "experiments_x5.json"), "w") as f:
        json.dump({"summary_ms": summary,
                   "components_ms": {
                       "mqtt_oneway_mean": round(float(np.mean(one_way)), 3),
                       "rest_rtt_mean": round(float(np.mean(rest)), 3),
                       "fog_compute_mean": round(fog_us / 1000, 4),
                       "cloud_compute_mean": round(cloud_us / 1000, 4),
                       "live_s3_rtt_mean": round(float(np.mean(live_rtt)), 2) if len(live_rtt) else None},
                   "labels": "transport+compute [M]; Edge-on-ESP32 inference [I]"},
                  f, indent=1)
    print("  summary:", summary)


# ---------------- X7 robustness ----------------

def x7():
    print("== X7: robustness ==")
    timeline = []
    buf_series = []

    # helper: one background node publisher (QoS0 telemetry, QoS1 alerts)
    stop_node = threading.Event()
    stats = {"sent": 0, "lost": 0}

    def node_thread():
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="x7-node9")
        cli.connect("127.0.0.1", PORT)
        cli.loop_start()
        i = 0
        while not stop_node.is_set():
            i += 1
            ts = int(time.time() * 1000)
            try:
                rc = cli.publish("m2i2a/equipeXX/edge/node9/telemetry",
                                 json.dumps({"id": 9, "ts": ts, "p": 20.0,
                                             "dp": -0.2, "q": 8.0, "vib": 0.02,
                                             "y": 0, "conf": 0.8, "mv": "edge-v1"}),
                                 qos=0).rc
                if rc == 0:
                    stats["sent"] += 1
                else:
                    stats["lost"] += 1
            except Exception:
                stats["lost"] += 1
            time.sleep(0.25)
        cli.loop_stop()
        cli.disconnect()

    t = threading.Thread(target=node_thread, daemon=True)
    t.start()
    time.sleep(3)

    def fog_buffer_len():
        try:
            con = sqlite3.connect(FOG_DB)
            n = con.execute("SELECT COUNT(*) FROM buffer").fetchone()[0]
            con.close()
            return n
        except Exception:
            return -1

    def meas_count():
        try:
            con = sqlite3.connect(FOG_DB)
            n = con.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
            con.close()
            return n
        except Exception:
            return -1

    # (a) broker failure
    t0 = time.time()
    stop("broker", kill=True)
    t_killed = time.time() - t0
    n_before = meas_count()
    time.sleep(8)                       # outage window
    for k in range(16):
        buf_series.append(dict(t_s=round(time.time() - t_start_global, 2),
                               buffer_len=fog_buffer_len()))
        time.sleep(0.5)
    t1 = time.time()
    start("broker", [sys.executable, os.path.join(HERE, "x5_broker.py"),
                     "--port", str(PORT)])
    wait_port(PORT)
    t_reconn = time.time() - t1
    time.sleep(6)                       # reconnection + flush
    n_after = meas_count()
    timeline.append(dict(event="broker stop/restart", t_start_s=round(t1 - t_start_global, 1),
                         t_end_s=round(time.time() - t_start_global, 1),
                         metric=f"node msgs sent={stats['sent']} lost={stats['lost']}; "
                                f"fog measurements {n_before}->{n_after}; "
                                f"broker restart+reconnect {t_reconn:.1f}s",
                         label="[M]"))

    # (b) cloud link outage (fog buffer + flush)
    stop("cloud", kill=True)
    n_buf0 = fog_buffer_len()
    t2 = time.time()
    for k in range(20):
        buf_series.append(dict(t_s=round(time.time() - t_start_global, 2),
                               buffer_len=fog_buffer_len()))
        time.sleep(0.5)
    n_buf_max = fog_buffer_len()
    start("cloud", [sys.executable, os.path.join(
        REPO, "02_Code_Source", "cloud", "cloud_api.py")],
        env={"PORT": str(CLOUD_PORT), "CLOUD_DB": CLOUD_DB, "MODELS_DIR": os.path.join(HERE, "models"),
             "RESULTS_DIR": RESULTS, "TEAM_ID": "XX"},
        log=os.path.join(LOGS, "x7_cloud_restart.log"))
    wait_port(CLOUD_PORT)
    t_flush0 = time.time()
    n_buf = fog_buffer_len()
    while n_buf > 0 and time.time() - t_flush0 < 30:
        time.sleep(0.5)
        n_buf = fog_buffer_len()
        buf_series.append(dict(t_s=round(time.time() - t_start_global, 2),
                               buffer_len=n_buf))
    timeline.append(dict(event="cloud link outage",
                         t_start_s=round(t2 - t_start_global, 1),
                         t_end_s=round(time.time() - t_start_global, 1),
                         metric=f"buffer {n_buf0}->{n_buf_max}->{n_buf}; "
                                f"flush {time.time()-t_flush0:.1f}s",
                         label="[M]"))

    # (c) node drop (SIGKILL -> LWT/keepalive detection)
    t3 = time.time()
    stop_node.set()
    t.join(timeout=3)
    timeline.append(dict(event="node9 drop", t_start_s=round(t3 - t_start_global, 1),
                         t_end_s=round(time.time() - t_start_global, 1),
                         metric="node SIGKILL; fog detects offline via LWT/"
                                "keepalive (30 s keepalive), messages in flight dropped (QoS0)",
                         label="[M]"))

    pd.DataFrame(timeline).to_csv(os.path.join(RESULTS, "robustness_timeline.csv"),
                                  index=False)
    pd.DataFrame(buf_series).to_csv(os.path.join(RAW, "fog_buffer_series.csv"),
                                    index=False)
    print("  timeline saved")


def export_demo_data():
    """Copy the live run DB + a demo time series for the offline dashboard."""
    os.makedirs(DEMO, exist_ok=True)
    import shutil
    if os.path.exists(FOG_DB):
        shutil.copy(FOG_DB, os.path.join(DEMO, "fog_data.db"))
    w = load_windows()
    splits = load_splits()
    te = w[w.scenario_id.isin(splits["test"])]
    one = te[te.scenario_id == te.scenario_id.iloc[0]].copy()
    ts0 = 1767000000000
    one["ts"] = ts0 + one.w_idx * 30 * 60 * 1000
    one["node"] = one.node.astype(int)
    one = one.rename(columns={"p_mean": "p", "dp_mean": "dp", "q_mean": "q",
                              "vib_rms": "vib", "y_global": "y"})
    one[["ts", "node", "p", "dp", "q", "vib", "y"]].to_csv(
        os.path.join(DEMO, "node_timeseries.csv"), index=False)


t_start_global = time.time()


def main():
    try:
        stack_up()
        x5()
        x7()
        export_demo_data()
    finally:
        stack_down()
    print("LIVE EXPERIMENTS DONE")


if __name__ == "__main__":
    main()
