"""
Streamlit dashboard (E6) - water-network leak detection & localization.

6 pages (brief Section 10):
  1 Live network    2 Node detail    3 Placement comparison
  4 Robustness      5 Model & drift  6 Sensor sweep & ablations

Data sources:
  - Fog SQLite DB (FOG_DB env, default fog_data.db): measurements/alerts/localizations
  - results/*.json|csv produced by the experiment scripts (02_Code_Source/ia)
  - Cloud API (CLOUD_URL) for model versions/drift when available

DEMO MODE: works fully offline - if the Fog DB is missing, the dashboard falls
back to stored demo data (dashboard/demo_data/) so the defense can run without
any infrastructure (brief Section 10: demo mode).

Run: streamlit run app.py
"""
import json
import os
import sqlite3
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Leak AIoT dashboard", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "results")
FOG_DB = os.environ.get("FOG_DB", os.path.join(HERE, "demo_data", "fog_data.db"))
CLOUD_URL = os.environ.get("CLOUD_URL", "http://localhost:8000")
DEMO_DIR = os.path.join(HERE, "demo_data")

st.sidebar.title("Edge-Fog-Cloud leak AIoT")
st.sidebar.caption("Hanoi benchmark network - Equipe-XX")
page = st.sidebar.radio("Page", [
    "1 Live network", "2 Node detail", "3 Placement comparison",
    "4 Robustness", "5 Model & drift", "6 Sensor sweep & ablations"])
st.sidebar.markdown("---")
demo_mode = not os.path.exists(FOG_DB)
st.sidebar.info("DEMO MODE (offline data)" if demo_mode else f"DB: {FOG_DB}")


# ---------------- helpers ----------------

@st.cache_data(ttl=5)
def load_table(query, params=()):
    if not os.path.exists(FOG_DB):
        return pd.DataFrame()
    con = sqlite3.connect(FOG_DB)
    try:
        df = pd.read_sql_query(query, con, params=params)
    finally:
        con.close()
    return df


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def hanoi_graph():
    """Static Hanoi topology from the committed .inp (always available)."""
    G = nx.Graph()
    try:
        import wntr
        wn = wntr.network.WaterNetworkModel(os.path.join(REPO, "04_Donnees", "Hanoi.inp"))
        for _, p in wn.pipes():
            G.add_edge(p.start_node_name, p.end_node_name)
    except Exception:
        # fallback: pipe list from the dataset adjacency
        G.add_edges_from([("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"),
                          ("6", "7"), ("7", "8"), ("8", "9"), ("9", "10"), ("10", "11"),
                          ("3", "12"), ("12", "13"), ("13", "14"), ("14", "15"),
                          ("15", "16"), ("16", "17"), ("4", "18"), ("18", "19"),
                          ("19", "20"), ("20", "21"), ("5", "22"), ("22", "23"),
                          ("23", "24"), ("6", "25"), ("25", "26"), ("26", "27"),
                          ("7", "28"), ("28", "29"), ("29", "30"), ("30", "31"),
                          ("8", "32"), ("32", "25"), ("12", "18"), ("13", "23")])
    return G


G = hanoi_graph()

# ---------------- Page 1: live network ----------------
if page.startswith("1"):
    st.header("1 - Live network")

    alerts = load_table("SELECT ts, node, conf, mv FROM alerts ORDER BY ts DESC LIMIT 20")
    locs = load_table("SELECT ts, topk, probs, model_version FROM localizations ORDER BY ts DESC LIMIT 1")
    meas = load_table("SELECT node, p, dp FROM measurements ORDER BY ts DESC LIMIT 400")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active alerts (last 20 shown)", int(len(alerts)))
    nodes_seen = load_table("SELECT COUNT(DISTINCT node) AS n FROM measurements")
    c2.metric("Nodes online", int(nodes_seen["n"][0]) if len(nodes_seen) else 0)
    c3.metric("Localizations", int(len(load_table("SELECT id FROM localizations"))))
    if len(locs):
        c4.metric("Last localization", f"node {json.loads(locs.topk[0])[0]}" if locs.topk[0] else "-")

    col_graph, col_table = st.columns([3, 2])
    with col_graph:
        st.subheader("Network state (dp: pressure deviation)")
        dp = {}
        if len(meas):
            g = meas.groupby("node")["dp"].mean()
            dp = {str(k): float(v) for k, v in g.items()}
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        colors = [dp.get(n, 0.0) for n in G.nodes]
        nx.draw(G, pos, ax=ax, node_color=colors, cmap="coolwarm",
                node_size=420, with_labels=True, font_size=8)
        topk = json.loads(locs.topk[0]) if len(locs) else []
        if topk:
            nx.draw_networkx_nodes(G, pos, nodelist=[str(t) for t in topk[:3]],
                                   node_color="gold", node_size=650, ax=ax)
        ax.set_title("blue = low dp (possible leak), red = high dp; gold = top-3 candidates")
        st.pyplot(fig)
    with col_table:
        st.subheader("Last alerts")
        if len(alerts):
            alerts["time"] = pd.to_datetime(alerts.ts, unit="ms")
            st.dataframe(alerts[["time", "node", "conf", "mv"]], use_container_width=True)
        else:
            st.write("No alerts yet." if not demo_mode else "Demo mode: run the pipeline to see alerts.")

# ---------------- Page 2: node detail ----------------
elif page.startswith("2"):
    st.header("2 - Node detail")
    node = st.selectbox("Node", [str(i) for i in range(2, 33)])
    df = load_table("SELECT ts, p, dp, q, vib, y FROM measurements WHERE node=? ORDER BY ts",
                    (node,))
    if df.empty:
        demo = load_csv(os.path.join(DEMO_DIR, "node_timeseries.csv"))
        if not demo.empty:
            df = demo[demo.node.astype(str) == node]
            st.caption("DEMO MODE: stored time series")
    if df.empty:
        st.warning("No data for this node.")
    else:
        df["time"] = pd.to_datetime(df.ts, unit="ms")
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        for ax, col in zip(axes, ["p", "dp", "q", "vib"]):
            if col in df:
                ax.plot(df["time"], df[col], lw=1)
                ax.set_ylabel(col)
        if "y" in df:
            axes[-1].plot(df["time"], df["y"], "r.", ms=2, label="edge decision")
            axes[-1].legend()
        st.pyplot(fig)

# ---------------- Page 3: placement comparison (core result) ----------------
elif page.startswith("3"):
    st.header("3 - Inference placement comparison (S1-S4)")
    st.markdown(
        "S1 all-Cloud | S2 Fog | S3 Edge-triggered-Fog | S4 dynamic hybrid. "
        "Labels: **[M]** measured, **[I]** indicative (simulator), **[E]** estimated.")
    tab = load_csv(os.path.join(RESULTS, "placement_comparison.csv"))
    if tab.empty:
        st.warning("placement_comparison.csv not found - run experiments first.")
    else:
        st.dataframe(tab, use_container_width=True)
        metric = st.selectbox("Bar chart metric", [c for c in tab.columns
                                                   if c not in ("scenario", "label")])
        if metric:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(tab.iloc[:, 0].astype(str), tab[metric])
            ax.set_ylabel(metric)
            st.pyplot(fig)
        st.caption("Latency distributions (>= 100 reps/scenario):")
        lat = load_csv(os.path.join(RESULTS, "raw", "latency_reps.csv"))
        if not lat.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            groups = [g["latency_ms"].values for _, g in lat.groupby("scenario")]
            ax.boxplot(groups, labels=[g["scenario"].iloc[0] for _, g in lat.groupby("scenario")])
            ax.set_ylabel("end-to-end latency (ms)")
            st.pyplot(fig)

# ---------------- Page 4: robustness ----------------
elif page.startswith("4"):
    st.header("4 - Robustness")
    st.markdown("Injected failures: broker stop, node drop, Cloud link outage.")
    rob = load_csv(os.path.join(RESULTS, "robustness_timeline.csv"))
    if rob.empty:
        st.warning("robustness_timeline.csv not found - run X7 first.")
    else:
        st.dataframe(rob, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for _, row in rob.iterrows():
            ax.axvspan(row["t_start_s"], row["t_end_s"], alpha=0.3,
                       label=row["event"])
        ax.set_xlabel("time (s)")
        ax.set_title("failure timeline")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        st.pyplot(fig)
        buf = load_csv(os.path.join(RESULTS, "raw", "fog_buffer_series.csv"))
        if not buf.empty:
            st.line_chart(buf.set_index("t_s")["buffer_len"])

# ---------------- Page 5: model & drift ----------------
elif page.startswith("5"):
    st.header("5 - Model versions & drift")
    vers = load_json(os.path.join(RESULTS, "cloud_train_log.jsonl"), None)
    tab = pd.DataFrame([json.loads(l) for l in open(
        os.path.join(RESULTS, "cloud_train_log.json"))] if os.path.exists(
        os.path.join(RESULTS, "cloud_train_log.json")) else [])
    if tab.empty:
        st.info("No retraining log yet (results/cloud_train_log.json).")
    else:
        st.dataframe(tab, use_container_width=True)
    st.subheader("Current versions")
    st.write(load_json(os.path.join(RESULTS, "model_versions.json"),
                       {"edge": "edge-v1", "fog": "fog-v1"}))
    st.subheader("Drift score")
    drift = load_csv(os.path.join(RESULTS, "drift_series.csv"))
    if drift.empty:
        st.info("Run X10 (drift experiment) to populate this chart.")
    else:
        st.line_chart(drift.set_index("t_s")[["drift_score", "threshold"]])

# ---------------- Page 6: sensor sweep & ablations ----------------
elif page.startswith("6"):
    st.header("6 - Sensor sweep & ablations")
    m = load_json(os.path.join(RESULTS, "fog_localizer_metrics.json"), {})
    sweep = m.get("sensor_sweep", {})
    if not sweep:
        st.warning("fog_localizer_metrics.json not found - run training first.")
    else:
        rows = []
        for k, e in sweep.items():
            rows.append({"k": k,
                         "RF top-1": e.get("rf", {}).get("top1"),
                         "RF top-3": e.get("rf", {}).get("top3"),
                         "MLP top-1": e.get("mlp", {}).get("top1"),
                         "MLP top-3": e.get("mlp", {}).get("top3"),
                         "baseline top-1": e.get("baseline_cosine", {}).get("top1"),
                         "RF ATD (hops)": e.get("rf", {}).get("atd"),
                         "MLP ATD (hops)": e.get("mlp", {}).get("atd")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.line_chart(pd.DataFrame(rows).set_index("k")[["RF top-1", "MLP top-1", "baseline top-1"]])

        st.subheader("Leak-size ablation (best model)")
        abl = m.get("leak_size_ablation", {})
        if abl:
            st.bar_chart(pd.DataFrame(abl).T["top1"])
        st.caption("ATD = average topological distance (pipe hops) between "
                   "predicted and true leak node.")
