#!/usr/bin/env python3
"""Generate report figures from results/ (01_Rapport/figures/)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "01_Rapport", "figures")
os.makedirs(FIGS, exist_ok=True)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    box(0.02, 0.62, 0.26, 0.3, "EDGE (x31)\nESP32 + FreeRTOS\ntaskAcq|taskInfer|taskCom\nTinyML detector (RF 8x6, C)", "#cfe8ff")
    box(0.02, 0.08, 0.26, 0.28, "EDGE model.h\ntree vote confidence\nreplay test_data.h", "#e8f4ff")
    box(0.37, 0.30, 0.26, 0.34, "FOG (Docker)\nMosquitto auth+ACL\nfog/service.py\nlocalization (MLP/RF)\nSQLite + outage buffer", "#ffe6cc")
    box(0.71, 0.30, 0.27, 0.34, "CLOUD (FastAPI)\ntraining + registry\ndrift detection (PSI)\nmodel versions\n/api/mesures /api/modele", "#d9f2d9")

    ax.annotate("", xy=(0.37, 0.55), xytext=(0.28, 0.72),
                arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.31, 0.68, "MQTT QoS0/QoS1\ntelemetry+alerts", fontsize=7.5)
    ax.annotate("", xy=(0.28, 0.22), xytext=(0.37, 0.38),
                arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.29, 0.27, "params (retained)\ncommande/modele", fontsize=7.5)
    ax.annotate("", xy=(0.71, 0.55), xytext=(0.63, 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(0.645, 0.58, "REST /api/mesures\n(buffered)", fontsize=7.5)
    ax.annotate("", xy=(0.63, 0.42), xytext=(0.71, 0.42),
                arrowprops=dict(arrowstyle="->", lw=1.6, linestyle="--"))
    ax.text(0.635, 0.35, "GET /api/modele/latest\n(model pull)", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_architecture.png"), dpi=150)
    plt.close(fig)


def fig_timing():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_ylim(3.4, -0.6)
    ax.set_xlim(0, 12)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["timer ISR\n(1 s period)", "taskAcq\nprio 3", "taskInfer\nprio 2", "taskCom\nprio 1"], fontsize=8)
    ax.set_xlabel("time (s)", fontsize=9)
    for s in range(12):
        ax.add_patch(plt.Rectangle((s, -0.35), 0.12, 0.3, fc="red", ec="none"))
    ax.text(0.2, -0.5, "ISR: xSemaphoreGiveFromISR", fontsize=7, color="red")
    for s in range(12):
        ax.add_patch(plt.Rectangle((s + 0.15, 0.7), 0.35, 0.35, fc="#2c7fb8", ec="none"))
    ax.text(0.15, 1.12, "read 3 sensors + MA filter -> queue", fontsize=7)
    for s in range(0, 12, 2):
        ax.add_patch(plt.Rectangle((s + 0.55, 1.7), 0.5, 0.35, fc="#41ab5d", ec="none"))
    ax.text(0.55, 2.12, "window 6 steps -> clf.predict (micros)", fontsize=7)
    for s in range(0, 12, 2):
        ax.add_patch(plt.Rectangle((s + 1.1, 2.7), 0.7, 0.35, fc="#e8a33d", ec="none"))
    ax.text(1.1, 3.12, "MQTT publish + mqtt.loop() + ring buffer", fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_timing.png"), dpi=150)
    plt.close(fig)


def fig_latency():
    p = os.path.join(RESULTS, "raw", "latency_reps.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4))
    order = ["S1 all-Cloud", "S2 Fog", "S3 Edge-triggered-Fog", "S4 dynamic hybrid"]
    data = [df[df.scenario == s].latency_ms.values for s in order]
    ax.boxplot(data, tick_labels=[s.split()[0] for s in order], showmeans=True)
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_xlabel("placement scenario")
    ax.set_title("Round-trip latency per placement (>=100 reps; network+fog [M], edge [I])")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_latency.png"), dpi=150)
    plt.close(fig)


def fig_sensor_sweep():
    p = os.path.join(RESULTS, "fog_localizer_metrics.json")
    if not os.path.exists(p):
        return
    m = json.load(open(p))
    rows = []
    for k, e in m["sensor_sweep"].items():
        rows.append((k, e["rf"]["top1"], e["mlp"]["top1"], e["baseline_cosine"]["top1"]))
    order = {"2": 0, "3": 1, "5": 2, "8": 3, "all": 4}
    rows.sort(key=lambda r: order[r[0]])
    ks = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ks, [r[1] for r in rows], "o-", label="RandomForest")
    ax.plot(ks, [r[2] for r in rows], "s-", label="MLP")
    ax.plot(ks, [r[3] for r in rows], "^-", label="sensitivity-cosine baseline")
    ax.set_xlabel("number of pressure sensors k")
    ax.set_ylabel("top-1 localization accuracy (test)")
    ax.set_title("Localization vs sensor count (31 candidate nodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sensorsweep.png"), dpi=150)
    plt.close(fig)


def fig_drift():
    p = os.path.join(RESULTS, "drift_series.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(df.t_s / 3600, df.drift_score, "o-", label="PSI drift score")
    ax.axhline(0.2, color="r", ls="--", label="retraining threshold")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("drift score (max PSI)")
    ax.set_title("Drift detection (demand-pattern shift) and retraining trigger")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_drift.png"), dpi=150)
    plt.close(fig)


def fig_drift_effect():
    p = os.path.join(RESULTS, "experiments_x10.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = [d["fog_top1_old_model_on_drift"], d["fog_top1_retrained_on_drift"]]
    ax.bar(["fog model v1\n(stale)", "retrained\non drifted data"], bars,
           color=["#d9534f", "#5cb85c"])
    for i, v in enumerate(bars):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("top-1 localization accuracy (drifted data)")
    ax.set_title("Effect of drift-triggered retraining")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_drift_effect.png"), dpi=150)
    plt.close(fig)


def fig_compression():
    p = os.path.join(RESULTS, "edge_detector_table.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(df.variant, df.test_f1, color=["#9ecae1", "#3182bd", "#08519c"])
    for i, v in enumerate(df.test_f1):
        ax.text(i, v + 0.0004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.96, 0.985)
    ax.set_ylabel("test F1 (detection)")
    ax.set_title("Edge model compression: accuracy loss < 0.5 pt")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_compression.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_timing()
    fig_latency()
    fig_sensor_sweep()
    fig_drift()
    fig_drift_effect()
    fig_compression()
    print("figures saved to 01_Rapport/figures/")
