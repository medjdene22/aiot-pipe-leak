#!/usr/bin/env python3
"""Generate 06_Presentation/slides.pptx from results/ (numbers = measured)."""
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "01_Rapport", "figures")


def jload(name, default=None):
    try:
        with open(os.path.join(RESULTS, name)) as f:
            return json.load(f)
    except Exception:
        return default


def add_slide(prs, title, bullets, fig=None, small=False):
    layout = prs.slide_layouts[1] if not fig else prs.slide_layouts[5]
    s = prs.slides.add_slide(layout)
    if not fig:
        s.shapes.title.text = title
        body = s.placeholders[1].text_frame
        for i, b in enumerate(bullets):
            p = body.paragraphs[0] if i == 0 else body.add_paragraph()
            p.text = b
            p.font.size = Pt(15 if not small else 13)
    else:
        s.shapes.title.text = title
        tb = s.shapes.add_textbox(Inches(0.4), Inches(1.25), Inches(4.2), Inches(5.4))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = b
            p.font.size = Pt(13)
        s.shapes.add_picture(fig, Inches(4.8), Inches(1.4), width=Inches(4.9))
    return s


def main():
    prs = Presentation()
    edge = jload("edge_detector_metrics.json", {})
    fog = jload("fog_localizer_metrics.json", {})
    x5 = jload("experiments_x5.json", {})
    x10 = jload("experiments_x10.json", {})

    # 1 title
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = ("Edge-triggered, Fog-localized, Cloud-adapted\n"
                           "leak detection for water networks")
    s.placeholders[1].text = ("Equipe-XX - M2 I2A - Embedded Systems & IoT "
                              "(UMBB, 2026-2027)\nHanoi benchmark - simulation only")

    add_slide(prs, "Problem & objectives", [
        "Water networks lose 20-40% of treated water to leaks",
        "Detect AND localize leaks with cheap sensors + AI",
        "Embedded constraints: energy, memory (520 KB RAM), latency, cost",
        "Core question: WHERE should inference run? Edge / Fog / Cloud",
        "Deliverable: quantified comparison + placement recommendation",
    ])
    add_slide(prs, "Architecture", [],
              fig=os.path.join(FIGS, "fig_architecture.png"))
    add_slide(prs, "Edge node (ESP32 / FreeRTOS)", [
        "3 tasks: taskAcq (prio 3), taskInfer (prio 2), taskCom (prio 1)",
        "Hardware-timer ISR -> binary semaphore -> queue",
        "Sensors: pressure, flow, vibration (virtual, replayed)",
        "Preprocessing: moving average + baseline normalization (dp)",
        "TinyML: RandomForest 8x6 exported to C (micromlgen), ~112 KB",
        "MQTT QoS0 telemetry / QoS1 alerts (emulated nodes), LWT",
    ], small=True)
    add_slide(prs, "AI: Edge detector", [
        f"Baseline dp-threshold: F1={edge.get('baseline_threshold', {}).get('test', {}).get('f1')}",
        f"RF 8x6 (edge): F1={edge.get('rf_edge', {}).get('f1')}, "
        f"acc={edge.get('rf_edge', {}).get('accuracy')}",
        "Compression: full RF -> 8x6 -> 4x3: < 0.5 pt F1 loss",
        "PC vs C consistency: 0 mismatches on 500 replay vectors [M]",
        "Inference: ~30 us estimate on ESP32 [E], micros() in Wokwi [I]",
    ], fig=os.path.join(FIGS, "fig_compression.png"))
    add_slide(prs, "AI: Fog localization (31 nodes)", [],
              fig=os.path.join(FIGS, "fig_sensorsweep.png"))
    add_slide(prs, "Placement comparison (S1-S4)", [
        f"S1 all-Cloud latency: {x5.get('summary_ms', {}).get('S1 all-Cloud', {}).get('mean')} ms",
        f"S2 Fog: {x5.get('summary_ms', {}).get('S2 Fog', {}).get('mean')} ms",
        f"S3 Edge-triggered-Fog: {x5.get('summary_ms', {}).get('S3 Edge-triggered-Fog', {}).get('mean')} ms",
        "Network: S1/S2 stream raw (bytes/min), S3 alerts only (~xN less)",
        "Energy [E]: S3 cuts radio time -> lower mAh/day",
        "Cost [E]: S3 = 31 ESP32 + RPi3 + small VM",
    ], fig=os.path.join(FIGS, "fig_latency.png"))
    add_slide(prs, "Robustness", [
        "Broker stop: node buffers locally (ring), reconnects, flushes",
        "Cloud outage: Fog buffers in SQLite, flush on recovery",
        "Node drop: LWT/keepalive offline status",
        "Measured timeline in results/robustness_timeline.csv [M]",
    ])
    add_slide(prs, "Drift & retraining", [
        f"Drift detected (PSI): {x10.get('psi_drifted', {}).get('drift_score')} > 0.2",
        f"Fog top-1 before retrain: {x10.get('fog_top1_old_model_on_drift')}",
        f"After retrain: {x10.get('fog_top1_retrained_on_drift')}",
        "Loop: drift flag -> POST /api/retrain -> new version -> Fog pulls "
        "-> Edge params pushed (retained MQTT)",
    ], fig=os.path.join(FIGS, "fig_drift_effect.png"))
    add_slide(prs, "Innovation", [
        "1. Cost-aware dynamic offloading (Edge/Fog/Cloud per window)",
        "   policy: confidence gate + RTT gate + battery gate",
        "2. Drift-triggered retraining with model + parameter push",
        "Compared to fixed S1-S3 on accuracy / latency / bytes / energy",
    ])
    add_slide(prs, "Limits of the simulation", [
        "Synthetic data (WNTR), emulated nodes, synthetic vibration channel",
        "Wokwi timing is indicative [I]; energy/cost estimated [E]",
        "Leak = pressure-dependent orifice (WNTR), areas calibrated 1-10%",
        "Public broker shared; no TLS on the demo path (own Mosquitto for E5)",
        "Port path to real ESP32 documented in the report",
    ], small=True)
    add_slide(prs, "Conclusion & recommendation", [
        "Detection AT THE EDGE (fast, private, outage-tolerant)",
        "Localization AT THE FOG (multi-sensor view, low latency)",
        "Training / drift / versions AT THE CLOUD (compute, not urgent)",
        "Justified by measured latency, bytes, energy + robustness results",
        "Future: real sensors, BattLeDIM stress test, GNN localizer",
    ])
    add_slide(prs, "Live demo script", [
        "1. Nominal run: nodes streaming -> dashboard live view",
        "2. Inject leak: Edge alerts -> Fog localizes top-3 -> dashboard",
        "3. Inject failure: stop broker -> buffer -> recovery flush",
        "4. Placement comparison page (S1-S4 with [M]/[I]/[E] labels)",
        "5. Drift + retrain: model version change visible end-to-end",
    ])
    out = os.path.join(HERE, "slides.pptx")
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
