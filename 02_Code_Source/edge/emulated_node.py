#!/usr/bin/env python3
"""
Emulated Edge node (paho-mqtt) - brief Section 3 "Simulation topology" and
Annex F.4 option E.

Replays one node's data column from the generated dataset and publishes the
same JSON telemetry as the ESP32 firmware, but with FULL MQTT QoS support:
  telemetry  QoS0 (bandwidth, loss tolerable)
  alert      QoS1 (must not be lost)   <-- PubSubClient cannot publish QoS1;
             the QoS1 behaviour of E5 is demonstrated with this script
  status     QoS1 retained + LWT

Usage:
  python emulated_node.py --node 15 --scenario 0 --speed 0
  python emulated_node.py --node 16 --scenario 0 --speed 0 --qos-test
  python emulated_node.py --node 17 --scenario 0 --drop-after 10 --drop-sec 30
        (outage injection: stop publishing 30 s after 10 messages, buffer, resume)

Environment: MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASSWORD, TEAM_ID (or flags).
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "04_Donnees", "generated")
sys.path.append(os.path.join(HERE, "..", "ia"))
from common import load_residuals, load_windows  # noqa: E402

TEAM_ID = os.environ.get("TEAM_ID", "XX")
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", f"edge_{TEAM_ID}")
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', "")
MODEL_VERSION = os.environ.get("MV", "edge-v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, help="node id, e.g. 15")
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.0,
                    help="seconds between windows (0 = as fast as possible)")
    ap.add_argument("--host", default=MQTT_HOST)
    ap.add_argument("--port", type=int, default=MQTT_PORT)
    ap.add_argument("--user", default=MQTT_USER)
    ap.add_argument("--password", default=os.environ.get("MQTT_PASSWORD", ""))
    ap.add_argument("--qos-test", action="store_true",
                    help="publish alerts at QoS1 and verify PUBACK flow")
    ap.add_argument("--drop-after", type=int, default=0,
                    help="simulate outage after N messages")
    ap.add_argument("--drop-sec", type=float, default=30.0)
    args = ap.parse_args()

    w = load_windows()
    rows = w[(w.scenario_id == args.scenario) & (w.node == str(args.node))].sort_values("w_idx")
    if rows.empty:
        raise SystemExit(f"no rows for scenario={args.scenario} node={args.node}")

    prefix = f"m2i2a/equipe{TEAM_ID}"
    t_telem = f"{prefix}/edge/node{args.node}/telemetry"
    t_alert = f"{prefix}/edge/node{args.node}/alert"
    t_status = f"{prefix}/edge/node{args.node}/status"
    t_cmd = f"{prefix}/edge/commande/modele"

    stats = {"sent": 0, "alerts": 0, "puback": 0, "buffered": 0, "flushed": 0}
    offline_buffer = []

    def on_publish(client, userdata, mid, reason_code=None, properties=None):
        stats["puback"] += 1

    def on_message(client, userdata, msg):
        print(f"[cmd] {msg.topic}: {msg.payload.decode()[:120]}", flush=True)

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                      client_id=f"emu-edge-{TEAM_ID}-{args.node}")
    if args.user:
        cli.username_pw_set(args.user, args.password or None)
    cli.will_set(t_status, json.dumps({"status": "offline", "ts": int(time.time()*1000),
                                       "id": int(args.node)}), qos=1, retain=True)
    cli.on_publish = on_publish
    cli.on_message = on_message
    cli.connect(args.host, args.port, keepalive=30)
    cli.subscribe(t_cmd, qos=1)
    cli.loop_start()
    cli.publish(t_status, json.dumps({"status": "online", "ts": int(time.time()*1000),
                                      "id": int(args.node)}), qos=1, retain=True)

    outage_done = False
    for _, row in rows.iterrows():
        ts = int(time.time() * 1000)
        payload = json.dumps({
            "id": int(args.node), "ts": ts,
            "p": round(float(row.p_mean), 2),
            "dp": round(float(row.dp_mean), 2),
            "q": round(float(row.q_mean), 2),
            "vib": round(float(row.vib_rms), 4),
            "y": int(row.y_global), "conf": 0.91, "mv": MODEL_VERSION,
        })
        in_outage = (args.drop_after and stats["sent"] >= args.drop_after
                     and not outage_done)
        if in_outage:
            # outage: buffer locally, do not publish
            offline_buffer.append((t_telem, payload))
            stats["buffered"] += 1
            if stats["buffered"] == 1:
                print(f"[outage] start ({args.drop_sec}s)", flush=True)
                time.sleep(args.drop_sec)
                print("[outage] end - flushing buffer", flush=True)
                for topic, p in offline_buffer:
                    cli.publish(topic, p, qos=0)
                    stats["flushed"] += 1
                offline_buffer.clear()
                outage_done = True
        else:
            cli.publish(t_telem, payload, qos=0)
            if int(row.y_local) == 1:
                qos = 1 if args.qos_test else 0
                info = cli.publish(t_alert, payload, qos=qos)
                if qos == 1:
                    info.wait_for_publish(timeout=5)
                stats["alerts"] += 1
            stats["sent"] += 1
        if args.speed > 0:
            time.sleep(args.speed)

    time.sleep(1.0)   # let PUBACKs arrive
    cli.publish(t_status, json.dumps({"status": "offline", "ts": int(time.time()*1000),
                                      "id": int(args.node)}), qos=1, retain=True)
    cli.loop_stop()
    cli.disconnect()
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
