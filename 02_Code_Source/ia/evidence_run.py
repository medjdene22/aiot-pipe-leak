#!/usr/bin/env python3
"""Evidence run (Annex F.2): one end-to-end chain pass with a verbose subscriber
transcript, emulated nodes (incl. QoS1 + offline-buffer demo), fog + cloud.

Writes:
  03_Simulation/logs/broker_transcript.log   (mosquitto_sub -v style)
  03_Simulation/logs/emulated_node_qos1.json (PUBACK statistics)
  03_Simulation/logs/emulated_node_outage.json
"""
import json
import os
import subprocess
import sys
import threading
import time

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGS = os.path.join(REPO, "03_Simulation", "logs")
sys.path.insert(0, HERE)
import experiments_live as el  # noqa: E402

PORT = el.PORT
out = open(os.path.join(LOGS, "broker_transcript.log"), "w")
out.write(f"# verbose MQTT transcript ({time.strftime('%Y-%m-%d %H:%M:%S')}) "
          "# amqtt sandbox broker; deliverable uses Mosquitto (docker-compose)\n")
lock = threading.Lock()


def on_msg(cli, u, msg):
    with lock:
        out.write(f"{msg.topic} {msg.payload.decode()[:160]}\n")
        out.flush()


def main():
    el.stack_up()
    sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="evidence-sub")
    sub.on_message = on_msg
    sub.connect("127.0.0.1", PORT)
    sub.subscribe("m2i2a/equipeXX/#", qos=1)
    sub.loop_start()
    time.sleep(1)

    env = dict(os.environ, MQTT_HOST="127.0.0.1", MQTT_PORT=str(PORT),
               MQTT_USER="", TEAM_ID="XX")
    env['MQTT_PASSWORD'] = os.environ.get('MQTT_PASSWORD', '')
    node = os.path.join(REPO, "02_Code_Source", "edge", "emulated_node.py")

    print("run 1: QoS1 alert test")
    r = subprocess.run([sys.executable, node, "--node", "3", "--scenario", "249",
                        "--qos-test"], env=env, capture_output=True, text=True)
    open(os.path.join(LOGS, "emulated_node_qos1.log"), "w").write(
        r.stdout + r.stderr)

    print("run 2: offline buffer + flush")
    r = subprocess.run([sys.executable, node, "--node", "4", "--scenario", "250",
                        "--drop-after", "5", "--drop-sec", "8"],
                       env=env, capture_output=True, text=True)
    open(os.path.join(LOGS, "emulated_node_outage.log"), "w").write(
        r.stdout + r.stderr)

    time.sleep(2)
    sub.loop_stop()
    sub.disconnect()
    out.close()
    el.stack_down()
    print("evidence logs written")


if __name__ == "__main__":
    main()
