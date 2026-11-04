# 03_Simulation - reproducible simulation evidence (E8)

## Public links (fill in and verify BEFORE submission - Annex D)

- Wokwi project: <TO BE FILLED - publish the Wokwi project and paste the link>
- GitHub repository: <TO BE FILLED>
- Google Drive (large files/video): <TO BE FILLED>

> The links must open without logging in (human verifies - agent brief Section 15).

## Contents

| File | Purpose |
|---|---|
| `diagram.json` | Wokwi wiring: ESP32 DevKit C + alert LED on GPIO2 (sensors are virtual/replayed) |
| `sketch/` | The exact firmware sources (`edge_node.ino`, `model.h`, `test_data.h`, `sensor_stream.h`) |
| `screens/` | Screenshots: Wokwi serial monitor, Streamlit dashboard pages, security proof |
| `logs/` | `mosquitto_sub -v` transcript, Docker logs, emulated-node QoS1 test output |
| `links.txt` | Same links as above (kept for the grading checklist) |

## How to run the Wokwi simulation

1. Create a new ESP32 project in the Wokwi web editor.
2. Paste `sketch/edge_node.ino` (and upload `model.h`, `test_data.h`, `sensor_stream.h`
   as additional files in the Wokwi project file tree).
3. Libraries: `PubSubClient` (Nick O'Leary) and `ArduinoJson` (Benoit Blanchon) - add
   via the Wokwi library manager (`wokwi.toml` lists them for the CLI route).
4. Start the simulation. The node connects to `Wokwi-GUEST` (channel 6) and publishes
   JSON telemetry to `m2i2a/equipeXX/edge/node3/telemetry` on `broker.emqx.io`.
5. Watch the serial monitor: window inference results with `micros()` timings.

### Accuracy replay (X1)

Compile with `-DREPLAY_TEST` (or uncomment `#define REPLAY_TEST` in the sketch):
the node replays `test_data.h` (the held-out TEST split, balanced 250/250) and prints:

```
Accuracy on the Edge: x.xxx (N=500)
avg inference: xxx.x us [I]
```

### End-to-end chain proof (Annex F.2)

```
# terminal 1: watch the broker
mosquitto_sub -h broker.emqx.io -t "m2i2a/equipeXX/#" -v | tee logs/mosquitto_sub.log

# terminal 2: Fog service (02_Code_Source/fog)
python fog/service.py

# terminal 3: Cloud API (02_Code_Source/cloud)
uvicorn cloud_api:app --port 8000
```

The Edge (Wokwi or `emulated_node.py`) -> Mosquitto -> Fog (localization) ->
Cloud (REST) chain is demonstrated in `logs/` and in the video (07_Video).

## Sensor reality statement (Annex F.1)

The three sensors (pressure, flow, vibration) are **virtual**: the firmware
replays a recorded scenario column (`sensor_stream.h`, generated from the
dataset). This is explicitly allowed by Annex F.2 provided the dataset source
and generation are described - see `04_Donnees/dataset_card.md`. The vibration
channel is synthetic (correlated with the leak flow); this limitation is stated
in the report's "Limits of the simulation" box.
