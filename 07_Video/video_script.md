# 07_Video - demo video (3-5 min) - HUMAN records

> Agent brief Section 15: the agent cannot record the video. This is the
> storyboard/script with exact commands and expected outputs.

## Storyboard (4 min)

| # | Time | Screen | Narration |
|---|---|---|---|
| 1 | 0:00-0:20 | Title slide | "Edge-triggered, Fog-localized, Cloud-adapted leak detection for water networks - Equipe XX." |
| 2 | 0:20-0:50 | Architecture slide | One-paragraph problem + why Edge/Fog/Cloud. |
| 3 | 0:50-1:40 | Wokwi simulation | Start the ESP32 sim: serial shows WiFi connect, window inference `y=0/1 conf=... inf_us=...`, MQTT publishes. Point at the alert LED on leak. |
| 4 | 1:40-2:20 | Terminal `mosquitto_sub -v` | Show telemetry/alert topics flowing, then the Fog publishing `fog/localization` top-3 candidates. |
| 5 | 2:20-3:00 | Streamlit dashboard | Page 1 live network: node colours, top-3 highlighted, alert table; Page 3 placement comparison bars. |
| 6 | 3:00-3:30 | Failure injection | `docker stop mosquitto` -> node buffers (serial), `docker start mosquitto` -> flush. Cloud outage -> Fog buffer -> recovery. |
| 7 | 3:30-4:00 | Conclusion slide | Recommendation: detection at Edge, localization at Fog, training at Cloud - with the measured numbers. |

## Commands shown in the demo

```bash
# Wokwi: open the published project link, press "Start simulation"
# (REPLAY_TEST variant prints: Accuracy on the Edge: x.xxx (N=500))

# broker log
mosquitto_sub -h <broker> -t "m2i2a/equipeXX/#" -v

# fog + cloud + dashboard (or: docker compose up)
python 02_Code_Source/fog/fog/service.py
uvicorn cloud_api:app --port 8000
streamlit run 02_Code_Source/fog/dashboard/app.py

# failure injection (nominal demo)
docker stop mosquitto ; sleep 20 ; docker start mosquitto
docker stop cloud ; sleep 30 ; docker start cloud
```

## Recording tips

- 1080p, terminal font >= 14 pt, hide personal files/tabs.
- The dashboard has a **demo mode** (stored data) as a fallback if the live
  stack fails during recording (never show a broken pipeline).
- Upload to Google Drive/YouTube (unlisted) and put the link in
  `07_Video/link.txt`; the ZIP keeps only this text file (20 MB limit).
- Every member must be able to explain the shown code during the defense.
