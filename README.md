# SEI2027_Equipe-XX - Edge-Fog-Cloud water-network leak detection

Course project: Embedded Systems and IoT (UEM), Master 2 I2A, UMBB Boumerdes
2026-2027. Theme: **water-network leak detection and localization** on the
Hanoi benchmark (31 nodes), fully simulated (no hardware required).

## What is where

| Folder | Content |
|---|---|
| `00_Equipe/` | Team sheet + contribution table + generative-AI declaration (HUMAN fills names) |
| `01_Rapport/` | `report.pdf` (15-25 pp. scientific format), `report.md` source, figures, renderer |
| `02_Code_Source/` | `edge/` (ESP32 firmware + TinyML model), `fog/` (MQTT/ACL + service + dashboard), `cloud/` (FastAPI + training), `ia/` (data/training/experiments) + README |
| `03_Simulation/` | Wokwi `diagram.json` + sketch, links, screenshots, evidence logs |
| `04_Donnees/` | `generate_data.py` (1,320 scenarios), `dataset_card.md`, samples |
| `05_Planification/` | Gantt initial/final + deviation analysis |
| `06_Presentation/` | `slides.pptx` (+ generator) |
| `07_Video/` | demo video script + link placeholder (HUMAN records) |
| `08_Startup/` | startup track (not chosen; skeleton only) |
| `results/` | every measured number in the report maps to a file here |

## Quick start (reproduce everything)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r 02_Code_Source/requirements.txt
python 04_Donnees/generate_data.py          # ~3 min, SEED=42
python 02_Code_Source/ia/train_edge_detector.py
python 02_Code_Source/ia/train_fog_localizer.py
python 02_Code_Source/ia/consistency_check.py
python 02_Code_Source/ia/experiments_live.py
python 02_Code_Source/ia/experiments_static.py
python 01_Rapport/render_report.py
streamlit run 02_Code_Source/fog/dashboard/app.py   # demo mode works offline
```

Full instructions and the Docker stack: `02_Code_Source/README.md`.

## Before submission (HUMAN checklist)

- [ ] Fill team names/percentages (`00_Equipe/`), verify the AI declaration
- [ ] Verify the 5 references marked **[UNVERIFIED]** in the report
- [ ] Publish Wokwi project + GitHub/Drive links (`03_Simulation/links.txt`)
- [ ] Run the Wokwi serial replay, paste `Accuracy on the Edge` output into
      `03_Simulation/logs/` (and the `micros()` timing into `results/experiments_x3.csv`)
- [ ] Record the 3-5 min video (`07_Video/video_script.md`), put the link in `07_Video/link.txt`
- [ ] Rename the archive to `SEI2027_Equipe-XX.zip` (equipe number) if required,
      send to r.djerbi@univ-boumerdes.dz with subject
      `[M2-I2A][SEI][PROJET]-Equipe XX` before 3 Jan 2027 23:59
