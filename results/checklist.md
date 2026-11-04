# Final checklist (agent brief Section 16) - run date 2026-11-03

| # | Item | Status | Note |
|---|---|---|---|
| 1 | ZIP opens, structure matches, size <= 20 MB, no .exe, no password | PASS | `test2.zip` 2.28 MB, 109+ files, all folders 00-09 + results. Rename to `SEI2027_Equipe-XX.zip` at submission (human) |
| 2 | Wokwi / GitHub / Drive links present and public | **FAIL -> HUMAN** | `03_Simulation/links.txt` has placeholders; human publishes and verifies no-login access |
| 3 | README reproduces the work from a clean install | PASS | root `README.md` + `02_Code_Source/README.md` (exact commands) |
| 4 | No secrets; `.env.example` present | PASS | `.env` excluded from ZIP; passwords only via env; checked no credentials in sources |
| 5 | >= 3 FreeRTOS tasks, queue/semaphore, timer interrupt, timing diagram | PASS | `edge_node.ino` (acq/infer/com + timer ISR + 2 queues + semaphore), `01_Rapport/figures/fig_timing.png` |
| 6 | >= 3 sensors, preprocessing, timestamped JSON | PASS | p/q/vib + MA filter + normalization + dp; JSON contract in report 3.3 |
| 7 | MQTT topics + QoS justified, REST, auth + ACL proof | PARTIAL -> HUMAN | topics/QoS/REST documented and transcripted; Mosquitto auth+ACL config + `setup_auth.sh` shipped - the refusal proof needs Docker (sandbox had no Mosquitto; run locally and paste logs) |
| 8 | Storage + dashboard with screenshots | PASS | SQLite (fog+cloud), Streamlit 6 pages, `03_Simulation/screens/page1..6` verified headings |
| 9 | Baseline + advanced models, compression with loss, >= 2 layers, monitoring + retraining | PASS | threshold/RF/MLP; compression < 0.5 pt; Edge C + Fog joblib + Cloud training; PSI drift loop |
| 10 | S1-S3 (+S4), >= 100 latency reps, [M]/[I]/[E], recommendation | PASS | `results/` X5-X9 + `placement_comparison.csv`; 100-120 reps/scenario; recommendation in report 6.7 |
| 11 | "Limits of the simulation" box | PASS | report 6.10 |
| 12 | >= 10 references, verified or flagged | PASS | 16 references; 5 marked **[UNVERIFIED]** -> HUMAN verifies |
| 13 | Initial + final Gantt, contribution table, AI declaration | PASS (templates) | Gantt PDFs + deviation analysis done; `00_Equipe/` needs real names (HUMAN) |
| 14 | Report 15-25 pp excl. annexes; 3-5 min video; slides | PARTIAL -> HUMAN | report 16 pp (body ~15) in `01_Rapport/report.pdf`; slides 13; video = script ready, HUMAN records |

## Human action list (agent cannot do these)

1. Fill team names, percentages, commit counts; sign the AI declaration (`00_Equipe/`).
2. Verify/replace the 5 [UNVERIFIED] references.
3. Publish Wokwi project (paste `03_Simulation/sketch/` sources + headers) and
   GitHub/Drive links; verify they open without login.
4. Run Wokwi `REPLAY_TEST`, save the serial output (`Accuracy on the Edge: ...`)
   to `03_Simulation/logs/`; fill the [I] row of `results/experiments_x3.csv`.
5. Run the Docker stack + `mosquitto_passwd`/ACL refusal proof (Section 7 of
   the report describes the expected log); paste into `03_Simulation/logs/`.
6. Record the 3-5 min video (`07_Video/video_script.md`), put the link in `07_Video/link.txt`.
7. Send the ZIP to r.djerbi@univ-boumerdes.dz, subject
   `[M2-I2A][SEI][PROJET]-Equipe XX`, before 3 Jan 2027 23:59 (single version!).
