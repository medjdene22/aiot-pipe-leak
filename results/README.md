# results/ - measured outputs (evidence for the report)

Every table/figure in `01_Rapport/report.pdf` maps to a file here. Re-running
`02_Code_Source/ia/experiments_live.py` and `experiments_static.py`
(SEED=42) reproduces them.

## Included in the submission ZIP

- `experiments_x*.csv|json` (X1-X11 tables)
- `edge_detector_metrics.json`, `edge_detector_table.csv`
- `fog_localizer_metrics.json`, `fog_localizer_table.csv`, `sensitivity_matrix.csv`,
  `sensor_subsets.json`
- `consistency_check.json`, `placement_comparison.csv`, `robustness_timeline.csv`,
  `drift_series.csv`, `run_meta.json`
- `raw/latency_reps.csv`, `raw/live_s3_rtt.csv`, `raw/fog_buffer_series.csv`

## Excluded from the ZIP (20 MB rule) - regenerate with the training scripts

| File(s) | Size | Regenerate with |
|---|---|---|
| `raw/edge_*.joblib`, `fog_model_*.joblib` | 10-350 MB each | `ia/train_edge_detector.py`, `ia/train_fog_localizer.py` |
| `raw/edge_model_predictions.csv` | 9 MB | `ia/train_edge_detector.py` |
| `raw/x5_fog.db`, `raw/x5_cloud.db` | <1 MB | `ia/experiments_live.py` |
| `04_Donnees/generated/` (full dataset) | ~150 MB | `python 04_Donnees/generate_data.py` |

The scikit-learn pickles are large because the RF is fit on ~2M string-labelled
rows; the *deployed* artifacts are the C header (`edge/model.h`, 115 KB) and the
per-k joblibs rebuilt by `cloud/train.py`.
