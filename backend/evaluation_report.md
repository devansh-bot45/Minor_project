# Model evaluation on repository recordings

Generated: 2026-10-28T05:08:54+00:00

- Windows: 60 samples at 20 Hz (resampled from the recording timestamps), 50% overlap; gaps > 1 s in a recording are skipped.
- 'model' = pure model argmax; 'pipeline' = model + rules + per-session smoothing (what the dashboard shows).
- Recordings with used_in_training=true were part of the training data, so their scores are optimistic.
- The public-dataset test split from training cannot be reproduced (datasets and X_all.npy are not in the repo).

| Recording | True label | In training? | Windows | Model acc. | Pipeline acc. | Old pipeline acc. |
|---|---|---|---|---|---|---|
| still data.csv | Still | yes | 103 | 100.0% | 100.0% | 98.1% |
| jogging.csv | Jogging | yes | 64 | 100.0% | 100.0% | 100.0% |
| sensor_recording_20260426_162439.csv | Walking | yes | 63 | 100.0% | 100.0% | 100.0% |
| sensor_recording_20260427_174501.csv | Walking | yes | 48 | 100.0% | 100.0% | 91.7% |
| sensor_recording_20260426_145530.csv | Stairs | no | 63 | 0.0% | 0.0% | 0.0% |
| sensor_recording_20260426_145547.csv | Stairs | no | 54 | 0.0% | 0.0% | 0.0% |
| eating.csv | unlabelled | - | 62 | - | - | - |

**Seen In Training**: 278 windows, model 100.0%, pipeline 100.0%

**Held Out**: 117 windows, model 0.0%, pipeline 0.0%

Prediction distribution per recording (pipeline):

- still data.csv: {'Still': 103}
- jogging.csv: {'Jogging': 64}
- sensor_recording_20260426_162439.csv: {'Walking': 63}
- sensor_recording_20260427_174501.csv: {'Walking': 48}
- sensor_recording_20260426_145530.csv: {'Jogging': 50, 'Walking': 7, 'Uncertain': 6}
- sensor_recording_20260426_145547.csv: {'Jogging': 3, 'Still': 9, 'Walking': 36, 'Uncertain': 6}
- eating.csv: {'Jogging': 31, 'Still': 21, 'Uncertain': 10}
