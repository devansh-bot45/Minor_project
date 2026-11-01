# Model evaluation on repository recordings

Generated: 2026-10-28T05:07:57+00:00

- Windows: 60 samples at 20 Hz (resampled from the recording timestamps), 50% overlap; gaps > 1 s in a recording are skipped.
- 'model' = pure model argmax; 'pipeline' = model + rules + per-session smoothing (what the dashboard shows).
- Recordings with used_in_training=true were part of the training data, so their scores are optimistic.
- Model v2: trained by train_model_v2.py on windows from prepare_data_v2.py (WISDM, Heterogeneity and UCI HAR converted to phone units at 20 Hz, plus the still, jogging and walking1 phone recordings).
- stairs1 and stairs2 are ONE stair-climbing session recorded on two devices at the same time, so they are not independent test data.
- Results on public-dataset subjects that were never used for training: model_v2/comparison.md.

| Recording | True label | In training? | Windows | Model acc. | Pipeline acc. | Old pipeline acc. |
|---|---|---|---|---|---|---|
| still data.csv | Still | yes | 103 | 100.0% | 100.0% | 98.1% |
| jogging.csv | Jogging | yes | 64 | 100.0% | 100.0% | 100.0% |
| sensor_recording_20260426_162439.csv | Walking | yes | 63 | 100.0% | 100.0% | 100.0% |
| sensor_recording_20260427_174501.csv | Walking | no | 48 | 87.5% | 81.2% | 79.2% |
| sensor_recording_20260426_145530.csv | Stairs | no | 63 | 61.9% | 84.1% | 59.4% |
| sensor_recording_20260426_145547.csv | Stairs | no | 54 | 27.8% | 16.7% | 22.2% |
| eating.csv | unlabelled | - | 62 | - | - | - |

**Seen In Training**: 230 windows, model 100.0%, pipeline 100.0%

**Held Out**: 165 windows, model 58.2%, pipeline 61.2%

Prediction distribution per recording (pipeline):

- still data.csv: {'Still': 103}
- jogging.csv: {'Jogging': 64}
- sensor_recording_20260426_162439.csv: {'Walking': 63}
- sensor_recording_20260427_174501.csv: {'Jogging': 5, 'Stairs': 4, 'Walking': 39}
- sensor_recording_20260426_145530.csv: {'Sports': 3, 'Stairs': 53, 'Uncertain': 7}
- sensor_recording_20260426_145547.csv: {'Hand Activity': 5, 'Stairs': 9, 'Still': 5, 'Walking': 23, 'Uncertain': 12}
- eating.csv: {'Hand Activity': 1, 'Sports': 8, 'Still': 20, 'Uncertain': 33}
