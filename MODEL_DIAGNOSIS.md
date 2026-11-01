# Diagnosis: why the model fails on held-out Stairs data

Date: 2026-10-27 · Model diagnosed: `backend/har_model.keras` (unchanged)
Backup of the current artifacts with SHA-256 checksums: `backend/model_backup_2026-10-27/`
Scripts that reproduce every number below: `backend/diagnostics/`

## 1. Measured metrics of the current model

### Held-out phone recordings (2 Stairs files, 117 windows, not used in training)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Stairs | 0.000 (no window was predicted Stairs) | **0.000** | **0.000** | 117 |

Confusion (model argmax): true Stairs → Jogging 62, Walking 50, Still 5, Stairs 0.
Average Stairs probability: 0.33 and 0.13 on the two files.

### All labelled phone recordings (395 windows; 278 of them were also training data)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Jogging | 0.508 | 1.000 | 0.674 | 64 |
| Stairs | 0.000 | 0.000 | 0.000 | 117 |
| Still | 0.954 | 1.000 | 0.976 | 103 |
| Walking | 0.689 | 1.000 | 0.816 | 111 |
| Accuracy | | | 0.704 | 395 |

Hand Activity and Sports cannot be measured: there is no labelled phone recording for them.
Low precision for Jogging and Walking comes entirely from Stairs windows predicted as those classes.

### UCI HAR public dataset (used, partly, in training)

| Input format | Split | Stairs recall | Stairs F1 | Walking recall | Still recall |
|---|---|---|---|---|---|
| Exactly as `prepare_data.py` built training windows | train (7,352) | 0.030 | 0.053 | 0.001 | 0.995 |
| Exactly as `prepare_data.py` built training windows | test (2,947) | 0.080 | 0.129 | 0.000 | 0.996 |
| Converted to phone units, true 20 Hz, 3 s windows | test (2,338) | 0.000 | 0.000 | 0.043 | 0.974 |

UCI Stairs and Walking windows are predicted mostly as **Jogging**. So the model does not recognise Stairs
even on data from one of its own training sources.

## 2. Causes, in order of impact

**a. The phone ("custom") Stairs training data were 100% mislabelled.**
The original `prepare_data.py` labelled `sensor_recording_20260424_131838.csv` and
`sensor_recording_20260424_184254.csv` as Stairs. These files are byte-identical to `still data.csv` and
`jogging.csv`. That gave 584 Stairs windows: 296 of Still data and 288 of Jogging data. The same windows
were also in training as Still and Jogging. No real phone Stairs recording was used at all.
Evidence: on the Still recording the model gives Stairs 0.41 average probability, and on the Jogging
recording also 0.41. That is the signature of identical inputs carrying two labels.

**b. Public datasets were mixed with inconsistent units and time scales.**
- UCI HAR was fed as `body_acc`, which is in units of *g* with gravity already removed. The phone and
  WISDM/Heterogeneity data are in m/s² with gravity included. Average |body acceleration| per window:
  UCI as trained ≈ 0.31–0.34 for Stairs, against ≈ 3.0 for the phone Stairs recordings (about 10× apart).
- UCI windows are 128 samples at 50 Hz (2.56 s) but were squeezed into 60 samples, so the model treats
  them as 3 s windows. Step frequencies therefore look about 15% slower than they are.
- Heterogeneity data is decimated by taking every 10th row. Its phones record at different rates, so the
  effective rate is not a consistent 20 Hz. This comes from reading the code; the dataset is not in the
  repo, so it was not measured here.
- The custom recordings were windowed by row rather than resampled to 20 Hz. A 60-row window spans
  2.7–4.1 s depending on the file.

**c. Stairs was deliberately under-represented.**
`balance_dataset` targets 1,500 Stairs windows against 3,500 for every other class, and three real Stairs
recordings were commented out ("Reduced Stairs data"). This was likely a reaction to Stairs being
over-predicted, which cause (a) itself produced. Class weights partly compensate.

**d. Preprocessing mismatch at inference: ruled out.**
With windows built exactly as training built them (raw rows, no resampling), the held-out Stairs files are
still misclassified: 1 of 126 windows correct. The original server's causal filter also gave 0%. The
failure is in the model and its training data, not in the app's pipeline.

**e. Too little real data.**
The repo has about 3.3 minutes of real phone Stairs recordings, all from what appears to be one session.
There are no phone recordings for Hand Activity or Sports.

## 3. Data leakage in the original evaluation (`train_model.py`)

- The split was a random `train_test_split` on windows *after* augmentation. The custom data is
  7,662 windows built from only 258 unique raw windows (25–30 augmented copies each). In a simulation of
  that split, **100%** of custom test windows had the same raw window, or an augmented copy, in training.
- All custom "Stairs" test windows were Still/Jogging data that also appeared in training under another
  label.
- Windows overlap by 50% and there is no split by subject or recording, so neighbouring windows from the
  same person sit on both sides of the split.
- The StandardScaler was fitted on all data before the split (minor leakage).

Any accuracy produced by this procedure overstates how the model does on new people or recordings.

## 4. Are the published accuracy figures supported?

| Figure | Where it appears | Reproducible? |
|---|---|---|
| **88.6 %** test accuracy (4-class table: Walking, Still, Jogging, Stairs; Stairs recall 0.57) | original `README.md` (now replaced); `Project_Synopsis_Hinglish.md` (line 16) | **No.** `X_all.npy`, `report.txt` and `confusion_matrix.png` are not in the repo (they are git-ignored). The table has 4 classes but the deployed model has 6. It was produced by the leaky split above. |
| **82.99 %** test accuracy, 7 classes incl. "Active Hands" and one unnamed class, 2,833 test windows | `report_part1.md` (line 23), `report_part4.md` (Chapter 8), `Minor_Project_Report_COMPLETE.md` (lines 23, 862 and following) | **No.** It describes a different model (Conv1D + LSTM, 7 classes). The deployed model is Conv1D + BiLSTM + Attention with 6 classes. Its Stairs F1 of 0.87 is contradicted by the measurements above. |
| — | `Minor Project format - final 2026.docx` | Blank template, no results. |

Correction: `IMPLEMENTATION_STATUS.md` previously said the report files quote 88.6%. They quote 82.99%;
88.6% appears in the old README and the synopsis.

## 5. Is retraining justified?

**Yes, but only after the data and evaluation are fixed.** Retraining the same pipeline would reproduce
the same problems. Changes needed:

1. **Labels:** done in `prepare_data.py` (duplicates removed, the real Stairs recordings enabled).
2. **Units and resampling:** build UCI from `total_acc × 9.80665` (m/s², gravity included) plus gyro,
   resampled 50 → 20 Hz with real 3 s windows. Resample Heterogeneity and the custom CSVs by timestamp to
   20 Hz (for example with `preprocessing.parse_sensor_csv`) instead of decimating by row.
3. **Evaluation without leakage:** split by subject or recording *before* augmentation, augment only the
   training part, fit the scaler on training data only. Report per-class precision, recall and F1, plus a
   separate phone-only held-out test.
4. **More real phone data:** several Stairs sessions (up and down), from at least 2–3 people and different
   phone positions. Keep at least one person or session entirely for testing. Without this there is no
   trustworthy way to show that Stairs works on phones.
5. **Balance:** drop the special 1,500-window Stairs cap once the labels are clean.
6. **Downloads and compute:** WISDM and Heterogeneity datasets (not in the repo), and CPU training time.

## 6. Recommended safest next step

1. Keep the current model in the app as it is (it works for Still and for the trained Walking/Jogging
   recordings), and present Stairs as a known limitation, citing these measurements.
2. Correct the 88.6% and 82.99% claims in the synopsis and reports before submission.
3. Record new labelled phone sessions, especially Stairs, with at least one held-out person or session.
4. Then retrain with the fixed pipeline into a **separate folder** (for example `backend/candidate_model/`).
   Compare it with the backup on the same held-out set using `evaluate.py`. Replace the deployed model only
   if the new one is measurably better.


## 7. Follow-up (2026-10-28): Stairs investigation and retraining (model v2)

Script: `backend/diagnostics/stairs_investigation.py` (read-only). Comparison: `backend/compare_models.py` -> `backend/model_v2/comparison.md`.

**New findings about the original model**
- Labels are consistent end to end (Stairs = index 3 in train_model.py, label_encoder, activity_names, model output; the frontend has a colour for every class).
- The two held-out Stairs files overlap in time for 94 of ~95 s and were logged at different sampling rates
  (accel 52.4 / 44.1 Hz, gyro 15.0 / 22.2 Hz): **one stair-climbing session recorded on two devices**, about 1.6 minutes of
  genuine Stairs data in total (not 3.3 minutes of independent recordings as stated in section 2e). `stairs1` has the same rates as `walking1`.
- Inference does not suppress Stairs: the model argmax is never Stairs (0/117); Stairs is the 2nd-ranked class in 59 windows;
  only one rule fired (low-variance Still, once); 0 windows where the model said Stairs and the dashboard did not.
- Mislabelled custom Stairs data (584 windows incl. augmentation) was about 11 % of the Stairs pool before balancing (estimate:
  WISDM and Heterogeneity contributions assumed at the 1,500 cap).
- UCI Stairs entered training about 10× too small (|body acc| 0.28–0.34) compared with the same windows in phone units (2.67–3.20).
- `augment_windows` produces negative magnitude channels in 9 % (3 copies) and 22 % (balancing) of augmented windows.
- WISDM's official `activity_key.txt` and description PDF (downloaded 2026-10-28) confirm the code mapping, the phone in the pocket,
  accelerometer in m/s² including gravity and gyroscope in rad/s at 20 Hz.

**Fix applied: corrected pipeline + retraining (same architecture)** — `prepare_data_v2.py`, `backend/train_model_v2.py`
- All sources converted to the live input format before windowing (m/s² incl. gravity, rad/s, exact 20 Hz); UCI total_acc × 9.80665,
  windows stitched and resampled 50 → 20 Hz; WISDM/Heterogeneity accel and gyro aligned by timestamp; phone CSVs via `parse_sensor_csv`.
- Split by person (WISDM 10 test / 5 val, Heterogeneity 2 / 1, UCI official test + 3 val) and by recording (phone: Stairs session and
  walking2 held out) before augmentation; no person in two splits. No Stairs cap; balanced class weights; scaler fitted on training only;
  physically valid augmentation (3-D rotation, ±10 % amplitude, noise) of the phone training recordings only.
- Trained once (best epoch 38 of 53, validation accuracy 0.836 on validation subjects). Output in `backend/model_v2/`; the original model is unchanged.

**Results on the same held-out windows** (original -> v2)
- Public-dataset subjects not used for v2 (27,352 windows): accuracy 33.6 % -> 77.9 %, macro F1 0.197 -> 0.796, Stairs recall 0.000 -> 0.928
  (the original model saw these datasets in training, in other units).
- Phone Stairs session: 0/117 -> 54/117 (model output); dashboard pipeline: stairs1 0 % -> 84 %, stairs2 0 % -> 17 %.
- Phone walking2 (held out for v2 only): 100 % -> 87.5 % (4 windows -> Stairs). Still recall on public data 0.963 -> 0.895.
- Parameters unchanged (840,647); single-window latency 19.5 vs 19.9 ms median.

**Remaining limitation:** one phone Stairs session from one person is the only phone test data for Stairs. New labelled phone
recordings from several people are needed before claiming that phone Stairs recognition works.
