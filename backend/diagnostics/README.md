# Model diagnostics (read-only — they never modify model files)

Run from the `backend` folder with the project venv:

```bash
.venv/bin/python diagnostics/per_class_metrics.py ..        # precision/recall/F1 + confusion matrices on repo recordings
.venv/bin/python diagnostics/training_style_windows.py ..   # rebuilds the custom training windows exactly as the original prepare_data.py did
.venv/bin/python diagnostics/uci_check.py .. "<path>/UCI HAR Dataset"   # needs the UCI HAR dataset (download link in uci_check.py)
```

UCI HAR download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
(unzip, then unzip the inner "UCI HAR Dataset.zip").

Findings are summarised in `../../MODEL_DIAGNOSIS.md`.
