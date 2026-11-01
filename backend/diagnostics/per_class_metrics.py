import os, sys, warnings
warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "backend"))
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import parse_sensor_csv, windows_from_segments
from model_service import HARModel, SessionSmoother
from recordings import RECORDINGS, path_for
har = HARModel(os.path.join(R, "backend")); C = har.classes
T, M, P, HO = [], [], [], []
for rec in RECORDINGS:
    segs, _ = parse_sensor_csv(open(path_for(rec), "rb").read())
    _, w = windows_from_segments(segs)
    pr = har.predict_probs(w); sm = SessionSmoother()
    T += [rec["label"]]*len(w); M += [C[i] for i in pr.argmax(1)]
    P += [har.postprocess(p, x, sm)["activity"] for p, x in zip(pr, w)]
    HO += [not rec["used_in_training"]]*len(w)
    print(f"{rec['file']:40s} {rec['label']:8s} mean probs: " + " ".join(f"{c[:5]}={v:.2f}" for c, v in zip(C, pr.mean(0))))
T, M, P, HO = map(np.array, (T, M, P, HO))
labels = C + ["Uncertain"]
for name, mask in (("ALL labelled recordings (mostly training data)", np.ones_like(HO)), ("HELD-OUT only", HO)):
    for kind, pred in (("model argmax", M), ("dashboard pipeline", P)):
        print(f"\n=== {name} — {kind} ===")
        present = sorted(set(T[mask]) | set(pred[mask]), key=labels.index)
        print(classification_report(T[mask], pred[mask], labels=present, zero_division=0, digits=3))
        print("confusion rows=true", present); print(confusion_matrix(T[mask], pred[mask], labels=present))
