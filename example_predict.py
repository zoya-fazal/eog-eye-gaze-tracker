"""
Example: predict gaze on a NEW recording with a trained model.

First train a model on the example calibration:
  python calibrate.py --calib example_data/EOG_example_calib.npz \
                      --eog example_data/EOG_example_recording.npz \
                      --channels 0 1 2 3 --marker 4
Then:
  python example_predict.py
"""
import matplotlib.pyplot as plt
import numpy as np

import eog_tracker as et

model = et.EOGModel.load("results/EOG_model_S01.json")
d = np.load("example_data/EOG_example_test.npz")
eog, fs = d["eog"], float(d["fs"])          # columns [L, R, U, D, trigger], 500 Hz
xy_true = d["xy_true"]

# The test recording starts and ends with ~2 s looking at the screen centre.
# Marking those samples lets predict() remove EOG drift accurately.
start = et.find_trigger(eog[:, 4])
task = np.flatnonzero(~np.isnan(xy_true[:, 0]))
centre = np.zeros(len(eog), dtype=bool)
centre[start + int(0.5 * fs): start + int(1.8 * fs)] = True
centre[task[-1] - int(1.8 * fs): task[-1]] = True

# ---- The actual prediction: raw EOG in, (x, y) out -----------------------
xy, blink = model.predict(eog, fs_in=fs, centre=centre)

# ---- Compare with where the eye really was (target delayed by the eye's lag)
d_s = int(round(float(d["target_delay_ms"]) / 1000 * fs))
eye = np.vstack([np.full((d_s, 2), np.nan), xy_true[:-d_s]])
k = ~np.isnan(eye[:, 0])
e = xy[k] - eye[k]
rmse = np.sqrt((e**2).mean(axis=0))
print(f"Test recording: RMSE x = {rmse[0]:.3f}, y = {rmse[1]:.3f}, "
      f"median 2-D error = {np.median(np.hypot(e[:, 0], e[:, 1])):.3f} screen units")

t = np.arange(len(xy)) / fs
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for c, lbl in enumerate("xy"):
    axs[c].plot(t, eye[:, c], "k", lw=1, label="true gaze")
    axs[c].plot(t, xy[:, c], "r", lw=0.9, label="predicted")
    axs[c].set_ylabel(lbl)
axs[0].legend(loc="upper right")
axs[0].set_title("Predicted gaze on unseen data")
axs[1].set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig("results/example_predict.png", dpi=110)
plt.show()
