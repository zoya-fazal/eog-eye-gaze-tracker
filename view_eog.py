"""
Plot an EOG recording so you can see what goes into calibrate.py.

  python3 view_eog.py
      shows example_data/EOG_example_recording.npz

  python3 view_eog.py --eog example_data/EOG_example_test.npz
  python3 view_eog.py --eog my_recording.csv --eog-fs 500

Top: every column of the file, as recorded.
Bottom: the two signals the tracker actually uses:
  horizontal EOG = R - L  (goes up when the eyes look right)
  vertical EOG   = U - D  (goes up when the eyes look up; blinks = tall spikes)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import eog_tracker as et

ap = argparse.ArgumentParser()
ap.add_argument("--eog", default="example_data/EOG_example_recording.npz")
ap.add_argument("--eog-fs", type=float, help="sampling rate, if not stored in the file (CSV)")
ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3],
                help="[L R U D] or [H V] columns (0-based)")
args = ap.parse_args()

eog, fs = et.read_eog_file(args.eog, args.eog_fs)
t = np.arange(len(eog)) / fs
names = ["L (left of eye)", "R (right of eye)", "U (above eye)", "D (below eye)", "trigger"]
if eog.shape[1] != 5:
    names = [f"column {i}" for i in range(eog.shape[1])]

print(f"{args.eog}: {eog.shape[0]} samples x {eog.shape[1]} columns, "
      f"{fs:g} Hz, {eog.shape[0] / fs:.1f} s")
print("First 5 rows:")
print(np.array2string(eog[:5], precision=2, suppress_small=True))

fig, axs = plt.subplots(eog.shape[1] + 2, 1, figsize=(12, 9), sharex=True)
for c in range(eog.shape[1]):
    axs[c].plot(t, eog[:, c], lw=0.6)
    axs[c].set_ylabel(names[c], rotation=0, ha="right", va="center", fontsize=9)

hv = et.montage(eog, args.channels)
for k, (lbl, col) in enumerate([("horizontal\n(R - L)", "tab:blue"), ("vertical\n(U - D)", "tab:orange")]):
    ax = axs[eog.shape[1] + k]
    ax.plot(t, hv[:, k] - np.median(hv[:, k]), color=col, lw=0.6)
    ax.set_ylabel(lbl, rotation=0, ha="right", va="center", fontsize=9, color=col)
axs[-1].set_xlabel("Time (s)")
axs[0].set_title(f"{args.eog}  (raw, microvolts)  -  zoom in with the magnifier tool")
fig.tight_layout()
plt.show()
