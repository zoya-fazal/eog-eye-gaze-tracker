"""
ONE-CLICK DEMO

  python3 run_demo.py          (or open this file in VS Code and press the Run button)

What it does:
  1. Trains the gaze model on the example calibration recording
     (only the first time; the model is saved to results/).
  2. Opens a window where a red dot moves to where the model predicts
     the person is looking, while the EOG signals scroll underneath.

To watch a different recording, change RECORDING below.
"""
import subprocess
import sys
from pathlib import Path

# ---- Which recording to animate --------------------------------------------
# "test"    : moving-dot task (a white ring shows the real target)
# "reading" : looking at corners, reading 4 lines, following a circle
RECORDING = "test"
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
MODEL = HERE / "results" / "EOG_model_S01.json"
PY = sys.executable

RECORDINGS = {
    "test": ["--eog", "example_data/EOG_example_test.npz"],
    "reading": ["--eog", "example_data/EOG_example_reading.csv", "--eog-fs", "500",
                "--centre-first", "4", "--centre-last", "3"],
}


def run(args):
    r = subprocess.run([PY, *args], cwd=HERE)
    if r.returncode != 0:
        sys.exit(r.returncode)


missing = [f for f in ("example_data/EOG_example_calib.npz", "example_data/EOG_example_recording.npz",
                       "example_data/EOG_example_test.npz", "example_data/EOG_example_reading.csv")
           if not (HERE / f).exists()]
if missing:
    sys.exit("Can't find the example data:\n  " + "\n  ".join(missing) +
             "\nMake sure the 'example_data' folder is next to run_demo.py.")

if not MODEL.exists():
    print("Step 1/2: training the model on the example data (about 10 s)...\n")
    run(["calibrate.py",
         "--calib", "example_data/EOG_example_calib.npz",
         "--eog", "example_data/EOG_example_recording.npz",
         "--channels", "0", "1", "2", "3", "--marker", "4", "--no-show"])
    print()
else:
    print("Step 1/2: model already trained (delete the results folder to retrain).")

print("Step 2/2: playing the gaze animation. Space = pause, arrows = skip, q = quit.")
run(["animate_gaze.py", *RECORDINGS[RECORDING], *sys.argv[1:]])
