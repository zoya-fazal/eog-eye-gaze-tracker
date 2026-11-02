"""
Generates the example data in example_data/ (simulated, not a real person):

  EOG_example_calib.npz      targets shown during a 15-target calibration
  EOG_example_recording.npz  EOG recorded during it: 500 Hz, columns
                             [L, R, U, D, trigger], microvolts
  EOG_example_recording.csv  same recording as CSV (header row, microvolts, 500 Hz)
  EOG_example_test.npz       a separate recording with different targets,
                             plus the true target positions, for testing
                             predict() on unseen data

Run:  python make_example_data.py
"""
from pathlib import Path

import numpy as np

import eog_tracker as et

OUT = Path("example_data")
FS = 1000            # simulation rate
FS_FILE = 500        # rate of the saved "recording", like a real amplifier
CHANNELS = np.array(["L (left canthus)", "R (right canthus)", "U (above eye)",
                     "D (below eye)", "Trigger"])

CFG = dict(fs=FS, n_targets=15, n_flats=0.5, pause_interval=(1, 3), hold_time=0.5,
           move_time=1.0, constant_time=True, lead_time=2.0, ending_time=2.0,
           max_x=1.9, max_y=1.15)


def fake_frames(plan, rng):
    """~60 Hz display with jitter, some dropped frames and ~12 ms latency."""
    T = plan["n"] / plan["fs"]
    d = 1 / 60 + 0.002 * rng.standard_normal(int(T * 70))
    d += (rng.random(len(d)) < 0.02) * rng.integers(1, 4, len(d)) / 60
    frame_t = np.cumsum(np.maximum(d, 0.008))
    frame_t = frame_t[frame_t < T]
    frame_k = np.clip(np.floor((frame_t - 0.012) * plan["fs"]).astype(int), 0, plan["n"] - 1)
    return frame_t, frame_k


def to_file(bip, s0):
    """Bipolar [H, V] at 1000 Hz -> monopolar L R U D + trigger at 500 Hz."""
    H, V = bip[:, 0], bip[:, 1]
    mono = np.column_stack([-H / 2 + 120, H / 2 + 80, V / 2 - 40, -V / 2 - 60])
    mono = et.resample(mono, FS_FILE, FS)
    trig = np.zeros(len(mono))
    s0f = s0 * FS_FILE // FS
    trig[s0f: s0f + 10] = 5                        # 20 ms TTL pulse at animation start
    return np.column_stack([np.round(mono, 2), trig]), s0f


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    # Calibration session
    plan = et.build_plan(CFG, rng)
    frame_t, frame_k = fake_frames(plan, rng)
    x_true, y_true = et.truth_from_frames(plan, frame_t, frame_k)
    bip, s0, _ = et.simulate_eog(x_true, y_true, FS, rng)
    eog, _ = to_file(bip, s0)
    np.savez_compressed(OUT / "EOG_example_calib.npz", plan_x=plan["x"], plan_y=plan["y"],
             plan_seg=plan["seg"], fs=FS, frame_t=frame_t, frame_k=frame_k,
             x_true=x_true, y_true=y_true)
    np.savez_compressed(OUT / "EOG_example_recording.npz", eog=eog, fs=FS_FILE, channels=CHANNELS,
             units="microvolts")
    np.savetxt(OUT / "EOG_example_recording.csv", eog, delimiter=",", fmt="%.2f",
               header="L,R,U,D,trigger", comments="")
    print(f"Calibration: {plan['n'] / FS:.1f} s, recording {eog.shape} at {FS_FILE} Hz")

    # Separate test session with different targets
    cfg2 = dict(CFG, n_targets=10)
    plan2 = et.build_plan(cfg2, rng)
    ft2, fk2 = fake_frames(plan2, rng)
    xt2, yt2 = et.truth_from_frames(plan2, ft2, fk2)
    bip2, s02, _ = et.simulate_eog(xt2, yt2, FS, rng)
    eog2, s0f = to_file(bip2, s02)
    xy_true = np.full((len(eog2), 2), np.nan)      # target position on the file's timeline
    tgt = np.column_stack([xt2, yt2])[:: FS // FS_FILE]
    n = min(len(tgt), len(eog2) - s0f)
    xy_true[s0f: s0f + n] = tgt[:n]
    np.savez_compressed(OUT / "EOG_example_test.npz", eog=eog2, fs=FS_FILE, channels=CHANNELS,
             units="microvolts", xy_true=xy_true, target_delay_ms=200)
    print(f"Test recording: {eog2.shape} at {FS_FILE} Hz")


if __name__ == "__main__":
    main()
