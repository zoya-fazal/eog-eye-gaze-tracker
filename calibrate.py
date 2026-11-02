"""
EOG eye tracker: calibration.

  1. Shows moving targets fullscreen (pygame) and logs exactly what was on
     screen and when.
  2. Gets the EOG recorded during it (from a file, or simulated).
  3. Cleans the EOG, finds how far the eye lags the target, removes drift
     using the centre fixations, and fits EOG -> (x, y).
  4. Cross-validates, plots, and saves the model (results/EOG_model_SXX.json).

Examples
  python calibrate.py --demo
      no screen, simulated EOG: checks everything works

  python calibrate.py --calib example_data/EOG_example_calib.npz \
                      --eog example_data/EOG_example_recording.npz \
                      --channels 0 1 2 3 --marker 4
      the example recording (4 electrodes + trigger)

  python calibrate.py --eog my_recording.csv --eog-fs 500 --channels 0 1 --marker 2
      shows the targets now; the EOG file is the one your amplifier saved
      during this run (the script asks for it after the animation if you
      don't pass --eog)

Then predict gaze on any new EOG:
  from eog_tracker import EOGModel
  model = EOGModel.load("results/EOG_model_S01.json")
  xy, blink = model.predict(raw_eog, fs_in=500)

Original paradigm by dcaro (13-Feb-2025); Python port.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import eog_tracker as et

# ==========================================================================
# Settings (command-line options override the main ones)
# ==========================================================================
CONFIG = dict(
    subject=1,
    fs=1000,                 # analysis sampling rate (Hz)

    # Calibration paradigm
    n_targets=12,            # moving targets (4 is too few to fit well)
    n_flats=0.5,             # fraction moving purely horizontally/vertically
    pause_interval=(1, 3),   # random centre pause before each target (s)
    hold_time=0.5,           # target sits still at its start point (s)
    move_time=1.0,           # time for each target to move (s)
    constant_time=True,      # False = move time proportional to distance
    lead_time=1.0,           # centre fixation before the first target (s)
    ending_time=1.5,         # centre fixation at the end (s)
    max_x=1.9, max_y=1.15,   # screen limits (axes units, centre = 0, 0)
    dot_radius=0.04,

    # Preprocessing (see eog_tracker.DEFAULT_PREP for everything)
    prep=dict(channels=[0, 1], flip=[1, 1], lp_cutoff=20, remove_blinks=True,
              blink_polarity=1, drift="poly", poly_order=1),

    # Model
    lag_range_ms=(-500, 1000),
    model_order=2,           # 1 = linear, 2 = quadratic (recommended)
    ridge=1e-3,
    k_folds=5,
    out_dir="results",
)


def send_trigger():
    """Hook: send a sync pulse to your amplifier when the animation starts,
    e.g. with pyserial: serial_port.write(b'\\x01'). Does nothing by default."""
    pass


# ==========================================================================
# Paradigm display
# ==========================================================================
def run_paradigm(plan, cfg):
    """Draws the targets fullscreen in real time. Every frame shows where the
    target should be *now* (slow rendering drops frames instead of stretching
    the test) and logs when it was drawn. Esc aborts."""
    import pygame

    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, vsync=1)
    except pygame.error:                           # vsync not available on this system
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)
    W, H = screen.get_size()
    scale = min(W / 4.0, H / 2.5)                  # axes [-2,2] x [-1.25,1.25], equal aspect

    def to_px(x, y):
        return int(W / 2 + x * scale), int(H / 2 - y * scale)

    r_px = max(3, int(cfg["dot_radius"] * scale))
    font = pygame.font.SysFont(None, 80)

    def show_text(s):
        screen.fill((0, 0, 0))
        img = font.render(s, True, (255, 255, 255))
        screen.blit(img, img.get_rect(center=(W // 2, H // 2)))
        pygame.display.flip()

    def check_quit():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                raise SystemExit("Calibration aborted.")

    for c in ("3", "2", "1"):
        show_text(c)
        t = time.perf_counter()
        while time.perf_counter() - t < 1:
            check_quit()
            time.sleep(0.01)

    fs, N = plan["fs"], plan["n"]
    T = N / fs
    frame_t, frame_k = [], []
    t0 = time.perf_counter()
    send_trigger()
    while True:
        el = time.perf_counter() - t0
        if el >= T:
            break
        k = min(N - 1, int(el * fs))
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), to_px(plan["x"][k], plan["y"][k]), r_px)
        pygame.display.flip()
        frame_t.append(time.perf_counter() - t0)
        frame_k.append(k)
        check_quit()
        clock.tick(240)                            # cap if vsync is off (timing is logged anyway)

    show_text("End")
    time.sleep(1)
    pygame.quit()
    return np.array(frame_t), np.array(frame_k, dtype=int)


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="EOG eye-tracker calibration")
    ap.add_argument("--demo", action="store_true", help="no display + simulated EOG")
    ap.add_argument("--simulate", action="store_true", help="use simulated EOG")
    ap.add_argument("--no-display", action="store_true", help="skip the animation (ideal 60 Hz timing)")
    ap.add_argument("--calib", help="re-analyse a saved calibration (.npz) instead of showing targets")
    ap.add_argument("--eog", help="EOG recording (.npz/.mat/.csv/.txt)")
    ap.add_argument("--eog-fs", type=float, help="sampling rate of the EOG file, if not stored in it")
    ap.add_argument("--start-sample", type=int, help="sample (0-based) where the animation started")
    ap.add_argument("--marker", type=int, help="trigger channel (0-based column)")
    ap.add_argument("--channels", type=int, nargs="+", help="[H V] or [L R U D] columns (0-based)")
    ap.add_argument("--blink-polarity", type=int, choices=[-1, 1])
    ap.add_argument("--targets", type=int, help="number of targets")
    ap.add_argument("--subject", type=int)
    ap.add_argument("--seed", type=int, help="random seed for the target plan")
    ap.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    cfg["prep"] = dict(CONFIG["prep"])
    if args.channels:
        cfg["prep"]["channels"] = args.channels
    if args.blink_polarity:
        cfg["prep"]["blink_polarity"] = args.blink_polarity
    if args.targets:
        cfg["n_targets"] = args.targets
    if args.subject:
        cfg["subject"] = args.subject
    simulate = args.demo or args.simulate or (args.eog is None and args.calib is None and args.no_display)
    no_display = args.demo or args.no_display

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = cfg["fs"]
    out = Path(cfg["out_dir"])
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- 1-2. Target plan + paradigm (or load a saved calibration) --------
    if args.calib:
        d = np.load(args.calib)
        plan = dict(x=d["plan_x"], y=d["plan_y"], seg=d["plan_seg"], n=len(d["plan_x"]),
                    fs=float(d["fs"]))
        frame_t, frame_k = d["frame_t"], d["frame_k"]
        if plan["fs"] != fs:
            raise SystemExit(f"{args.calib} was made at {plan['fs']} Hz; set fs = {plan['fs']}.")
        print(f"Loaded calibration from {args.calib} ({plan['n'] / fs:.1f} s)")
    else:
        plan = et.build_plan(cfg, rng)
        print(f"Planned calibration: {cfg['n_targets']} targets, {plan['n'] / fs:.1f} s")
        if cfg["n_targets"] < 8:
            print("NOTE: with fewer than ~8 targets the model sees too little of the screen "
                  "and the cross-validated accuracy is unreliable.")
        if no_display:
            frame_t = np.arange(0, plan["n"] / fs, 1 / 60)
            frame_k = np.floor(frame_t * fs).astype(int)
        else:
            frame_t, frame_k = run_paradigm(plan, cfg)

    x_true, y_true = et.truth_from_frames(plan, frame_t, frame_k)
    dt = np.diff(frame_t)
    print(f"Display: {len(frame_t)} frames, mean {1 / dt.mean():.1f} fps, longest frame {1000 * dt.max():.0f} ms")

    if not args.calib:
        f = out / f"EOG_calib_S{cfg['subject']:02d}_{datetime.now():%Y%m%d_%H%M%S}.npz"
        np.savez_compressed(f, plan_x=plan["x"], plan_y=plan["y"], plan_seg=plan["seg"], fs=fs,
                 frame_t=frame_t, frame_k=frame_k, x_true=x_true, y_true=y_true)
        print(f"Calibration targets saved to {f}")

    # ---- 3. EOG ------------------------------------------------------------
    sim_blink = None
    if simulate and not args.eog:
        eog, s0, sim_blink = et.simulate_eog(x_true, y_true, fs, rng)
    else:
        path = args.eog or input("Path to the EOG recording from this session: ").strip().strip('"')
        eog, s0 = et.load_eog(path, fs, args.eog_fs, args.start_sample, args.marker)

    N = plan["n"]
    idx = s0 + np.arange(N)
    ok = (idx >= 0) & (idx < len(eog))
    eog_cal = np.full((N, eog.shape[1]), np.nan)
    eog_cal[ok] = eog[idx[ok]]
    if not ok.all():
        print(f"WARNING: EOG covers only {100 * ok.mean():.0f}% of the calibration.")

    # ---- 4. Preprocess, lag, drift ------------------------------------------
    clean, blink = et.preprocess(eog_cal, fs, cfg["prep"])
    print(f"Blinks: {100 * blink.mean():.1f}% of samples removed")
    if sim_blink is not None:
        tb = np.zeros(N, dtype=bool)
        tb[ok] = sim_blink[idx[ok]]
        print(f"  Simulation check: {100 * blink[tb].mean():.0f}% of true blink samples caught")

    lag, lags, score = et.estimate_lag(clean, x_true, y_true, ~blink & ok, fs, cfg["lag_range_ms"])
    print(f"Eye lags target by {round(1000 * lag / fs)} ms")
    xa, ya = et.shift_signal(x_true, lag), et.shift_signal(y_true, lag)

    # Redo drift removal using the centre fixations between targets as anchors
    # (a polynomial over the whole recording is biased by where the person looked)
    centre = (xa == 0) & (ya == 0)
    w = int(round(0.25 * fs))                      # skip 250 ms around transitions
    centre = ~np.convolve(~centre, np.ones(2 * w + 1), mode="same").astype(bool)
    clean, blink = et.preprocess(eog_cal, fs, cfg["prep"], centre)
    valid = ok & ~blink & ~np.isnan(xa)
    target = np.column_stack([xa, ya])

    # ---- 5. Cross-validation ------------------------------------------------
    k = cfg["k_folds"]
    fold = np.minimum(k - 1, (np.arange(N) * k) // N)
    pred = np.full((N, 2), np.nan)
    for i in range(k):
        tr = valid & (fold != i)
        te = fold == i
        m = et.fit_model(clean[tr], target[tr], cfg["model_order"], cfg["ridge"])
        pred[te] = et.apply_model(m, clean[te])
    stats = et.gaze_stats(pred[valid], target[valid], cfg["max_x"], cfg["max_y"])
    print(f"\nCross-validated accuracy ({k} folds):")
    print(f"  RMSE x = {stats['rmse'][0]:.3f}  ({stats['rmse_pct_screen'][0]:.1f}% of screen width)")
    print(f"  RMSE y = {stats['rmse'][1]:.3f}  ({stats['rmse_pct_screen'][1]:.1f}% of screen height)")
    print(f"  R^2    = {stats['r2'][0]:.2f} (x), {stats['r2'][1]:.2f} (y)")
    print(f"  Median 2-D error = {stats['median_err']:.3f} screen units")

    # ---- 6. Final model ------------------------------------------------------
    model = et.fit_model(clean[valid], target[valid], cfg["model_order"], cfg["ridge"],
                         fs=fs, prep=cfg["prep"], lag_ms=1000 * lag / fs,
                         max_x=cfg["max_x"], max_y=cfg["max_y"], cv=stats,
                         info=dict(subject=cfg["subject"], created=datetime.now().isoformat(timespec="seconds")))
    model_file = out / f"EOG_model_S{cfg['subject']:02d}.json"
    model.save(model_file)
    print(f"\nModel saved to {model_file}")

    xy, _ = model.predict(eog_cal, centre=centre)  # sanity check of predict()
    diff = np.abs(xy[valid] - et.apply_model(model, clean[valid])).max()
    print(f"predict() check: max difference from fit = {diff:.2g}")

    # ---- 7. Plots ------------------------------------------------------------
    t = np.arange(N) / fs
    figs = {}

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(1000 * dt)
    ax.set(xlabel="Frame", ylabel="Frame duration (ms)",
           title="Display timing (spikes = dropped frames; the target trace accounts for them)")
    figs["timing"] = fig

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    raw_hv = et.montage(np.nan_to_num(eog_cal), cfg["prep"]["channels"], cfg["prep"]["flip"])
    for c, lbl in enumerate(["H", "V"]):
        axs[c].plot(t, raw_hv[:, c] - np.median(raw_hv[:, c]), color="0.7", lw=0.6, label="raw")
        axs[c].plot(t, clean[:, c], "b", lw=0.9, label="clean")
        lo, hi = axs[c].get_ylim()
        axs[c].fill_between(t, lo, hi, where=blink, color="r", alpha=0.15, lw=0, label="blink")
        axs[c].set_ylabel(f"{lbl} EOG")
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("Time (s)")
    fig.suptitle("EOG preprocessing")
    figs["preprocessing"] = fig

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(1000 * lags / fs, score)
    ax.plot(1000 * lag / fs, score.max(), "ro")
    ax.set(xlabel="Eye lag behind target (ms)", ylabel="$r_H^2 + r_V^2$",
           title=f"Best lag = {round(1000 * lag / fs)} ms")
    figs["lag"] = fig

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for c, (lbl, lim) in enumerate([("x", cfg["max_x"]), ("y", cfg["max_y"])]):
        axs[c].plot(t, target[:, c], "k", lw=1, label="target (lag-aligned)")
        axs[c].plot(t, pred[:, c], "r", lw=0.9, label="predicted")
        axs[c].set(ylabel=lbl, ylim=(-1.3 * lim, 1.3 * lim))
    axs[0].legend(loc="upper right")
    axs[0].set_title(f"Cross-validated: RMSE x = {stats['rmse'][0]:.3f}, y = {stats['rmse'][1]:.3f}")
    axs[1].set_xlabel("Time (s)")
    figs["predicted_vs_true"] = fig

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pred[valid, 0], pred[valid, 1], ".", color=(1, .6, .6), ms=1, label="predicted gaze")
    ax.plot(target[:, 0], target[:, 1], "k.", ms=1.5, label="targets")
    ax.add_patch(plt.Rectangle((-cfg["max_x"], -cfg["max_y"]), 2 * cfg["max_x"], 2 * cfg["max_y"],
                               fill=False, ec="0.5"))
    ax.set(aspect="equal", xlim=(-2.2, 2.2), ylim=(-1.4, 1.4), title="Positions on screen (cross-validated)")
    ax.legend(loc="upper right", markerscale=8)
    figs["gaze_on_screen"] = fig

    for name, fig in figs.items():
        fig.tight_layout()
        fig.savefig(out / f"S{cfg['subject']:02d}_{name}.png", dpi=110)
    print(f"Plots saved to {out}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
