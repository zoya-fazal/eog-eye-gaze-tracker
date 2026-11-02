"""
Live gaze animation: plays an EOG recording back in real time and moves a
red dot to where the model predicts the person is looking.

  Top    : the screen. Red dot = predicted gaze (with a short trail).
           White ring = the real target, when the file has it (example data).
           The dot turns grey during blinks.
  Bottom : the horizontal and vertical EOG scrolling past, so you can see
           how the signal drives the dot.

Run (after training with calibrate.py):
  python3 animate_gaze.py
      plays example_data/EOG_example_test.npz

  python3 animate_gaze.py --eog my_recording.csv --eog-fs 500 --centre-first 2 --centre-last 2
      your own recording; --centre-first/--centre-last = seconds at the
      start/end where the person looked at the screen centre (removes drift)

  python3 animate_gaze.py --save gaze.gif
      saves a GIF (or .mp4 if ffmpeg is installed) instead of opening a window

Keys while playing: space = pause/resume, left/right = jump 5 s, q = quit.
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import signal

import eog_tracker as et

EXAMPLE = "example_data/EOG_example_test.npz"


def main():
    ap = argparse.ArgumentParser(description="Live red-dot gaze animation from EOG")
    ap.add_argument("--model", default="results/EOG_model_S01.json")
    ap.add_argument("--eog", default=EXAMPLE, help="recording (.npz/.mat/.csv/.txt)")
    ap.add_argument("--eog-fs", type=float, help="sampling rate, if not stored in the file")
    ap.add_argument("--centre-first", type=float, help="seconds at the start looking at the centre")
    ap.add_argument("--centre-last", type=float, help="seconds at the end looking at the centre")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--trail", type=float, default=0.4, help="trail length (s)")
    ap.add_argument("--save", help="save to .gif or .mp4 instead of showing")
    args = ap.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")

    # ---- Load model + recording, predict gaze --------------------------------
    model = et.EOGModel.load(args.model)
    eog, fs = et.read_eog_file(args.eog, args.eog_fs)
    n = len(eog)

    is_example = args.eog == EXAMPLE
    first = args.centre_first if args.centre_first is not None else (3.0 if is_example else 0)
    last = args.centre_last if args.centre_last is not None else (2.5 if is_example else 0)
    centre = np.zeros(n, dtype=bool)
    if first > 0:
        centre[int(0.3 * fs): int(first * fs)] = True     # skip the very first 0.3 s
    if last > 0:
        centre[n - int(last * fs): n - int(0.3 * fs)] = True
    xy, blink = model.predict(eog, fs_in=fs, centre=centre if centre.any() else None)

    # Real target, if the file has it (example data)
    target = None
    if args.eog.endswith(".npz"):
        d = np.load(args.eog)
        if "xy_true" in d:
            target = d["xy_true"]

    # EOG for display: [H, V], low-passed, centred
    hv = et.montage(eog, model.prep["channels"], model.prep.get("flip", [1, 1]))
    sos = signal.butter(4, 20, fs=fs, output="sos")
    hv = signal.sosfiltfilt(sos, hv - np.median(hv, axis=0), axis=0)
    t = np.arange(n) / fs
    dur = n / fs

    # ---- Figure ---------------------------------------------------------------
    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(10, 8), facecolor="black")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12,
                          left=0.06, right=0.97, top=0.95, bottom=0.06)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("black")
    ax.set(xlim=(-2, 2), ylim=(-1.25, 1.25), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((-model.max_x, -model.max_y), 2 * model.max_x, 2 * model.max_y,
                               fill=False, ec="0.35", lw=1))
    ax.plot([0], [0], "+", color="0.3", ms=12)
    trail, = ax.plot([], [], "-", color="red", alpha=0.35, lw=3, solid_capstyle="round")
    tgt, = ax.plot([], [], "o", mfc="none", mec="white", ms=22, mew=1.5,
                   label="real target") if target is not None else (None,)
    dot, = ax.plot([], [], "o", color="red", ms=16, label="predicted gaze")
    title = ax.set_title("", color="white", fontsize=12, loc="left")
    status = ax.text(1.95, 1.2, "", color="0.8", ha="right", va="top", fontsize=11)
    leg = ax.legend(loc="lower right", facecolor="black", edgecolor="0.3", labelcolor="white",
                    markerscale=0.7, fontsize=9)
    leg.set_zorder(0)

    ex = fig.add_subplot(gs[1])
    ex.set_facecolor("black")
    ex.tick_params(colors="0.6", labelsize=8)
    for s in ex.spines.values():
        s.set_color("0.3")
    win = 5.0                                          # seconds of EOG shown
    lh, = ex.plot([], [], color="#4fc3f7", lw=1, label="horizontal EOG")
    lv, = ex.plot([], [], color="#ffb74d", lw=1, label="vertical EOG")
    cur = ex.axvline(0, color="red", lw=1)
    lim = np.percentile(np.abs(hv), 99.5) * 1.1
    ex.set(ylim=(-lim, lim))
    ex.set_ylabel("µV", color="0.6")
    ex.legend(loc="upper left", facecolor="black", edgecolor="0.3", labelcolor="white", fontsize=8)

    ntrail = max(1, int(args.trail * fs))
    state = dict(t0=time.perf_counter(), offset=0.0, paused=False, pause_pos=0.0)

    def position():
        if state["paused"]:
            return state["pause_pos"]
        return (state["offset"] + (time.perf_counter() - state["t0"]) * args.speed) % dur

    def draw(pos):
        i = min(n - 1, int(pos * fs))
        a = max(0, i - ntrail)
        trail.set_data(xy[a: i + 1, 0], xy[a: i + 1, 1])
        dot.set_data([xy[i, 0]], [xy[i, 1]])
        dot.set_color("0.5" if blink[i] else "red")
        status.set_text("blink" if blink[i] else "")
        if target is not None and not np.isnan(target[i, 0]):
            tgt.set_data([target[i, 0]], [target[i, 1]])
        elif target is not None:
            tgt.set_data([], [])
        title.set_text(f"t = {pos:5.1f} s / {dur:.0f} s    gaze x = {xy[i, 0]:+.2f}, y = {xy[i, 1]:+.2f}")
        s0 = max(0, int((pos - win) * fs))
        lh.set_data(t[s0: i + 1], hv[s0: i + 1, 0])
        lv.set_data(t[s0: i + 1], hv[s0: i + 1, 1])
        ex.set_xlim(max(0, pos - win), max(win, pos) + 0.2)
        cur.set_xdata([pos, pos])
        return trail, dot, title, status, lh, lv, cur

    def on_key(ev):
        if ev.key == " ":
            if state["paused"]:
                state["offset"], state["t0"] = state["pause_pos"], time.perf_counter()
                state["paused"] = False
            else:
                state["pause_pos"], state["paused"] = position(), True
        elif ev.key in ("left", "right"):
            p = (position() + (5 if ev.key == "right" else -5)) % dur
            state["offset"], state["t0"], state["pause_pos"] = p, time.perf_counter(), p
        elif ev.key == "q":
            plt.close(fig)

    if args.save:
        out_fps = 20 if args.save.lower().endswith(".gif") else 30
        frames = int(dur / args.speed * out_fps)
        anim = FuncAnimation(fig, lambda k: draw(k * args.speed / out_fps), frames=frames, blit=False)
        writer = "pillow" if args.save.lower().endswith(".gif") else "ffmpeg"
        print(f"Saving {frames} frames to {args.save} ...")
        anim.save(args.save, writer=writer, fps=out_fps, dpi=70 if writer == "pillow" else 100)
        print("Done.")
    else:
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.manager.set_window_title("EOG gaze - live")
        anim = FuncAnimation(fig, lambda _: draw(position()), interval=30, blit=False,
                             cache_frame_data=False)
        plt.show()
    return anim


if __name__ == "__main__":
    main()
