"""
EOG eye tracker: turn EOG signals into the (x, y) position someone is
looking at on a screen.

Main pieces
    preprocess(raw, fs, prep, centre)   raw EOG -> clean [H, V] + blink mask
    fit_model(clean, target, ...)       learn EOG -> (x, y) regression
    predict(model, raw, fs_in, centre)  raw EOG -> predicted (x, y)
    EOGModel.save / EOGModel.load       store a trained model as JSON

Helpers for calibration
    build_plan, truth_from_frames, estimate_lag, shift_signal,
    gaze_stats, simulate_eog, load_eog

Screen units: x in [-1.9, 1.9], y in [-1.15, 1.15] (centre = 0, 0),
same as the original MATLAB paradigm.

Channel indices are 0-based (Python), so 4 monopolar electrodes in the
first four columns are channels=[0, 1, 2, 3] in the order [L, R, U, D].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
DEFAULT_PREP = {
    "channels": [0, 1],      # [H, V] bipolar, or [L, R, U, D] monopolar (H = R-L, V = U-D)
    "flip": [1, 1],          # sign of H and V (-1 if a channel is inverted)
    "lp_cutoff": 20.0,       # Hz. EOG content is < ~20 Hz; also removes 50/60 Hz mains
    "lp_order": 4,
    "remove_blinks": True,
    "blink_k": 5.0,          # blink threshold, in robust SDs
    "blink_max_dur": 0.4,    # s
    "blink_polarity": 1,     # +1 if blinks push V up, -1 if down
    "drift": "poly",         # 'poly' | 'highpass' | 'none' (used when no centre mask)
    "poly_order": 1,
    "hp_cutoff": 0.05,       # Hz, for drift='highpass'
}


def with_defaults(prep: dict | None) -> dict:
    p = dict(DEFAULT_PREP)
    if prep:
        p.update({k: v for k, v in prep.items() if v is not None})
    return p


def montage(raw, channels, flip=(1, 1)) -> np.ndarray:
    """Select / combine raw channels into [H, V]."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2:
        raise ValueError("raw must be a 2-D array (samples x channels)")
    ch = list(channels)
    if len(ch) == 2:
        hv = raw[:, ch]
    elif len(ch) == 4:
        L, R, U, D = ch
        hv = np.column_stack([raw[:, R] - raw[:, L], raw[:, U] - raw[:, D]])
    else:
        raise ValueError("channels must have 2 ([H, V]) or 4 ([L, R, U, D]) entries")
    hv = hv * np.asarray(flip, dtype=float)
    for c in range(2):
        hv[:, c] = _fill_nans(hv[:, c])
    return hv


def preprocess(raw, fs, prep=None, centre=None):
    """
    Raw EOG -> clean [H, V].

    raw    : (N, C) raw samples
    fs     : sampling rate (Hz)
    prep   : dict of options (see DEFAULT_PREP)
    centre : optional (N,) bool, samples where the person looked at the
             screen centre. If given, drift is removed by interpolating
             between these fixations (most accurate). Otherwise prep['drift'].

    Returns clean (N, 2) and blink (N,) bool.
    """
    p = with_defaults(prep)
    x = montage(raw, p["channels"], p["flip"])
    x = x - np.median(x, axis=0)
    n = len(x)

    # 1. Low-pass, zero phase
    sos = signal.butter(p["lp_order"], p["lp_cutoff"], btype="low", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)

    # 2. Blinks: detected on V, interpolated out of both channels
    blink = np.zeros(n, dtype=bool)
    if p["remove_blinks"]:
        blink = detect_blinks(x[:, 1], fs, p)
        if blink.any():
            x = _interp_mask(x, blink)

    # 3. Drift
    if centre is not None:
        c = np.asarray(centre, dtype=bool)[:n] & ~blink[: len(centre)]
        if c.any():
            cc = np.zeros(n, dtype=bool)
            cc[: len(c)] = c
            return _anchor_detrend(x, cc, fs), blink

    drift = str(p["drift"]).lower()
    if drift == "poly":
        t = np.linspace(-1, 1, n)
        good = ~blink
        for c in range(2):
            coef = np.polyfit(t[good], x[good, c], p["poly_order"])
            x[:, c] -= np.polyval(coef, t)
    elif drift == "highpass":
        sos = signal.butter(1, p["hp_cutoff"], btype="high", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)
    elif drift != "none":
        raise ValueError(f"Unknown drift option {p['drift']!r}")
    return x, blink


def detect_blinks(v, fs, p) -> np.ndarray:
    """
    A blink is a short pulse on V (up and back within ~0.4 s); a vertical
    saccade is a step that stays. A running median about twice the blink
    length keeps steps and ramps (eye movements) and removes short pulses,
    so V minus its running median leaves only the blinks.
    """
    n = len(v)
    q = max(1, int(fs // 100))                     # run the median at ~100 Hz (fast)
    vd = v[::q]
    w = int(round(2 * p["blink_max_dur"] * fs / q)) | 1
    med = ndimage.median_filter(vd, size=w, mode="nearest")
    med = np.interp(np.arange(n), np.arange(0, n, q)[: len(med)], med)
    r = p["blink_polarity"] * (v - med)
    sd = 1.4826 * np.median(np.abs(r - np.median(r)))
    if sd == 0:
        return np.zeros(n, dtype=bool)
    thr = p["blink_k"] * sd
    core = r > thr
    edge = r > 0.15 * thr                          # grow out to where the pulse starts/ends
    lab, _ = ndimage.label(edge)
    hit = np.unique(lab[core])
    blink = np.isin(lab, hit[hit > 0])
    pad = int(round(0.03 * fs))                    # safety margin
    if pad > 0 and blink.any():
        blink = ndimage.binary_dilation(blink, structure=np.ones(2 * pad + 1, dtype=bool))
    return blink


def _anchor_detrend(x, centre, fs):
    """Median of each centre fixation = drift there; interpolate in between,
    hold constant before the first and after the last."""
    lab, _ = ndimage.label(centre)
    tc, lv = [], []
    for sl in ndimage.find_objects(lab):
        s, e = sl[0].start, sl[0].stop
        if e - s < 0.1 * fs:                      # ignore fixations < 100 ms
            continue
        tc.append((s + e - 1) / 2)
        lv.append(np.median(x[s:e], axis=0))
    if not tc:
        return x
    lv = np.asarray(lv)
    if len(tc) == 1:
        return x - lv[0]
    t = np.arange(len(x))
    drift = np.column_stack([np.interp(t, tc, lv[:, c]) for c in range(2)])
    return x - drift


def _interp_mask(x, mask):
    good = np.flatnonzero(~mask)
    bad = np.flatnonzero(mask)
    if len(good) < 2:
        return x
    x = x.copy()
    for c in range(x.shape[1]):
        x[bad, c] = np.interp(bad, good, x[good, c])
    return x


def _fill_nans(y):
    nn = np.isnan(y)
    if not nn.any():
        return y
    if nn.all():
        return np.zeros_like(y)
    i = np.arange(len(y))
    y = y.copy()
    y[nn] = np.interp(i[nn], i[~nn], y[~nn])
    return y


def resample(x, fs_out, fs_in):
    """Polyphase resampling along axis 0 (like MATLAB resample)."""
    if fs_out == fs_in:
        return np.asarray(x, dtype=float)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    x = np.asarray(x, dtype=float)
    base = np.median(x, axis=0)                    # reduce edge transients
    y = signal.resample_poly(x - base, frac.numerator, frac.denominator, axis=0)
    return y + base


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def features(clean, mu, sd, order):
    """order 1: [1 h v]; 2: + h^2 v^2 hv; 3: + h^3 v^3 h^2v hv^2."""
    h = (clean[:, 0] - mu[0]) / sd[0]
    v = (clean[:, 1] - mu[1]) / sd[1]
    cols = [np.ones_like(h), h, v]
    if order >= 2:
        cols += [h**2, v**2, h * v]
    if order >= 3:
        cols += [h**3, v**3, h**2 * v, h * v**2]
    return np.column_stack(cols)


@dataclass
class EOGModel:
    B: np.ndarray                  # regression weights (n_features, 2)
    mu: np.ndarray                 # feature normalisation
    sd: np.ndarray
    order: int
    fs: float = 1000.0
    prep: dict = field(default_factory=dict)
    lag_ms: float = 0.0
    max_x: float = 1.9
    max_y: float = 1.15
    cv: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def predict(self, raw, fs_in=None, centre=None):
        return predict(self, raw, fs_in, centre)

    def save(self, path):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        Path(path).write_text(json.dumps(d, indent=2, default=_json_default))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        for k in ("B", "mu", "sd"):
            d[k] = np.asarray(d[k], dtype=float)
        return cls(**d)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def fit_model(clean, target, order=2, ridge=1e-3, **kw) -> EOGModel:
    """Ridge least squares from EOG features to [x, y]."""
    mu = clean.mean(axis=0)
    sd = clean.std(axis=0, ddof=1)
    F = features(clean, mu, sd, order)
    R = ridge * len(F) * np.eye(F.shape[1])
    R[0, 0] = 0                                    # don't penalise the offset
    B = np.linalg.solve(F.T @ F + R, F.T @ target)
    return EOGModel(B=B, mu=mu, sd=sd, order=order, **kw)


def apply_model(model: EOGModel, clean):
    return features(clean, model.mu, model.sd, model.order) @ model.B


def predict(model: EOGModel, raw, fs_in=None, centre=None):
    """
    Raw EOG -> predicted gaze.

    raw    : (N, C) raw EOG, same channel layout as the calibration
    fs_in  : sampling rate of raw (if different from model.fs it is resampled)
    centre : optional (N,) bool, same length as raw: samples where the person
             looked at the screen centre (e.g. 2 s at start and end).
             Strongly recommended: it removes EOG drift.

    Returns xy (N, 2) in screen units and blink (N,) bool, at the input rate.
    """
    p = with_defaults(model.prep)
    hv = montage(raw, p["channels"], p["flip"])
    n_in = len(hv)
    fs_in = model.fs if fs_in is None else fs_in
    resampled = fs_in != model.fs
    if resampled:
        hv = resample(hv, model.fs, fs_in)
        if centre is not None:
            idx = np.clip(np.round(np.arange(len(hv)) / model.fs * fs_in).astype(int), 0, n_in - 1)
            centre = np.asarray(centre, dtype=bool)[idx]

    p2 = dict(p, channels=[0, 1], flip=[1, 1])     # montage already applied
    clean, blink = preprocess(hv, model.fs, p2, centre)
    xy = apply_model(model, clean)
    lim = 1.2 * np.array([model.max_x, model.max_y])
    xy = np.clip(xy, -lim, lim)

    if resampled:
        t_new = np.arange(len(xy)) / model.fs
        t_in = np.arange(n_in) / fs_in
        xy = np.column_stack([np.interp(t_in, t_new, xy[:, c]) for c in range(2)])
        blink = blink[np.clip(np.round(t_in * model.fs).astype(int), 0, len(blink) - 1)]
    return xy, blink


# --------------------------------------------------------------------------
# Calibration helpers
# --------------------------------------------------------------------------
def build_plan(cfg: dict, rng=None) -> dict:
    """Random targets: each appears, holds still, moves, then the target
    returns to the centre for a random pause. Positions at cfg['fs']."""
    rng = np.random.default_rng() if rng is None else rng
    fs, n = cfg["fs"], cfg["n_targets"]
    mx, my = cfg["max_x"], cfg["max_y"]

    pauses = cfg["pause_interval"][0] + np.diff(cfg["pause_interval"])[0] * rng.random(n)
    x = -mx + 2 * mx * rng.random((n, 2))          # [start, end]
    y = -my + 2 * my * rng.random((n, 2))

    flats = rng.permutation(n)[: int(n * cfg["n_flats"])]
    hflats, vflats = flats[: len(flats) // 2], flats[len(flats) // 2:]
    y[hflats, 1] = y[hflats, 0]                    # purely horizontal
    x[vflats, 1] = x[vflats, 0]                    # purely vertical

    if cfg["constant_time"]:
        move = np.full(n, cfg["move_time"], dtype=float)
    else:
        move = np.maximum(0.5, np.hypot(np.diff(x, axis=1)[:, 0], np.diff(y, axis=1)[:, 0]))

    T = cfg["lead_time"] + pauses.sum() + n * cfg["hold_time"] + move.sum() + cfg["ending_time"]
    N = int(np.ceil(T * fs)) + n                   # + n: room for rounding
    px, py, seg = np.zeros(N), np.zeros(N), np.zeros(N, dtype=int)

    cur = int(round(cfg["lead_time"] * fs))
    nh = int(round(cfg["hold_time"] * fs))
    for i in range(n):
        cur += int(round(pauses[i] * fs))
        nm = int(round(move[i] * fs))
        ih = cur + np.arange(nh)
        im = cur + nh + np.arange(nm)
        px[ih], py[ih] = x[i, 0], y[i, 0]
        px[im] = np.linspace(x[i, 0], x[i, 1], nm)
        py[im] = np.linspace(y[i, 0], y[i, 1], nm)
        seg[ih] = seg[im] = i + 1
        cur += nh + nm
    N = cur + int(round(cfg["ending_time"] * fs))
    return dict(x=px[:N], y=py[:N], seg=seg[:N], n=N, fs=fs,
                start=np.column_stack([x[:, 0], y[:, 0]]),
                stop=np.column_stack([x[:, 1], y[:, 1]]))


def truth_from_frames(plan, frame_t, frame_k):
    """What was actually on screen at every sample: each drawn frame stays
    on screen until the next one (zero-order hold)."""
    t = np.arange(plan["n"]) / plan["fs"]
    fi = np.clip(np.searchsorted(frame_t, t, side="right") - 1, 0, len(frame_t) - 1)
    k = np.asarray(frame_k)[fi]
    return plan["x"][k], plan["y"][k]


def shift_signal(x, L):
    """y[t] = x[t - L], NaN padded."""
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)
    n = len(x)
    if L >= 0:
        y[L:] = x[: n - L]
    else:
        y[: n + L] = x[-L:]
    return y


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if d == 0 else float((a * b).sum() / d)


def _lag_score(L, clean, x, y, mask):
    xs, ys = shift_signal(x, L), shift_signal(y, L)
    k = mask & ~np.isnan(xs)
    if k.sum() < 10:
        return 0.0
    return _corr(clean[k, 0], xs[k]) ** 2 + _corr(clean[k, 1], ys[k]) ** 2


def estimate_lag(clean, x, y, mask, fs, range_ms=(-500, 1000)):
    """Lag (samples) that best aligns target with EOG (H~x, V~y).
    Positive = eye behind target. Returns lag, coarse lags, scores."""
    lags = np.arange(round(range_ms[0] * fs / 1000), round(range_ms[1] * fs / 1000) + 1)
    coarse = lags[:: max(1, round(fs / 200))]
    sc = np.array([_lag_score(L, clean, x, y, mask) for L in coarse])
    best = coarse[np.argmax(sc)]
    fine = lags[np.abs(lags - best) <= round(fs / 100)]
    sf = np.array([_lag_score(L, clean, x, y, mask) for L in fine])
    return int(fine[np.argmax(sf)]), coarse, sc


def gaze_stats(pred, truth, max_x, max_y) -> dict:
    e = pred - truth
    rmse = np.sqrt(np.mean(e**2, axis=0))
    r2 = 1 - np.sum(e**2, axis=0) / np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    return dict(rmse=rmse.tolist(), r2=r2.tolist(),
                median_err=float(np.median(np.hypot(e[:, 0], e[:, 1]))),
                rmse_pct_screen=(100 * rmse / (2 * np.array([max_x, max_y]))).tolist())


# --------------------------------------------------------------------------
# Data in
# --------------------------------------------------------------------------
def simulate_eog(x, y, fs, rng=None):
    """
    Synthetic 2-channel EOG [H, V] in microvolts, to test without hardware:
    eye follows the target 200 ms late with ~30 ms dynamics, plus crosstalk,
    a slight nonlinearity, slow drift, 60 Hz mains, noise and blinks.
    The recording starts 2 s before the animation.
    Returns eog (M, 2), s0 (index of animation start), blink_true (M,).
    """
    rng = np.random.default_rng() if rng is None else rng
    pre, post, L = 2 * fs, 1 * fs, int(round(0.2 * fs))
    ex = np.concatenate([np.zeros(pre + L), x, np.zeros(post - L)])
    ey = np.concatenate([np.zeros(pre + L), y, np.zeros(post - L)])
    a = np.exp(-1 / (0.03 * fs))
    ex = signal.lfilter([1 - a], [1, -a], ex)
    ey = signal.lfilter([1 - a], [1, -a], ey)

    M = len(ex)
    t = np.arange(M) / fs
    drift = np.cumsum(rng.standard_normal((M, 2)), axis=0) * 0.05
    drift += np.array([40, -25]) * (t / t[-1])[:, None]
    drift += 15 * np.sin(2 * np.pi * 0.01 * t[:, None] + np.array([0, 1]))
    mains = 15 * np.sin(2 * np.pi * 60 * t[:, None] + np.array([0, 0.5]))
    noise = 5 * rng.standard_normal((M, 2))

    H = 180 * ex + 12 * ey + 0.15 * 90 * ex**3
    V = 150 * ey + 10 * ex

    blinks = np.zeros(M)
    nb = int(round(M / fs / 60 * 20))              # ~20 blinks/min
    w = int(round(0.25 * fs))
    bump = np.sin(np.pi * np.arange(w) / w) ** 2
    for bt in rng.integers(pre, M - post, nb):
        blinks[bt: bt + w] += 300 * bump[: M - bt]

    eog = np.column_stack([H + 0.05 * blinks, V + blinks]) + drift + mains + noise + [200, -150]
    return eog, pre, blinks > 0


def read_eog_file(path, default_fs=None):
    """Read .npz (arrays 'eog', optional 'fs'), .mat ('eog', optional 'fs')
    or .csv/.txt (numbers; header lines are skipped). Returns eog, fs."""
    path = Path(path)
    ext = path.suffix.lower()
    fs = default_fs
    if ext == ".npz":
        d = np.load(path, allow_pickle=False)
        eog = d["eog"]
        if "fs" in d:
            fs = float(d["fs"])
    elif ext == ".mat":
        from scipy.io import loadmat
        d = loadmat(path)
        eog = d["eog"] if "eog" in d else d[[k for k in d if not k.startswith("__")][0]]
        if "fs" in d:
            fs = float(np.squeeze(d["fs"]))
    else:
        delim = "," if ext == ".csv" else None
        skip = 0                                   # skip header lines (anything non-numeric)
        with open(path) as fh:
            for skip, line in enumerate(fh):
                parts = [v for v in line.strip().split(delim) if v.strip()]
                try:
                    [float(v) for v in parts]
                    if parts:
                        break
                except ValueError:
                    continue
        eog = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2)
    eog = np.asarray(eog, dtype=float)
    if eog.ndim == 1:
        eog = eog[:, None]
    if eog.shape[0] < eog.shape[1]:
        eog = eog.T                                # samples in rows
    if fs is None:
        raise ValueError(f"No sampling rate stored in {path}; pass eog_fs.")
    return eog, fs


def find_trigger(marker):
    """Index of the first rising edge on a trigger channel."""
    m = np.asarray(marker, dtype=float)
    th = (np.nanmin(m) + np.nanmax(m)) / 2
    idx = np.flatnonzero((m[:-1] < th) & (m[1:] >= th))
    if len(idx) == 0:
        raise ValueError("No trigger found on the marker channel.")
    return int(idx[0] + 1)


def load_eog(path, fs_target, eog_fs=None, start_sample=None, marker_channel=None):
    """
    Load a recording and find where the calibration animation started.
    Returns raw (all channels, resampled to fs_target) and s0 (start index).
    """
    eog, fs_file = read_eog_file(path, eog_fs)
    if start_sample is not None:
        s0 = int(start_sample)
    elif marker_channel is not None:
        s0 = find_trigger(eog[:, marker_channel])
    else:
        print("WARNING: no start sample or marker channel given; assuming the "
              "recording started with the animation (the lag search absorbs small offsets).")
        s0 = 0
    if fs_file != fs_target:
        for c in range(eog.shape[1]):
            eog[:, c] = _fill_nans(eog[:, c])
        s0 = int(round(s0 * fs_target / fs_file))
        eog = resample(eog, fs_target, fs_file)
    return eog, s0
