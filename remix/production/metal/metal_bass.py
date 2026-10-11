"""metal_bass.py - modern metal bass for the METAL remix.

Public API
----------
render_bass(events, total_len=None) -> mono float64 array
    events: list of (t, midi, dur, vel, slide_from, mute)
        t          onset in seconds (absolute)
        midi       MIDI note (the song plays 24..34: C1 = 32.7 Hz up to Bb1; C2 = 36 also fine)
        dur        seconds the note is held (then the finger damps it: ~45 ms release)
        vel        0..1 pick velocity (level + pick brightness + distortion amount)
        slide_from None or MIDI note to glide from (glide takes ~60 ms, with fret-scrape noise)
        mute       True -> palm-muted thump (fast decay, dark, more pick clank)
    Returns a mono array. Length = int(total_len*SR) if total_len is given, else last event + tail.
    Strictly monophonic: a note is damped 7 ms before the next onset, 6 ms raised-cosine (notes starting
    within 4 ms of each other are treated as a chord and ring together).

Signal path (a Darkglass-style "clean low / dirty mid" split, done on the whole DI stream so that the
compressor and distortion react to the line the way a real rig does):

    DI  = extended Karplus-Strong string. One-period excitation designed in the frequency domain:
          coherent Helmholtz pulse (1/k * |sin(k*pi*pick_pos)| harmonic envelope) + shaped noise,
          pick-hardness lowpass; loop = brightness 2-tap + one-pole damping designed per pitch so that
          the fundamental has T60 = OPEN_T60 (5 s at C2, 7 s at C1) while 1.5 kHz has T60 = OPEN_HF_T60
          (1.5 s): harmonics die faster than the fundamental like on a real string; palm mute: T60 0.16 s,
          HF ~0.1 s. Exact tuning and 60 ms slides by variable-rate read; + pick click, roundwound scrape,
          fret-scrape noise on slides, bridge-pickup comb, pickup resonance ~4.2 kHz
    LOW = LR4 lowpass 150 Hz -> very light saturation -> compressor with a one-period running-max
          detector (no gain ripple at 32 Hz, 15 ms look-ahead = tight)
    MID = LR4 highpass 150 Hz -> pre-EQ (900 Hz grunt bump, 2.5k attack shelf) -> 2-stage asymmetric
          waveshaper -> post-EQ (grit bump 1 kHz, low-mid cut -4 dB @320 Hz, 3.8 kHz 24 dB/oct lowpass)
    SUM = LOW + MID -> bus compressor (2.5:1, velocity survives mostly as grit/attack, ~2 dB level)
          -> 6 kHz lowpass -> true-peak limiter at -6 dBFS (peak ~0.5; works <= ~1-2 dB on accents)

Everything is synthesized; no samples. Pure functions, only an in-memory cache of DI strings
(round-robin seeds so repeated notes hit the cache; bounded to ~CACHE_MB).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
import engine as E

SR = E.SR

# ------------------------------------------------------------------ parameters
RELEASE_S = 0.045          # finger damping after `dur` (exp tau = RELEASE_S/3)
NOTE_TAIL_S = 0.09         # extra samples rendered after dur to hold the release
SLIDE_S = 0.060            # glide length for slide_from
CHOKE_PRE_S = 0.007        # a note is damped this long before the next onset (monophonic string)
CHOKE_FADE_S = 0.006       # raised-cosine length of that damping (ends 1 ms before the onset)
CHORD_WINDOW_S = 0.004     # onsets closer than this ring together (dyad)
OPEN_T60 = 5.0             # fundamental T60 at C2 (scales with sqrt(65.4/f0): ~7 s at C1)
OPEN_HF_T60 = 1.5          # T60 of the 1.5 kHz region on an open string (independent of pitch)
MUTE_T60 = 0.16            # palm mute T60 (all pitches)
MUTE_HF_T60 = 0.04         # palm mute: HF target (clamped by the max damping coefficient)
HF_REF_HZ = 1500.0
CROSS_HZ = 150.0           # low/mid crossover (LR4)
FINAL_LP_HZ = 6000.0
PEAK_CEIL_DB = -6.0        # ~0.5 linear
ROUND_ROBIN = 4            # distinct string variants per (pitch, vel, mute)
CACHE_MB = 160.0           # the DI cache is cleared when it grows past this
_cache = {}                # in-memory DI cache: key -> (src, f_synth)


# ------------------------------------------------------------------ small helpers
def _loop_filter(S, damp):
    """KS loop filter: (1-S + S z^-1) * (1-damp)/(1 - damp z^-1).  Returns b, a."""
    b = np.convolve(np.array([1 - S, S]), np.array([1 - damp]))
    a = np.array([1.0, -damp])
    return b, a


def _resp(b, a, f):
    """complex response of the (FIR b / IIR a) loop filter at f (closed form: freqz is 100x slower)."""
    z = np.exp(-2j * np.pi * f / SR)
    num = np.polyval(b[::-1], z)           # b0 + b1 z^-1 + ...
    den = np.polyval(a[::-1], z)
    return num / den


def _mag(b, a, f):
    return float(np.abs(_resp(b, a, f)))


def _phase_delay(b, a, f):
    w = 2 * np.pi * f / SR
    return float(-np.angle(_resp(b, a, f)) / w)


def _design_loop(f0, S, t60, t60_hf, damp_max):
    """Choose the one-pole damping so that, per period, the loop loses what gives T60 = t60 at f0 and
    T60 = t60_hf at HF_REF_HZ.  Returns (b, a, g): loop filter and the per-period gain."""
    P = 1.0 / f0
    want_f0 = 10 ** (-3.0 * P / t60)            # per-period gain wanted at the fundamental
    want_hf = 10 ** (-3.0 * P / t60_hf)         # ... and at the reference high frequency
    ratio = want_hf / want_f0                   # extra loss the filter must provide at HF_REF (< 1)
    fr = min(HF_REF_HZ, SR * 0.45)
    lo, hi = 0.0, damp_max
    for _ in range(40):                         # bisection: the ratio |H(fr)|/|H(f0)| falls with damp
        mid = 0.5 * (lo + hi)
        b, a = _loop_filter(S, mid)
        r = _mag(b, a, fr) / _mag(b, a, f0)
        if r > ratio:
            lo = mid
        else:
            hi = mid
    damp = 0.5 * (lo + hi)
    b, a = _loop_filter(S, damp)
    g = want_f0 / _mag(b, a, f0)
    return b, a, g


def _ks_core(exc, N, g, b, a, n_total):
    """Karplus-Strong recurrence y[n] = g * H(z) y[n-N], processed one period per block."""
    out = np.zeros(n_total + N)
    out[:N] = exc
    zi = np.zeros(max(len(a), len(b)) - 1)
    s = N
    L = len(out)
    while s < L:
        e = min(s + N, L)
        y, zi = signal.lfilter(b, a, out[s - N:e - N], zi=zi)
        out[s:e] = g * y
        s = e
    return out[:n_total]


def _raised_cos_in(n):
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / max(1, n)))


def _raised_cos_out(n):
    return 0.5 * (1 + np.cos(np.pi * np.arange(n) / max(1, n)))


# ------------------------------------------------------------------ DI string
def _string_src(f0, n_src, vel, mute, seed):
    """Raw plucked-string loop output (no click), synthesised at the integer-period pitch nearest f0.
    The excitation is designed in the frequency domain of one period so that its harmonic envelope is
    exactly what a velocity pickup sees from a pick-plucked string: ~1/k * |sin(k*pi*pick_pos)|, times a
    pick-hardness lowpass, as a coherent Helmholtz pulse plus shaped noise.  Returns (src, f_synth)."""
    vel = float(np.clip(vel, 0.05, 1.0))
    if mute:
        S = 0.5
        t60 = MUTE_T60
        t60_hf = MUTE_HF_T60
        damp_max = 0.9                       # keeps the loop's phase delay small (harmonics stay in tune)
        fc_exc = 500 + 900 * vel
        noise_mix = 0.5
    else:
        S = 0.42 - 0.06 * vel                # harder pick = brighter string
        t60 = OPEN_T60 * (65.4 / f0) ** 0.5
        t60_hf = OPEN_HF_T60
        damp_max = 0.9
        fc_exc = 1200 + 3800 * vel
        noise_mix = 0.35
    b, a, g = _design_loop(f0, S, t60, t60_hf, damp_max)
    tau = _phase_delay(b, a, f0)
    N = int(round(SR / f0 - tau))
    f_synth = SR / (N + tau)

    r = np.random.default_rng(seed)
    p = 0.11 + 0.05 * r.random()         # pick position (fraction of the string, near the bridge)
    w = max(2, int(p * N))
    pulse = np.zeros(N); pulse[:w] = 1.0; pulse -= pulse.mean()
    Xp = np.fft.rfft(pulse)                                   # 1/k * |sin(k pi p)| envelope, coherent phases
    k = np.arange(len(Xp)); fk = k * f_synth
    mag = np.abs(Xp)
    Xn = mag * (0.7 + 0.6 * r.random(len(Xp))) * np.exp(2j * np.pi * r.random(len(Xp)))
    X = Xp * (1 - noise_mix) + Xn * noise_mix
    X *= 1.0 / (1.0 + (fk / fc_exc) ** 2)                     # pick hardness (2-pole-ish)
    X[0] = 0.0
    exc = np.fft.irfft(X, N)
    exc /= (np.sqrt(np.mean(exc ** 2)) + 1e-9)
    src = _ks_core(exc, N, g, b, a, n_src)
    # bridge-pickup comb: fixed physical distance from the bridge -> fraction grows with pitch
    pp = min(0.45, 0.16 * f0 / 65.4)
    dp = max(1, int(pp * N))
    src = src - 0.7 * np.concatenate([np.zeros(dp), src[:-dp]])
    return src * 0.35, f_synth


def _get_src(f0, n_src, vel, mute, seed):
    """Cached string source of at least n_src samples (KS is deterministic: a longer render has the same
    prefix, so the cache keeps the longest one per key and slices)."""
    key = (round(f0, 3), round(float(vel), 2), bool(mute), int(seed))
    hit = _cache.get(key)
    if hit is not None and len(hit[0]) >= n_src:
        return hit[0][:n_src], hit[1]
    n_make = max(n_src, int(0.6 * SR))      # make at least 0.6 s so short repeated notes share one source
    src, f_synth = _string_src(f0, n_make, vel, mute, seed)
    if sum(v[0].nbytes for v in _cache.values()) > CACHE_MB * 1e6:
        _cache.clear()
    _cache[key] = (src, f_synth)
    return src[:n_src], f_synth


def _pick_click(n, vel, mute, r):
    t = E.t_axis(n)
    click = r.standard_normal(n) * np.exp(-t / 0.0011)
    click = E.eq(click, ('hp', 1800, 0.7), ('peak', 3500, 1.0, 3))
    # roundwound pick scrape: ~6 ms of 3-7 kHz noise
    scrape = r.standard_normal(n) * np.exp(-t / 0.004)
    scrape = E.eq(scrape, ('bp', 4500, 0.8))
    lvl = 0.55 * vel ** 2
    y = click * lvl * (1.4 if mute else 1.0) + scrape * lvl * 0.45
    k = min(n, int(0.0004 * SR))             # the pick leaves the string in ~0.4 ms, not in one sample
    y[:k] *= _raised_cos_in(k)
    return y


def _slide_noise(n, vel, r):
    """fret scrape during a glide: 2-4 kHz noise that fades over the slide."""
    t = E.t_axis(n)
    nz = r.standard_normal(n) * np.exp(-t / (SLIDE_S * 0.6))
    nz = E.eq(nz, ('bp', 3000, 0.7))
    k = min(n, int(0.002 * SR))
    nz[:k] *= _raised_cos_in(k)
    return nz * 0.25 * vel


def _note(midi, dur, vel, slide_from, mute, seed):
    """One DI note: exact-pitch (and glide) via variable-rate read of a cached KS source."""
    f0 = E.midi2hz(midi)
    n_out = int((dur + NOTE_TAIL_S) * SR)
    f_from = E.midi2hz(slide_from) if slide_from is not None else f0
    n_sl = int(SLIDE_S * SR)
    # instantaneous frequency: exponential glide (linear in semitones), then constant
    f_inst = np.full(n_out, f0)
    do_slide = slide_from is not None and n_sl > 1 and abs(f_from - f0) > 1e-3
    if do_slide:
        k = min(n_sl, n_out)
        f_inst[:k] = f_from * (f0 / f_from) ** (np.arange(k) / n_sl)
    max_rate = max(1.0, f_from / f0) * 1.01
    n_src = int(n_out * max_rate) + 64
    src, f_synth = _get_src(f0, n_src, vel, mute, seed)
    rate = f_inst / f_synth
    pos = np.cumsum(rate) - rate[0]
    y = np.interp(pos, np.arange(len(src)), src)
    # finger release after dur
    i_rel = int(dur * SR)
    if i_rel < n_out:
        tt = np.arange(n_out - i_rel) / SR
        y[i_rel:] *= np.exp(-tt / (RELEASE_S / 3.0))
    k = int(0.0008 * SR)
    y[:k] *= _raised_cos_in(k)                # no hard edge at the onset
    k = min(n_out, int(0.005 * SR))
    y[n_out - k:] *= _raised_cos_out(k)       # no hard edge where the tail is truncated
    r = np.random.default_rng(seed * 7919 + 13)
    nc = min(n_out, int(0.02 * SR))
    y[:nc] += _pick_click(nc, vel, mute, r)
    if do_slide:
        ns = min(n_out, int(SLIDE_S * 1.5 * SR))
        y[:ns] += _slide_noise(ns, vel, r)
    amp = (0.12 + 0.88 * float(np.clip(vel, 0, 1)) ** 1.3)
    if mute:
        amp *= 1.4        # the palm takes brightness away; give the thump its weight back
    return y * amp


def render_di(events, total_len=None):
    """Clean DI stream (before the amp chain). Same events format as render_bass."""
    if len(events) == 0:
        return np.zeros(int((total_len or 1.0) * SR))
    ev = sorted([tuple(e) for e in events], key=lambda e: e[0])
    last_end = max(e[0] + e[2] + NOTE_TAIL_S for e in ev)
    n = int(total_len * SR) if total_len is not None else int((last_end + 0.6) * SR)
    out = np.zeros(n)
    onsets = np.array([e[0] for e in ev], dtype=np.float64)
    for i, (t, midi, dur, vel, slide_from, mute) in enumerate(ev):
        seed = (int(midi) * 97 + (i % ROUND_ROBIN) * 7 + (17 if mute else 0)) % 100003
        y = _note(int(midi), float(dur), float(vel), slide_from, bool(mute), seed)
        # monophonic string: damp this note just before the next onset (a chord = onsets within 4 ms)
        later = onsets[i + 1:]
        later = later[later - t > CHORD_WINDOW_S]
        if len(later):
            gap = float(later[0]) - t
            limit = max(0, int((gap - CHOKE_PRE_S) * SR))
            if limit < len(y):
                fl = min(int(CHOKE_FADE_S * SR), len(y) - limit)
                y = y[:limit + fl]
                y[limit:] *= _raised_cos_out(fl)
        i0 = int(round(t * SR))
        if i0 >= n or i0 < 0:
            continue
        e = min(n, i0 + len(y))
        out[i0:e] += y[:e - i0]
    # pickup resonance (passive humbucker-ish); gentle sub-sonic cut that leaves C1 (32.7 Hz) intact
    out = E.eq(out, ('lp', 4200, 1.3), ('hp', 20, 0.7))
    return out


# ------------------------------------------------------------------ amp / tone chain
def _asym_dist(x):
    """two-stage asymmetric waveshaper: stage 1 biased tanh (even harmonics), stage 2 harder,
    with the positive half clipping earlier than the negative half (tube-ish)."""
    bias = 0.18
    y = np.tanh(2.6 * (x + bias)) - np.tanh(2.6 * bias)
    y = np.where(y > 0, np.tanh(2.0 * y) / 2.0, np.tanh(1.3 * y) / 1.3)
    return y


LOW_PRE = 4.0          # gain into the sub band's saturator/compressor
LOW_GAIN = 1.7         # sub band level at the sum
MID_GAIN = 1.1         # grit band level at the sum
LOW_SAT = dict(drive=1.0, mix=0.15)   # very light: the sub stays clean (< ~-30 dB distortion products)
LOW_DET_S = 0.030      # sub compressor detector: running max over one C1 period (15 ms look-ahead)
LOW_COMP = dict(thresh_db=-12, ratio=3.0, attack=0.001, release=0.15, knee=6.0, makeup_db=0.0)
BUS = dict(thresh_db=-16, ratio=2.5, attack=0.003, release=0.09, knee=6.0, makeup_db=0.0)


def low_band(di):
    """clean sub band: LR4 lowpass, very light saturation, compression with a ripple-free detector.
    (A plain peak follower ripples at 2*f0 on a 30 ms period and AM-modulates the sub into sidebands;
    a running max over one period is flat on a steady tone and anticipates the attack by 15 ms.)"""
    low = E.lp(di, CROSS_HZ, order=2) * LOW_PRE
    low = E.saturate(low, LOW_SAT['drive'], LOW_SAT['mix'])
    sc = maximum_filter1d(np.abs(low), size=max(3, int(LOW_DET_S * SR)), mode='nearest')
    low = E.compress(low, sidechain=sc, **LOW_COMP)
    return low * LOW_GAIN


def mid_band(di):
    """dirty mid band: LR4 highpass, pre-EQ, 2-stage asymmetric drive, post-EQ with the grit bump."""
    m = E.hp(di, CROSS_HZ, order=2)
    m = E.eq(m, ('peak', 900, 0.8, 5.0), ('hs', 2500, 0.7, 1.0))
    m = _asym_dist(m * 2.2)
    m = E.eq(m, ('hp', 110, 0.7), ('peak', 1000, 1.0, 3.0), ('peak', 320, 0.9, -4.0),
             ('lp', 3800, 0.8), ('lp', 3800, 0.8))
    return m * MID_GAIN


def bus(y):
    """sum bus: gentle compressor for note weight, final lowpass, true-peak limiter at ~0.5."""
    y = E.compress(y, **BUS)
    y = E.eq(y, ('lp', FINAL_LP_HZ, 0.707), ('lp', FINAL_LP_HZ, 0.707), ('hp', 28, 0.7))
    y = E.limiter(y, ceiling_db=PEAK_CEIL_DB, lookahead=0.003, release=0.06)
    return y


def amp_chain(di):
    """Low/mid split, distortion, compression, final lowpass and peak control. Mono in, mono out."""
    di = np.asarray(di, dtype=np.float64)
    return bus(low_band(di) + mid_band(di))


def render_bass(events, total_len=None):
    """events: (t, midi, dur, vel, slide_from, mute) -> mono array, peak ~0.5."""
    di = render_di(events, total_len)
    if not np.any(di):
        return di
    return amp_chain(di)


__all__ = ['render_bass', 'render_di', 'amp_chain', 'low_band', 'mid_band', 'bus']
