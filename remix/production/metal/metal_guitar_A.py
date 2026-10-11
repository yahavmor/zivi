"""metal_guitar_A.py — Approach A: physical-modelling metal guitar (Drop C, 48 kHz, numpy only).

Signal path
  DI   : extended Karplus-Strong / waveguide per string: one-period shaped-noise + pick-position comb
         excitation, loop filter = per-period loss (T60 at the fundamental) * one-pole HF damping *
         1st-order dispersion allpass (slight inharmonicity) * Thiran fractional-delay allpass (exact
         tuning), processed one period per block with scipy.lfilter (fast, no per-sample Python).
         Bridge-pickup comb + resonant pickup lowpass, pick-noise burst, attack pitch-settle (hard pick
         momentarily sharpens the string), palm mute = much shorter T60 + darker loop + pick near bridge.
         Pitch bends / vibrato / slides / dive-bombs are done by variable-rate reading of the fixed-pitch
         string (the string is rendered long enough to cover the fastest rate).
  Amp  : input HPF -> tubescreamer-style "tight" boost (HPF'd blend + mid push + gentle clip) ->
         4x oversampled chain of 3 inverting asymmetric tanh gain stages with coupling HPF and
         Miller LPF between stages -> tone stack (mid scoop, presence) -> power-amp soft clip ->
         noise gate keyed on the DI -> 4x12 cabinet IR.
  Cab  : synthesised minimum-phase IR from a target magnitude (speaker resonance ~115 Hz, dip ~500 Hz,
         presence 2.5-4 kHz, cone-breakup ripple, steep rolloff >5.5 kHz) + short early-reflection
         noise inside the cab, seed-dependent (L/R takes get different cabs). Cached in ./cache.

Public API (see SPEC.md metal_guitar.py contract)
  render_rhythm(events, seed=0, total_len=None) -> mono        events: (t, root_midi, dur, vel, mute)
  render_rhythm_pair(events, total_len=None) -> stereo         two takes, hard L/R
  render_lead(events, total_len=None) -> mono                  events: (t, midi, dur, vel, bend_semis, vibrato, slide_from)
  render_clean(events, total_len=None) -> stereo               events: (t, midi, dur, vel)
  extras: feedback_swell(midi, dur), pick_scrape(dur), pinch_harmonic(midi), dive_bomb(midi, dur)
  diagnostics: render_rhythm_di, render_lead_di, amp_sim, cab_ir, string_note
"""
import os
import sys
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine as E  # noqa: E402

SR = E.SR
CACHE = os.path.join(HERE, 'cache')
os.makedirs(CACHE, exist_ok=True)
OS = 4                    # oversampling factor inside the distortion chain
CAB_VERSION = 4           # bump whenever cab_ir's design changes (invalidates ./cache/gtrA_cab_*.wav)
_ir_cache = {}


# ============================================================================ small DSP helpers
def _bq(kind, f0, Q=0.707, g_db=0.0, fs=SR):
    """RBJ biquad at an arbitrary sample rate (engine.biquad is fixed to 48 k). returns (b, a)."""
    A = 10 ** (g_db / 40); w0 = 2 * np.pi * f0 / fs; cw, sw = np.cos(w0), np.sin(w0); al = sw / (2 * Q)
    if kind == 'lp': b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == 'hp': b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == 'peak': b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == 'ls':
        s = np.sqrt(A)
        b = [A * ((A + 1) - (A - 1) * cw + 2 * s * al), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - 2 * s * al)]
        a = [(A + 1) + (A - 1) * cw + 2 * s * al, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - 2 * s * al]
    elif kind == 'hs':
        s = np.sqrt(A)
        b = [A * ((A + 1) + (A - 1) * cw + 2 * s * al), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - 2 * s * al)]
        a = [(A + 1) - (A - 1) * cw + 2 * s * al, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - 2 * s * al]
    else:
        raise ValueError(kind)
    b = np.array(b, dtype=np.float64) / a[0]; a = np.array(a, dtype=np.float64) / a[0]
    return b, a


def _eq(x, bands, fs=SR):
    y = x
    for band in bands:
        b, a = _bq(band[0], band[1], band[2] if len(band) > 2 else 0.707, band[3] if len(band) > 3 else 0.0, fs)
        y = signal.lfilter(b, a, y)
    return y


_sos_cache = {}


def _butter(x, f, kind, order=2, fs=SR):
    key = (tuple(f) if np.ndim(f) else float(f), kind, order, fs)
    sos = _sos_cache.get(key)
    if sos is None:
        sos = _sos_cache[key] = signal.butter(order, f, kind, fs=fs, output='sos')
    return signal.sosfilt(sos, x)


def _onepole(x, fc, fs=SR):
    p = np.exp(-2 * np.pi * fc / fs)
    return signal.lfilter([1 - p], [1, -p], x)


def _phase_delay(b, a, f, fs=SR):
    w = 2 * np.pi * f / fs
    _, h = signal.freqz(b, a, worN=[w])
    return float(-np.angle(h[0]) / w)


def _delay_arr(x, d):
    if d <= 0:
        return x
    y = np.zeros_like(x); y[d:] = x[:-d]
    return y


def _place(buf, x, i):
    """add x into buf at sample index i (clipped)."""
    if i < 0:
        x = x[-i:]; i = 0
    n = len(buf)
    if i >= n or len(x) == 0:
        return
    e = min(n, i + len(x)); buf[i:e] += x[:e - i]


def _var_rate_read(src, semis):
    """read src at a time-varying rate 2**(semis/12): pitch modulation of a fixed-pitch string."""
    rate = 2.0 ** (np.asarray(semis, dtype=np.float64) / 12.0)
    pos = np.empty(len(rate)); pos[0] = 0.0; np.cumsum(rate[:-1], out=pos[1:])
    pos = np.minimum(pos, len(src) - 1.001)
    return np.interp(pos, np.arange(len(src)), src)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


_FLOOR = np.random.default_rng(12345).standard_normal(1 << 16) * 1e-13


def _antidenormal(x):
    """add a -260 dB white-noise floor so IIR filter states never decay into subnormal doubles during
    silent stretches (measured: sosfilt runs ~40x slower on denormal states)."""
    return x + np.resize(_FLOOR, len(x))


# ============================================================================ string model
def _loop_filter(f0, t60, hf_pole, disp):
    """Design the KS loop filter g*LP1(hf_pole)*AP(disp)*AP(thiran) so that the total loop delay is exactly
    SR/f0 samples. Returns (N, b, a): integer delay N and loop IIR coefficients."""
    P = SR / f0
    b_lp = np.array([1.0 - hf_pole]); a_lp = np.array([1.0, -hf_pole])
    b_fix = np.convolve(b_lp, [disp, 1.0]); a_fix = np.convolve(a_lp, [1.0, disp])
    d_fix = _phase_delay(b_fix, a_fix, f0)
    N = max(2, int(np.floor(P - d_fix - 0.5)))
    d = P - N - d_fix                      # fractional part for the Thiran allpass, in [0.5, 1.5)
    dp = d
    for _ in range(6):
        c = (1 - dp) / (1 + dp)
        b = np.convolve(b_fix, [c, 1.0]); a = np.convolve(a_fix, [1.0, c])
        err = P - N - _phase_delay(b, a, f0)
        if abs(err) < 1e-4:
            break
        dp += err
    _, h = signal.freqz(b_lp, a_lp, worN=[2 * np.pi * f0 / SR])
    g = 10 ** (-3.0 / (t60 * f0)) / abs(h[0])
    return N, b * g, a


def _ks_run(exc, N, b, a):
    """y[n] = exc[n] + (h * y)[n - N], computed one period per block."""
    n = len(exc); y = np.zeros(n)
    if n <= N:
        return exc.copy()
    y[:N] = exc[:N]
    zi = np.zeros(max(len(a), len(b)) - 1)
    s = N
    while s < n:
        e = min(s + N, n)
        v, zi = signal.lfilter(b, a, y[s - N:e - N], zi=zi)
        y[s:e] = exc[s:e] + v
        s = e
    return y


def string_note(f0, n, vel=0.9, mute=False, seed=0, pick_pos=None, t60=None, hf_pole=None, bright=None,
                click=True, pickup=True, settle=True):
    """One plucked electric-guitar string at f0 Hz, n samples, DI level (hard pick ~0.35 peak).
    mute=True -> palm mute: short T60, dark loop, pick close to the bridge, stronger pick thud."""
    r = np.random.default_rng(seed)
    vel = float(np.clip(vel, 0.05, 1.0))
    lowness = (65.4 / f0) ** 0.5                  # low strings: longer sustain, less HF
    if mute:
        t60 = t60 if t60 is not None else 0.20 * lowness ** 0.5 * (0.85 + 0.3 * r.random())
        hf_pole = hf_pole if hf_pole is not None else 0.72 + 0.06 * (1 - vel)
        pick_pos = pick_pos if pick_pos is not None else 0.07 + 0.03 * r.random()
        bright = bright if bright is not None else 900 + 1500 * vel
    else:
        t60 = t60 if t60 is not None else 5.5 * lowness * (0.9 + 0.2 * r.random())
        hf_pole = hf_pole if hf_pole is not None else 0.30 + 0.12 * (1 - vel) + 0.12 * (1 - min(1, lowness))
        pick_pos = pick_pos if pick_pos is not None else 0.11 + 0.05 * r.random()
        bright = bright if bright is not None else 1500 + 4500 * vel
    N, b, a = _loop_filter(f0, t60, hf_pole, disp=-0.22)
    # --- excitation (one period): 1/k pluck body (what a velocity pickup sees from a plucked string)
    #     + broadband pick-edge noise + coherent pick-release pulse, all through the pick-position comb
    nz = r.standard_normal(N) * (0.6 + 0.4 * np.linspace(1, 0.4, N))
    body = _onepole(nz, 2.2 * f0); body /= (np.sqrt(np.mean(body ** 2)) + 1e-9)
    edge = _onepole(nz, bright); edge /= (np.sqrt(np.mean(edge ** 2)) + 1e-9)
    M = max(1, int(round(pick_pos * N)))
    pulse = np.zeros(N); w = max(2, int(0.35 * M)); pulse[:w] = np.hanning(2 * w)[w:]        # pick release
    pulse = _onepole(pulse, bright * 1.5); pulse /= (np.abs(pulse).max() + 1e-9)
    exc1 = 0.8 * body + (0.25 + 0.2 * vel) * edge + 0.9 * pulse * (0.8 if mute else 1.0)
    n_src = n + N + M
    exc = np.zeros(n_src)
    exc[:N] += exc1; exc[M:M + N] -= exc1                                                      # comb (1 - z^-M)
    y = _ks_run(exc, N, b, a)
    if pickup:  # bridge humbucker ~5 cm from the bridge on a 64.8 cm scale (shallow: wide magnetic aperture)
        pp = min(0.45, 0.075 * f0 / 65.4)
        y = y - 0.5 * _delay_arr(y, max(1, int(round(pp * N))))
    if settle:  # hard pick momentarily sharpens the string, settles in ~40 ms
        tt = np.arange(n) / SR
        semis = (0.28 if mute else 0.18) * vel ** 2 * np.exp(-tt / 0.045)
        y = _var_rate_read(y, semis)
    else:
        y = y[:n]
    y = y * (0.07 * vel ** 1.3)
    if click:
        pn = _pick_noise(min(n, int(0.008 * SR)), vel, mute, r)
        y[:len(pn)] += pn
    return y


def _pick_noise(n, vel, mute, r):
    """the 'chk' of the pick hitting the string: 2-8 kHz burst, ~2 ms."""
    t = np.arange(n) / SR
    z = r.standard_normal(n) * np.exp(-t / (0.0016 if mute else 0.0011))
    z = _butter(z, [1800, 9000], 'band', 2)
    return z * 0.018 * vel ** 2 * (1.5 if mute else 1.0)


def _palm_thud(vel, r, f_thud=95.0):
    """the palm/heel landing on the strings at the bridge on a palm mute: a short unpitched low thud
    (~90-110 Hz, 20 ms) + a low-passed noise burst. Added once per chug to the DI."""
    n = int(0.06 * SR); t = np.arange(n) / SR
    f = f_thud * (0.92 + 0.16 * r.random())
    tone = np.sin(2 * np.pi * f * t + r.uniform(0, 1.0)) * np.exp(-t / 0.018)
    nz = _butter(r.standard_normal(n), 160, 'low', 2) * np.exp(-t / 0.012) * 2.5
    return (tone + nz) * 0.11 * vel ** 2


# ============================================================================ DI renderers
def _chord_strings(root):
    return [(root, 1.0), (root + 7, 0.8), (root + 12, 0.62)]


def render_rhythm_di(events, seed=0, total_len=None, take=0):
    """DI stream for (t, root_midi, dur, vel, mute) events. take 0/1 -> different jitter sign and detune."""
    r = np.random.default_rng(1000 + seed)
    if not events:
        return np.zeros(int((total_len or 1.0) * SR))
    end = max(t + d for t, _, d, _, _ in events) + 1.0
    n_tot = int((total_len if total_len else end) * SR)
    out = np.zeros(n_tot)
    glob_cents = r.uniform(-4, 4)
    for k, (t, root, dur, vel, mute) in enumerate(events):
        vel = float(np.clip(vel, 0.05, 1.0))
        if take == 0:
            jit = 0.0 if seed == 0 else r.normal(0, 0.0015)
        else:
            jit = (-1 if take == 1 else 1) * r.uniform(0.003, 0.006) + r.normal(0, 0.001)
        t0 = t + jit
        if mute:
            n_note = int(max(0.40, dur + 0.10) * SR)
        else:
            n_note = int((dur + 0.06) * SR)
        note = np.zeros(n_note)
        strum = r.uniform(0.0008, 0.0018)
        for si, (m, amp) in enumerate(_chord_strings(root)):
            cents = glob_cents + r.uniform(-3, 3)
            f0 = E.midi2hz(m) * 2 ** (cents / 1200)
            v = vel * (1.0 if si == 0 else (0.75 if mute else 0.9))
            ns = n_note - int(si * strum * SR)
            s = string_note(f0, ns, v * (0.9 + 0.1 * r.random()), mute, seed=int(r.integers(1 << 30)))
            if mute and si > 0:
                s *= 0.7
            _place(note, s * amp, int(si * strum * SR))
        if mute:
            _place(note, _palm_thud(vel, r), 0)
        # release: fingers/palm stop the strings at dur (open), mutes die naturally + tidy fade
        if mute:
            i0 = int(max(dur + 0.06, 0.16) * SR); fl = int(0.06 * SR)
        else:
            i0 = int(dur * SR); fl = int(0.04 * SR)
        if i0 < n_note:
            e = min(n_note, i0 + fl)
            note[i0:e] *= np.linspace(1, 0, e - i0) ** 0.7
            note[e:] = 0
        _place(out, note, int(round(t0 * SR)))
    out = _eq(_antidenormal(out), [('lp', 3800, 1.4)])          # pickup resonance / LC rolloff
    return out


def _lead_pitch_track(n, dur, bend, vib, slide_semis, r, vib_rate=5.5):
    """semitone offset per sample for a lead note: slide-in, bend-up, finger vibrato (bend-up only),
    attack settle. Returns semis array."""
    t = np.arange(n) / SR
    s = np.zeros(n)
    if slide_semis:
        tau = 0.10 if abs(slide_semis) <= 3 else 0.16
        s += slide_semis * (1 - _smoothstep(t / tau))
    t_b = 0.0
    if bend:
        t_b = float(np.clip(0.4 * dur, 0.15, 0.45))
        s += bend * _smoothstep(t / t_b) ** 1.3
    if vib:
        start = max(t_b, 0.12 if slide_semis else 0.18)
        depth = vib * _smoothstep((t - start) / 0.35)
        rate = vib_rate * (0.95 + 0.1 * r.random())
        s += depth * (0.5 - 0.5 * np.cos(2 * np.pi * rate * (t - start) + r.uniform(0, 0.4)))
    return s


def render_lead_di(events, total_len=None, seed=5):
    """DI stream for lead events (t, midi, dur, vel, bend_semis, vibrato, slide_from)."""
    r = np.random.default_rng(seed)
    if not events:
        return np.zeros(int((total_len or 1.0) * SR))
    end = max(e[0] + e[2] for e in events) + 1.5
    n_tot = int((total_len if total_len else end) * SR)
    out = np.zeros(n_tot)
    evs = sorted(events, key=lambda e: e[0])
    for k, ev in enumerate(evs):
        t, midi, dur, vel = ev[0], int(ev[1]), float(ev[2]), float(np.clip(ev[3], 0.05, 1.0))
        bend = float(ev[4] or 0.0) if len(ev) > 4 else 0.0
        vib = float(ev[5] or 0.0) if len(ev) > 5 else 0.0
        sf = ev[6] if len(ev) > 6 else None
        nxt = evs[k + 1][0] if k + 1 < len(evs) else 1e9
        rel = 0.25
        n_note = int((min(dur, max(0.05, nxt - t + 0.02)) + rel) * SR)
        semis = _lead_pitch_track(n_note, dur, bend, vib, (float(sf) - midi) if sf is not None else 0.0, r)
        tt = np.arange(n_note) / SR
        semis += 0.12 * vel ** 2 * np.exp(-tt / 0.035)
        f0 = E.midi2hz(midi) * 2 ** (r.uniform(-2, 2) / 1200)
        n_src = int(np.ceil((2.0 ** (semis / 12)).sum())) + 16
        t60 = float(np.clip(6.0 * (130.8 / f0) ** 0.4, 1.5, 7.0))
        src = string_note(f0, n_src, vel * (0.6 if sf is not None else 1.0), False, seed=int(r.integers(1 << 30)),
                          pick_pos=0.13 + 0.05 * r.random(), t60=t60, hf_pole=0.22 + 0.1 * (1 - vel),
                          bright=2500 + 4000 * vel, settle=False)
        y = _var_rate_read(src, semis)
        i0 = int(min(dur, max(0.05, nxt - t + 0.02)) * SR); fl = int(min(rel, 0.12) * SR)
        if i0 < n_note:
            e = min(n_note, i0 + fl); y[i0:e] *= np.linspace(1, 0, e - i0) ** 0.5; y[e:] = 0
        _place(out, y, int(round(t * SR)))
    out = _eq(_antidenormal(out), [('lp', 4200, 1.3)])
    return out


# ============================================================================ cabinet
def cab_ir(seed=0, kind='4x12', length=0.07):
    """Synthesised 4x12 impulse response (minimum phase from a target magnitude + cab reflections)."""
    key = (seed, kind, length)
    if key in _ir_cache:
        return _ir_cache[key]
    path = os.path.join(CACHE, 'gtrA_cab_%s_%d_v%d.wav' % (kind, seed, CAB_VERSION))
    if os.path.exists(path):
        ir = E.read(path); ir = ir if ir.ndim == 1 else ir[:, 0]
        _ir_cache[key] = ir
        return ir
    r = np.random.default_rng(100 + seed)
    nfft = 16384
    f = np.fft.rfftfreq(nfft, 1 / SR)
    w = 2 * np.pi * f / SR
    mag = np.ones_like(f)

    def apply(b, a):
        nonlocal mag
        _, h = signal.freqz(b, a, worN=w)
        mag = mag * np.abs(h)
    if kind == '4x12':
        for sec in [('hp', 62, 0.8), ('peak', 105 + r.uniform(-8, 8), 1.1, 5.5), ('peak', 210, 1.0, 1.5),
                    ('peak', 520 + r.uniform(-40, 40), 0.75, -4.5), ('peak', 1150, 2.0, -1.0),
                    ('peak', 1750 + r.uniform(-100, 100), 1.6, 0.5), ('peak', 3000 + r.uniform(-200, 200), 1.1, 1.5),
                    ('peak', 4300 + r.uniform(-200, 200), 2.5, 1.0), ('hs', 6500, 0.7, -4.0)]:
            apply(*_bq(*sec))
        sos = signal.butter(4, 5600, 'low', fs=SR, output='sos'); _, h = signal.sosfreqz(sos, worN=w); mag *= np.abs(h)
        sos = signal.butter(2, 7200, 'low', fs=SR, output='sos'); _, h = signal.sosfreqz(sos, worN=w); mag *= np.abs(h)
        ripple_db = 0.9; refl = 0.07
    elif kind == 'lead':   # same cab, mic a little more on-axis: more 3-5 kHz, less low thump
        for sec in [('hp', 95, 0.8), ('peak', 125, 1.2, 2.0), ('peak', 480, 0.8, -3.0), ('peak', 1600, 1.6, 2.0),
                    ('peak', 3100 + r.uniform(-150, 150), 1.0, 3.0), ('peak', 4600, 2.0, 1.0), ('hs', 7000, 0.7, -5.0)]:
            apply(*_bq(*sec))
        sos = signal.butter(4, 6000, 'low', fs=SR, output='sos'); _, h = signal.sosfreqz(sos, worN=w); mag *= np.abs(h)
        sos = signal.butter(2, 8000, 'low', fs=SR, output='sos'); _, h = signal.sosfreqz(sos, worN=w); mag *= np.abs(h)
        ripple_db = 0.9; refl = 0.06
    else:                  # 'clean': open-back 2x12-ish, brighter and smoother
        for sec in [('hp', 80, 0.8), ('peak', 140, 1.0, 2.0), ('peak', 600, 0.8, -2.0), ('peak', 2600, 1.2, 2.5),
                    ('peak', 4500, 2.0, 1.0)]:
            apply(*_bq(*sec))
        sos = signal.butter(3, 7500, 'low', fs=SR, output='sos'); _, h = signal.sosfreqz(sos, worN=w); mag *= np.abs(h)
        ripple_db = 0.7; refl = 0.05
    # cone break-up ripple above ~700 Hz (smooth random in log-f) and a faint mic-distance comb
    lf = np.log(np.maximum(f, 20.0))
    grid = np.linspace(np.log(700), np.log(SR / 2), 24)
    rip = np.interp(lf, grid, r.standard_normal(len(grid)))
    rip = rip * np.clip((f - 600) / 600, 0, 1)
    mag *= 10 ** (ripple_db * rip / 20)
    tau = 0.00018 + 0.00008 * r.random()
    mag *= 1 + 0.10 * np.cos(2 * np.pi * f * tau + r.uniform(0, 6.28))
    # minimum phase via the real cepstrum
    logm = np.log(np.maximum(mag, 1e-5))
    c = np.fft.irfft(logm, nfft)
    fold = np.zeros(nfft); fold[0] = 1; fold[1:nfft // 2] = 2; fold[nfft // 2] = 1
    h = np.fft.irfft(np.exp(np.fft.rfft(c * fold, nfft)), nfft)
    n_ir = int(length * SR)
    ir = h[:n_ir].copy()
    k = int(0.4 * n_ir); ir[n_ir - k:] *= np.hanning(2 * k)[k:]
    # early reflections inside the cab / off the floor: short decaying band-limited noise
    t = np.arange(n_ir) / SR
    nz = r.standard_normal(n_ir) * np.exp(-t / 0.006) * (t > 0.0012)
    nz = _butter(_butter(nz, 150, 'high', 2), 2500, 'low', 4)
    ir += refl * nz / (np.abs(nz).max() + 1e-9) * np.abs(ir).max()
    # normalise so the 200 Hz - 2 kHz band has unity magnitude
    H = np.abs(np.fft.rfft(ir, nfft)); sel = (f > 200) & (f < 2000)
    ir /= np.exp(np.mean(np.log(H[sel] + 1e-12)))
    E.write(path, ir)
    _ir_cache[key] = ir
    return ir


# ============================================================================ amp simulation
PRESETS = {
    'rhythm': dict(hp_in=75, ts_hp=420, ts_mix=0.45, ts_mid=(820, 0.9, 3.0), ts_gain=2.2, pre_lp=6000,
                   stages=[(16.0, 0.12), (15.0, 0.14), (9.0, 0.10)], stage_hp=28, stage_lp=7000,
                   tone=[('peak', 560, 0.9, -4.5), ('ls', 170, 0.7, 2.0), ('peak', 3300, 1.0, 0.5), ('hs', 6800, 0.7, -4.0)],
                   power=1.4, gate=(-56, 0.004, 0.03), cab='4x12', out=0.14),
    'lead': dict(hp_in=140, ts_hp=700, ts_mix=0.75, ts_mid=(950, 0.8, 6.0), ts_gain=3.0, pre_lp=6500,
                 stages=[(18.0, 0.12), (17.0, 0.14), (12.0, 0.10)], stage_hp=35, stage_lp=7500,
                 tone=[('peak', 600, 1.0, -2.0), ('peak', 1400, 1.2, 2.0), ('peak', 3400, 1.0, 2.0), ('hs', 7000, 0.7, -4.0)],
                 power=1.6, gate=(-62, 0.004, 0.06), cab='lead', out=0.21),
    'clean': dict(hp_in=70, ts_hp=0, ts_mix=0.0, ts_mid=None, ts_gain=1.0, pre_lp=9000,
                  stages=[(8.0, 0.05)], stage_hp=20, stage_lp=9000,
                  tone=[('peak', 400, 0.9, -1.5), ('peak', 2800, 1.0, 2.0)],
                  power=1.0, gate=None, cab='clean', out=1.6),
}


def _shape(x, bias):
    """inverting tube-ish stage: asymmetric soft clip (positive half clips sooner than negative)."""
    z = x + bias
    neg = z < 0
    zz = np.where(neg, z * 0.72, z)        # one tanh for both halves, rescale the negative half after
    y = np.tanh(zz)
    y[neg] /= 0.72
    return -(y - np.tanh(bias))


def _gate_gain(di, thr_db, attack, release, hold=0.012):
    """noise-gate gain curve keyed on the DI level (hop 48 = 1 ms control rate)."""
    hop = 48; n = len(di); npts = n // hop + 1
    lvl = np.abs(di[:npts * hop].copy()); lvl.resize(npts * hop); lvl = lvl.reshape(npts, hop).max(1)
    env = E._env_follow(lvl, 0.0005, 0.012, hop)
    ldb = 20 * np.log10(env + 1e-9)
    g = _smoothstep((ldb - thr_db + 6) / 12.0)
    # hold: keep max over the next `hold` ms, then attack/release smoothing
    from scipy.ndimage import maximum_filter1d
    hk = max(1, int(hold * SR / hop))
    g = maximum_filter1d(g, size=hk, origin=(hk - 1) // 2)      # max over the PAST `hold` ms
    g = E._env_follow(g, attack, release, hop)
    return np.interp(np.arange(n), np.arange(npts) * hop, g)


_aa_sos_cache = {}


def _aa_sos(fs):
    """anti-image / anti-alias lowpass for the OS-times rate: 8th-order elliptic, 20 kHz, 90 dB stop."""
    if fs not in _aa_sos_cache:
        _aa_sos_cache[fs] = signal.ellip(8, 0.05, 90, 20000, fs=fs, output='sos')
    return _aa_sos_cache[fs]


def _gain_stages_os(x, P, chunk=1 << 15):
    """OS-times oversampled chain of inverting asymmetric gain stages with coupling HPF + Miller LPF,
    processed in ~0.7 s chunks with carried filter state. Chunking keeps every temporary array small
    (a 113 s song at 4x would otherwise mean 170 MB temporaries, which this box processes ~10x slower
    per sample); the up/down sampling uses stateful IIR filters so chunk boundaries are seamless."""
    n = len(x); fs = SR * OS
    aa = _aa_sos(fs)
    st_sos = []
    for _ in P['stages']:
        b1, a1 = _bq('hp', P['stage_hp'], 0.6, fs=fs); b2, a2 = _bq('lp', P['stage_lp'], 0.7, fs=fs)
        st_sos.append(np.array([np.r_[b1, a1], np.r_[b2, a2]]))
    zi_up = np.zeros((aa.shape[0], 2)); zi_dn = np.zeros((aa.shape[0], 2))
    zi_st = [np.zeros((2, 2)) for _ in P['stages']]
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk); blk = x[s:e]
        xs = np.zeros(len(blk) * OS); xs[::OS] = blk * OS                 # zero-stuff, keep amplitude
        xs, zi_up = signal.sosfilt(aa, xs, zi=zi_up)
        for i, (gdb, bias) in enumerate(P['stages']):
            xs = _shape(xs * E.db(gdb), bias)
            xs, zi_st[i] = signal.sosfilt(st_sos[i], xs, zi=zi_st[i])
        xs, zi_dn = signal.sosfilt(aa, xs, zi=zi_dn)
        out[s:e] = xs[::OS]
    return out


def amp_sim(di, preset='rhythm', seed=0, gate=True):
    """Full amp + cab on a DI stream (mono float). Output peak ~0.5 for a hard chord."""
    P = PRESETS[preset] if isinstance(preset, str) else preset
    x = _antidenormal(np.asarray(di, dtype=np.float64))
    n = len(x)
    x = _butter(x, P['hp_in'], 'high', 2)
    # tubescreamer-style tight boost: HPF'd blend + mid push + gentle clip
    if P['ts_mix'] > 0:
        x = x * (1 - P['ts_mix']) + _butter(x, P['ts_hp'], 'high', 1) * P['ts_mix']
        if P['ts_mid']:
            x = _eq(x, [('peak',) + tuple(P['ts_mid'])])
        x = np.tanh(x * P['ts_gain'] * 4.0) / 4.0 * 1.0
    x = _eq(x, [('lp', P['pre_lp'], 0.6)])
    x = _gain_stages_os(x, P)
    x = _eq(x, P['tone'])
    x = np.tanh(x * P['power']) / np.tanh(P['power'])          # power amp
    ir = cab_ir(seed, P['cab'])
    y = signal.fftconvolve(x, ir)[:n]
    y = _butter(y, 40, 'high', 2)                              # DC safety (before the gate: no filter sees gate zeros)
    if gate and P['gate']:
        thr, att, rel = P['gate']
        y = y * _gate_gain(di, thr, att, rel)
    y = y * P['out']
    return np.clip(y, -0.85, 0.85)                             # safety only; `out` is calibrated for ~0.5 peaks


# ============================================================================ public renderers
def render_rhythm(events, seed=0, total_len=None):
    """(t, root_midi, dur, vel, mute) -> mono distorted rhythm guitar."""
    di = render_rhythm_di(events, seed=seed, total_len=total_len, take=0)
    return amp_sim(di, 'rhythm', seed=seed)


def render_rhythm_pair(events, total_len=None):
    """two independent takes (seeds 1/2, +-3..6 ms jitter, detune, different cab) hard-panned L/R."""
    L = amp_sim(render_rhythm_di(events, seed=1, total_len=total_len, take=1), 'rhythm', seed=1)
    R = amp_sim(render_rhythm_di(events, seed=2, total_len=total_len, take=2), 'rhythm', seed=2)
    return np.stack([L, R], 1)


def render_lead(events, total_len=None):
    """(t, midi, dur, vel, bend_semis, vibrato, slide_from) -> mono lead guitar."""
    di = render_lead_di(events, total_len=total_len)
    y = amp_sim(di, 'lead', seed=3)
    y = E.compress(y, -16, 2.5, 0.01, 0.15, makeup_db=2.0)
    return np.clip(y, -0.8, 0.8)


def render_clean(events, total_len=None):
    """(t, midi, dur, vel) -> stereo clean/ambient guitar with chorus + reverb."""
    r = np.random.default_rng(9)
    if not events:
        return np.zeros((int((total_len or 1.0) * SR), 2))
    end = max(t + d for t, _, d, _ in events) + 3.0
    n_tot = int((total_len if total_len else end) * SR)
    out = np.zeros((n_tot, 2))
    evs = sorted(events, key=lambda e: e[0])
    for k, (t, midi, dur, vel) in enumerate(evs):
        vel = float(np.clip(vel, 0.05, 1.0))
        f0 = E.midi2hz(int(midi)) * 2 ** (r.uniform(-3, 3) / 1200)
        n_note = int((dur + 1.2) * SR)
        s = string_note(f0, n_note, vel, False, seed=int(r.integers(1 << 30)), pick_pos=0.2 + 0.08 * r.random(),
                        t60=float(np.clip(5.0 * (130.8 / f0) ** 0.5, 1.5, 7.0)), hf_pole=0.3, bright=1500 + 3000 * vel)
        i0 = int(dur * SR); fl = int(0.5 * SR)
        if i0 < n_note:
            e = min(n_note, i0 + fl); s[i0:e] *= np.exp(-np.arange(e - i0) / SR / 0.12); s[e:] = 0
        p = 0.35 * np.sin(k * 1.7) + r.uniform(-0.15, 0.15)
        _place(out[:, 0], s * np.cos((p + 1) / 4 * np.pi), int(round(t * SR)))
        _place(out[:, 1], s * np.sin((p + 1) / 4 * np.pi), int(round(t * SR)))
    out = np.stack([_eq(_antidenormal(out[:, c]), [('lp', 4200, 1.1)]) for c in range(2)], 1)     # pickup resonance
    y = np.stack([amp_sim(out[:, c], 'clean', seed=10 + c, gate=False) for c in range(2)], 1)
    y = E.compress(y, -22, 3.0, 0.006, 0.18, makeup_db=4.0)
    y = E.chorus(y, rate=0.7, depth=0.0028, mix=0.38, voices=2, base=0.011)
    y = E.reverb(y, decay=2.2, predelay=0.025, hf_damp=0.5, mix=0.22, lowcut=180, highcut=7000, seed=31)
    pk = np.abs(y).max()
    if pk > 1e-6:
        y = y * (0.4 / pk)                                      # texture layer: consistent 0.4 peak
    return y


# ============================================================================ extras
def feedback_swell(midi, dur, vel=0.9):
    """a held note that blooms into amp feedback (3rd partial, slow vibrato) over dur seconds."""
    r = np.random.default_rng(21)
    n = int((dur + 0.5) * SR); t = np.arange(n) / SR
    f0 = E.midi2hz(midi)
    src = string_note(f0, n + 64, vel, False, seed=22, t60=6.0, hf_pole=0.25, settle=False)
    semis = 0.1 * np.exp(-t / 0.04) + 0.25 * _smoothstep((t - 0.4 * dur) / (0.5 * dur)) * (0.5 - 0.5 * np.cos(2 * np.pi * 4.8 * t))
    y = _var_rate_read(src, semis)
    fb_env = _smoothstep((t - 0.25 * dur) / (0.55 * dur))
    ffb = 3 * f0 * 2 ** (semis / 12)
    fb = np.sin(2 * np.pi * np.cumsum(ffb) / SR) * fb_env * 0.05
    y = y + fb
    fade = int(0.3 * SR); y[-fade:] *= np.linspace(1, 0, fade)
    out = amp_sim(_eq(y, [('lp', 4200, 1.3)]), 'lead', seed=3)
    return E.compress(out, -16, 2.5, 0.01, 0.15, makeup_db=2.0)


def pick_scrape(dur=1.2, seed=0):
    """pick edge dragged along a wound low string towards the neck: falling resonance, through the amp."""
    r = np.random.default_rng(40 + seed)
    n = int(dur * SR); t = np.arange(n) / SR
    rate = 420 * (1 + 0.35 * np.sin(2 * np.pi * 0.9 * t))          # winding-crossing rate with hand speed wobble
    ph = np.cumsum(rate) / SR
    train = (np.diff(np.floor(ph), prepend=0) > 0).astype(float) * r.uniform(0.4, 1.0, n)
    z = train + 0.35 * r.standard_normal(n)
    fc = np.geomspace(3200, 260, n)
    y = E.lp_sweep_blocks(z, fc, Q=3.0, block=128)
    y = E.hp(y, 120, 0.7, 1)
    y *= np.linspace(0.6, 1.0, n) * (1 - np.exp(-t / 0.03))
    y[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    y = y * 0.25 / (np.abs(y).max() + 1e-9)
    return amp_sim(y, 'rhythm', seed=1)


def pinch_harmonic(midi, dur=1.2, vel=1.0):
    """squealing pinch harmonic: damped fundamental 'chk' + the 3rd partial ringing with wide vibrato."""
    r = np.random.default_rng(55)
    n = int((dur + 0.3) * SR); t = np.arange(n) / SR
    f0 = E.midi2hz(midi)
    fund = string_note(f0, n, vel, True, seed=56, t60=0.06)
    harm_src = string_note(3 * f0, n + 128, vel, False, seed=57, pick_pos=0.3, t60=4.0, hf_pole=0.2, bright=7000,
                           click=False, settle=False)
    semis = 0.5 * _smoothstep((t - 0.08) / 0.2) * (0.5 - 0.5 * np.cos(2 * np.pi * 5.8 * t)) + 0.15 * np.exp(-t / 0.05)
    harm = _var_rate_read(harm_src, semis) * 2.2
    y = fund * 0.6 + harm
    fade = int(0.2 * SR); y[-fade:] *= np.linspace(1, 0, fade)
    out = amp_sim(_eq(y, [('lp', 4500, 1.3)]), 'lead', seed=3)
    return E.compress(out, -16, 2.5, 0.01, 0.15, makeup_db=2.0)


def dive_bomb(midi, dur=2.0, vel=1.0, depth=-26.0):
    """whammy dive: picked note drops `depth` semitones over dur with a wobble, through the lead amp."""
    r = np.random.default_rng(66)
    n = int((dur + 0.4) * SR); t = np.arange(n) / SR
    f0 = E.midi2hz(midi)
    x = np.clip(t / dur, 0, 1)
    semis = depth * (x ** 1.6) + 0.12 * np.exp(-t / 0.04)
    semis += 0.6 * np.sin(2 * np.pi * 4.0 * t) * _smoothstep((t - 0.55 * dur) / (0.3 * dur))
    n_src = int(np.ceil((2.0 ** (semis / 12)).sum())) + 64
    src = string_note(f0, n_src, vel, False, seed=67, t60=6.0, hf_pole=0.22, settle=False)
    y = _var_rate_read(src, semis)
    fade = int(0.3 * SR); y[-fade:] *= np.linspace(1, 0, fade)
    out = amp_sim(_eq(y, [('lp', 4200, 1.3)]), 'lead', seed=3)
    return E.compress(out, -16, 2.5, 0.01, 0.15, makeup_db=2.0)
