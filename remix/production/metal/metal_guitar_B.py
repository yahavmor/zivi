"""metal_guitar_B.py -- approach B: oscillator-based metal guitar (rhythm / lead / clean + extras).

Signal path (all float64, 48 kHz):
  STRING (per string of the chord):
      band-limited PolyBLEP sawtooth + 2nd saw detuned by a few cents + sub-octave pulse (short "thump")
      + a pulse-shaped pluck component (saw minus shifted saw = ideal plucked-string spectrum, the shift being
        the pick position), attack pitch-sharpening (tension), exponential decay envelopes (fast for palm mute,
        long for open chords), a pick transient (filtered noise burst + click, plus a palm "thud" for mutes),
        and a time-varying lowpass that closes as the string decays. Strings of a chord are strum-offset by
        a couple of ms and summed BEFORE the amp (inter-string intermodulation = power-chord growl).
      -> pickup model (humbucker resonance peak ~2.9 kHz, 12 dB/oct roll-off above)
  AMP (whole DI stream, 2x oversampled around the non-linear stages):
      tight high-pass + "tube-screamer" mid pre-boost -> 3 asymmetric soft-clip stages (biased tanh, DC-blocked,
      inter-stage lowpass, a little sine wave-folding on the hottest stage) -> power stage with sag ->
      tone stack (low shelf, 550 Hz mid scoop, presence) -> noise gate keyed from the DI envelope
  CAB: filter bank (peaks 110 Hz / 2.8 kHz, dip 550 Hz, 8th-order Butterworth lowpass 5.5 kHz, HP 72 Hz)
      + a few sub-ms "mic/cone" reflections for comb ripple.
  Double tracking: two takes with different seeds, +-3..6 ms timing jitter, pitch/pick/amp variation, hard L/R.

Public API (contract from SPEC.md, metal_guitar.py):
  render_rhythm(events, seed=0, total_len=None) -> mono   events: (t, root_midi, dur, vel, mute)
  render_rhythm_pair(events, total_len=None)   -> stereo
  render_lead(events, total_len=None)          -> mono   events: (t, midi, dur, vel, bend_semis, vibrato, slide_from)
  render_clean(events, total_len=None)         -> stereo events: (t, midi, dur, vel)
  extras: render_rhythm_di(events, seed, total_len), amp_sim(di, mode, variant), feedback_swell(midi, dur),
          pick_scrape(dur), pinch_harmonic(midi, dur), dive_bomb(midi, dur)
"""
import os
import sys
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(HERE, '..'))
import engine as E

SR = E.SR
CACHE = os.path.join(HERE, 'cache')
os.makedirs(CACHE, exist_ok=True)

RHYTHM_TAIL = 2.5     # seconds of audio kept after the last rhythm event
LEAD_TAIL = 2.0
CLEAN_TAIL = 3.0


# =====================================================================================================
#  oscillators
# =====================================================================================================
def _t(n):
    return np.arange(n) / SR


def _phase(freq, ph0=0.0):
    """freq: per-sample Hz array -> (phase in [0,1), per-sample increment)."""
    dt = np.asarray(freq, dtype=np.float64) / SR
    ph = (np.cumsum(dt) + ph0) % 1.0
    return ph, dt


def _blep_saw(ph, dt):
    """PolyBLEP band-limited sawtooth (-1..1) for a time-varying phase increment."""
    y = 2.0 * ph - 1.0
    m = ph < dt
    if m.any():
        t = ph[m] / dt[m]
        y[m] -= (2.0 * t - t * t - 1.0)
    m = ph > (1.0 - dt)
    if m.any():
        t = (ph[m] - 1.0) / dt[m]
        y[m] -= (t * t + 2.0 * t + 1.0)
    return y


def _blep_pulse(ph, dt, width):
    """band-limited pulse of duty 'width' = saw(ph) - saw(ph + width). (has DC = 2*width-1 -> removed later)"""
    return _blep_saw(ph, dt) - _blep_saw((ph + width) % 1.0, dt)


def _smooth_noise(n, rate_hz, r, zi=None):
    """slow random drift (one-pole smoothed white noise), unit-ish variance."""
    b, a = E.biquad('lp', rate_hz, 0.7)
    x = signal.lfilter(b, a, r.standard_normal(n))
    return x / (np.std(x) + 1e-9)


# =====================================================================================================
#  string voice
# =====================================================================================================
def _string(freq, vel, r, mute=False, pick_pos=None, detune_cents=None, sub=0.18, saw2=0.55, sharpen=0.012):
    """One string's raw vibration (pre pickup). freq = per-sample Hz array. returns mono array."""
    n = len(freq)
    t = _t(n)
    f1 = freq * (1.0 + sharpen * (0.5 + 0.5 * vel) * np.exp(-t / 0.035))        # tension sharpening at the pick
    if detune_cents is None:
        detune_cents = 3.0 + 3.0 * r.random()
    if pick_pos is None:
        pick_pos = (0.085 + 0.03 * r.random()) if mute else (0.11 + 0.05 * r.random())
    ph1, dt1 = _phase(f1, r.random())
    saw_a = _blep_saw(ph1, dt1)
    y = 0.55 * saw_a + 0.75 * (saw_a - _blep_saw((ph1 + pick_pos) % 1.0, dt1))   # saw + plucked-string pulse
    ph2, dt2 = _phase(f1 * 2 ** (detune_cents / 1200.0), r.random())
    y += saw2 * _blep_saw(ph2, dt2)
    if sub > 0:                                                                 # sub-octave pulse: attack thump only
        ph3, dt3 = _phase(f1 * 0.5, r.random())
        y += sub * _blep_pulse(ph3, dt3, 0.22) * np.exp(-t / (0.06 if mute else 0.09))
    return y


def _pick_transient(vel, mute, r, bright=1.0):
    k = int(0.008 * SR)
    t = _t(k)
    nz = r.standard_normal(k) * np.exp(-t / 0.0011)
    nz = E.eq(nz, ('bp', 1800 if mute else 2600, 0.6), ('hp', 700, 0.7))
    click = np.sin(2 * np.pi * 3100 * t + r.random() * 6.28) * np.exp(-t / 0.0006)
    y = (nz * 0.9 + click * 0.35) * bright
    if mute:                                                                    # palm hitting the strings: low thud
        th = (np.sin(2 * np.pi * 105 * t) * np.exp(-t / 0.018) * 0.9
              + E.lp(r.standard_normal(k), 400, order=2) * np.exp(-t / 0.006) * 0.8)
        y = y + th
    return y * (vel ** 1.5)


def _pickup(x, kind='bridge'):
    """magnetic pickup: resonance + 2nd-order roll-off, mild low-cut (removes pulse DC)."""
    if kind == 'bridge':
        return E.eq(x, ('hp', 35, 0.7), ('peak', 2900, 1.6, 5.0), ('lp', 6500, 0.8), ('lp', 8000, 0.7))
    return E.eq(x, ('hp', 35, 0.7), ('peak', 2000, 1.2, 3.0), ('lp', 4200, 0.8), ('lp', 5500, 0.7))     # neck-ish


def _lp_close(x, fc_hi, fc_lo, tau, Q=0.75, block=192):
    """time-varying lowpass whose cutoff closes exponentially from fc_hi to fc_lo with time constant tau."""
    t = _t(len(x))
    cut = fc_lo + (fc_hi - fc_lo) * np.exp(-t / tau)
    return E.lp_sweep_blocks(x, cut, Q=Q, block=block)


def _release(n, start_s, rel_s, exp=False):
    """unity envelope that fades out over rel_s starting at start_s (linear, or exponential with tau = rel_s/4)."""
    e = np.ones(n)
    i0 = int(start_s * SR)
    k = max(1, int(rel_s * SR))
    if i0 < n:
        if exp:
            seg = np.exp(-np.arange(k) / (k / 4.0)) * np.linspace(1, 0, k) ** 0.25
        else:
            seg = np.linspace(1, 0, k)
        e[i0:i0 + k] = seg[:max(0, n - i0)]
        e[i0 + k:] = 0.0
    return e


# =====================================================================================================
#  rhythm DI
# =====================================================================================================
def _chord_di(root, dur, vel, mute, r, take_detune=0.0, strum_ms=1.4):
    """DI of one power chord (root + 5th + octave) summed, pre-amp. returns (array, pre-roll samples=0)."""
    vel = float(np.clip(vel, 0.05, 1.0))
    if mute:
        tau_a = 0.018 + 0.010 * vel                                             # palm damping: fast decay
        nlen = min(float(dur), 0.22)
        rel = 0.012
        fc_hi, fc_lo, tau_f = 2400 + 1400 * vel, 300.0, 0.03
    else:
        tau_a = 2.2
        nlen = float(dur)
        rel = 0.09                                                              # hand mute at the end of the chord
        fc_hi, fc_lo, tau_f = 6500.0, 1500.0, 0.55
    n = int((nlen + rel + 0.015) * SR)
    t = _t(n)
    out = np.zeros(n)
    strings = [(root, 1.0), (root + 7, 0.85), (root + 12, 0.68)]
    for i, (m, lvl) in enumerate(strings):
        f0 = E.midi2hz(m) * 2 ** ((take_detune + r.normal(0, 2.0)) / 1200.0)
        off = int((i * strum_ms * (0.7 if mute else 1.0) + r.normal(0, 0.3)) / 1000 * SR)
        off = max(0, off)
        m_n = n - off
        s = _string(np.full(m_n, f0), vel, r, mute=mute, sub=0.18 if mute else 0.10)
        env = np.exp(-t[:m_n] / (tau_a * (1.0 - 0.12 * i)))                     # higher strings decay a bit faster
        s *= env * lvl
        out[off:] += s
    out *= vel ** 1.2
    # pick transient on the chord (one pick, slight per-string spread handled by strum offsets)
    pk = _pick_transient(vel, mute, r)
    out[:len(pk)] += pk[:n] * 0.9
    out = _lp_close(out, fc_hi, fc_lo, tau_f)
    out *= _release(n, nlen, rel, exp=not mute)
    return out


def render_rhythm_di(events, seed=0, total_len=None, jitter_ms=0.0, take_detune=0.0, vel_jitter=0.0):
    """DI stream of all rhythm events (sum of strings, pickup applied). mono."""
    r = np.random.default_rng(1000 + seed)
    ev = sorted([(float(t), int(m), float(d), float(v), bool(mu)) for (t, m, d, v, mu) in events], key=lambda e: e[0])
    if not ev:
        return np.zeros(int((total_len or 1.0) * SR))
    end = max(t + d for (t, m, d, v, mu) in ev) + RHYTHM_TAIL
    n = int((total_len if total_len is not None else end) * SR)
    di = np.zeros(n + int(0.5 * SR))
    for (t, m, d, v, mu) in ev:
        v = float(np.clip(v * (1 + vel_jitter * r.normal()), 0.05, 1.0))
        tt = t + (r.normal(0, jitter_ms) / 1000.0 if jitter_ms > 0 else 0.0)
        i = int(round(max(0.0, tt) * SR))
        if i >= n:
            continue
        x = _chord_di(m, d, v, mu, r, take_detune=take_detune)
        e = min(len(di), i + len(x))
        di[i:e] += x[:e - i]
    di = di[:n]
    di = _pickup(di, 'bridge')
    di *= 0.30                                                                  # DI nominal: vel=1 chug peak ~1
    return di


# =====================================================================================================
#  amp + cab
# =====================================================================================================
def _follow(x, attack, release, hop=48):
    """peak envelope follower (control rate hop samples), returns per-sample envelope."""
    n = len(x)
    npts = (n + hop - 1) // hop
    lvl = np.abs(x)
    if npts * hop > n:
        lvl = np.concatenate([lvl, np.zeros(npts * hop - n)])
    lvl = lvl.reshape(npts, hop).max(1)
    env = E._env_follow(lvl, attack, release, hop)
    return np.interp(np.arange(n), np.arange(npts) * hop + hop // 2, env)


def _stage(x, g, bias, fold=0.0):
    """asymmetric soft clipper (biased tanh, DC compensated) with optional sine wave-folding blend."""
    z = g * x + bias
    y = np.tanh(z) - np.tanh(bias)
    if fold > 0:
        w = np.sin(np.clip(z, -5.5, 5.5) * 0.9)
        y = y * (1 - fold) + w * fold
    return y


def _gate(y, key, thresh_db, knee_db=8.0, attack=0.0006, release=0.025, hop=48):
    env = _follow(key, attack, release, hop)
    lvl = 20 * np.log10(env + 1e-9)
    g = np.clip((lvl - thresh_db) / knee_db + 1.0, 0.0, 1.0)
    g = g * g * (3 - 2 * g)                                                     # smoothstep
    return y * g


def _cab(y, variant=0.0, bass_peak=4.5, lp_hz=5500.0):
    """4x12 cabinet as a filter bank + a handful of sub-ms reflections (comb ripple)."""
    y = E.eq(y,
             ('hp', 72, 0.9),
             ('peak', 110 + 8 * variant, 1.0, bass_peak),
             ('peak', 550 + 35 * variant, 1.3, -5.5),
             ('peak', 1400, 1.2, -1.5),
             ('peak', 2800 + 160 * variant, 1.0, 4.5),
             ('peak', 4300 - 120 * variant, 3.0, 0.5))
    sos = signal.butter(8, lp_hz + 180 * variant, 'low', fs=SR, output='sos')
    y = signal.sosfilt(sos, y)
    z = y.copy()
    for d, g in [(0.00029 + 0.00004 * variant, -0.20), (0.00107, 0.13), (0.00263 + 0.0001 * variant, -0.08), (0.0049, 0.05)]:
        k = int(d * SR)
        z[k:] += g * y[:-k]
    return z


def amp_sim(di, mode='rhythm', variant=0.0, gain_db=0.0, gate_db=None):
    """High-gain amp + cab on a DI stream. mode: 'rhythm' | 'lead'. variant: small take-to-take differences."""
    di = np.asarray(di, dtype=np.float64)
    if mode == 'lead':
        x = E.eq(di, ('hp', 190, 0.8), ('hp', 120, 0.7), ('peak', 1000, 0.8, 6.0), ('hs', 2500, 0.7, 2.0), ('lp', 7500, 0.7))
        pre = E.db(13.0 + gain_db + 0.5 * variant)
        stages = [(5.0, 0.16, 0.0), (9.0, 0.24, 0.10), (6.0, 0.10, 0.0)]
        gate_thr = -46.0 if gate_db is None else gate_db
    else:
        x = E.eq(di, ('hp', 105 + 6 * variant, 0.75), ('hp', 65, 0.7), ('peak', 850, 0.8, 3.5), ('hs', 2600, 0.7, 2.5), ('lp', 7200, 0.7))
        pre = E.db(6.0 + gain_db + 0.7 * variant)
        stages = [(4.0, 0.14, 0.0), (7.0, 0.22, 0.12), (3.5, 0.08, 0.0)]
        gate_thr = -36.0 if gate_db is None else gate_db
    x = x * pre
    xs = signal.resample_poly(x, 2, 1)                                          # 2x oversampling for the clippers
    sr2 = SR * 2
    for i, (g, bias, fold) in enumerate(stages):
        xs = _stage(xs, g, bias * (1 + 0.15 * variant), fold)
        b, a = signal.butter(1, 32, 'high', fs=sr2); xs = signal.lfilter(b, a, xs)       # coupling cap
        b, a = signal.butter(1, 9000 - 800 * i, 'low', fs=sr2); xs = signal.lfilter(b, a, xs)  # Miller cap
        if i == 0:
            b, a = E.biquad('peak', 1300 / 2, 0.8, 2.0); xs = signal.lfilter(b, a, xs)   # f0/2: coefficients for the 2x rate
    # power stage: symmetric tanh with sag (slow envelope lowers gain)
    sag = _follow(xs, 0.006, 0.16, hop=96)
    xs = np.tanh(xs * 2.2 / (1.0 + 0.8 * sag))
    y = signal.resample_poly(xs, 1, 2)[:len(x)]
    # tone stack
    if mode == 'lead':
        y = E.eq(y, ('ls', 150, 0.7, -2.0), ('peak', 500, 1.0, -2.5), ('peak', 950, 1.0, 2.5), ('peak', 3300, 1.0, 2.5), ('hs', 5000, 0.7, -5.0))
    else:
        y = E.eq(y, ('ls', 140, 0.7, 0.0), ('peak', 550, 0.9, -5.0 - 0.8 * variant), ('peak', 1700, 1.2, -1.5), ('peak', 3200, 1.0, 2.5), ('hs', 5000, 0.7, -5.0))
    y = _gate(y, di, gate_thr + 20 * np.log10(E.db(0.0)))
    y = _cab(y, variant=variant, bass_peak=2.0 if mode == 'lead' else 3.5)
    return y


def _norm_peak(y, knee=0.45, ceiling=0.6):
    """soft ceiling: linear below knee, tanh-compressed above, never exceeds ceiling (levels stay content-independent)."""
    a = np.abs(y)
    over = a > knee
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (ceiling - knee) * np.tanh((a[over] - knee) / (ceiling - knee)))
    return y


# =====================================================================================================
#  public: rhythm
# =====================================================================================================
_RHY_GAIN = 0.20          # output trim: raw amp+cab peaks ~2.3-2.7 -> ~0.5 (calibrated in tests/test_metal_guitar_B.py)


def render_rhythm(events, seed=0, total_len=None, jitter_ms=0.0, take_detune=0.0, vel_jitter=0.0, variant=None):
    """mono distorted rhythm guitar. events: (t, root_midi, dur, vel, mute). mute=True -> palm-muted chug."""
    if variant is None:
        variant = float((seed % 3) - 1)
    di = render_rhythm_di(events, seed=seed, total_len=total_len, jitter_ms=jitter_ms, take_detune=take_detune, vel_jitter=vel_jitter)
    y = amp_sim(di, 'rhythm', variant=variant) * _RHY_GAIN
    return _norm_peak(y)


def render_rhythm_pair(events, total_len=None, seeds=(1, 2)):
    """two independent takes (different seeds, +-3..6 ms jitter, pitch/pick/amp differences) hard L / R."""
    a = render_rhythm(events, seed=seeds[0], total_len=total_len, jitter_ms=3.5, take_detune=-2.5, vel_jitter=0.05, variant=-1.0)
    b = render_rhythm(events, seed=seeds[1], total_len=total_len, jitter_ms=3.5, take_detune=+2.5, vel_jitter=0.05, variant=+1.0)
    b = np.concatenate([np.zeros(int(0.0025 * SR)), b])[:len(a)]              # take B a hair late (2.5 ms)
    n = max(len(a), len(b))
    out = np.zeros((n, 2))
    out[:len(a), 0] = a
    out[:len(b), 1] = b
    return out


# =====================================================================================================
#  public: lead
# =====================================================================================================
def _lead_freq(n, midi, dur, bend, vib, slide_from, r):
    t = _t(n)
    f0 = E.midi2hz(midi)
    semis = np.zeros(n)
    if slide_from is not None and int(slide_from) != int(midi):
        gl = float(np.clip(0.03 * abs(midi - slide_from), 0.07, 0.25))
        k = min(n, int(gl * SR))
        u = t[:k] / gl
        semis[:k] += (slide_from - midi) * (1 - (u * u * (3 - 2 * u)))
    if bend:
        t0 = min(0.08, 0.15 * dur)
        tb = float(np.clip(0.10 + 0.07 * abs(bend), 0.12, 0.45))
        u = np.clip((t - t0) / tb, 0, 1)
        semis += bend * (0.5 - 0.5 * np.cos(np.pi * u))
    if vib and vib > 0:
        onset = 0.16 + 0.08 * r.random()
        ramp = np.clip((t - onset) / 0.35, 0, 1)
        rate = 5.5 + r.normal(0, 0.25)
        lfo = np.sin(2 * np.pi * rate * t + r.random() * 6.28)
        semis += vib * ramp * (0.5 * lfo + 0.2)                                 # guitar vibrato is biased upward
    semis += 0.02 * _smooth_noise(n, 2.0, r)                                    # finger pressure drift (+-2 cents)
    return f0 * 2 ** (semis / 12.0)


def _lead_note_di(midi, length, dur, vel, bend, vib, slide_from, r, pinch=0):
    n = int(length * SR)
    t = _t(n)
    f = _lead_freq(n, midi, dur, bend, vib, slide_from, r)
    if pinch:
        f = f * pinch
    s = _string(f, vel, r, mute=False, pick_pos=0.13 + 0.03 * r.random(), sub=0.0 if midi > 55 else 0.08, saw2=0.45, sharpen=0.010)
    env = 0.30 + 0.70 * np.exp(-t / 1.7)                                        # sustain floor = compression/feedback
    s = s * env * vel ** 1.1
    if length > 1.4:                                                            # feedback harmonic creeps in on long notes
        ph, _ = _phase(f * 2.0, r.random())
        fb = np.sin(2 * np.pi * ph) * np.clip((t - 1.3) / 2.0, 0, 1) * 0.10
        s = s + fb
    pk = _pick_transient(vel, False, r, bright=1.2)
    s[:len(pk)] += pk[:n] * 0.8
    s = _lp_close(s, 7000.0, 2200.0, 0.8)
    return s


def render_lead_di(events, total_len=None, seed=5):
    r = np.random.default_rng(2000 + seed)
    ev = sorted([(float(e[0]), int(e[1]), float(e[2]), float(e[3]), float(e[4] or 0.0), float(e[5] or 0.0), e[6]) for e in events], key=lambda e: e[0])
    if not ev:
        return np.zeros(int((total_len or 1.0) * SR))
    end = max(t + d for (t, m, d, v, b, vi, sf) in ev) + LEAD_TAIL
    n = int((total_len if total_len is not None else end) * SR)
    di = np.zeros(n + int(0.5 * SR))
    for k, (t, m, d, v, b, vi, sf) in enumerate(ev):
        nxt = ev[k + 1][0] if k + 1 < len(ev) else None
        rel = 0.03
        if nxt is not None and nxt < t + d:                                     # legato: cut at next note
            length = (nxt - t) + 0.02
            rel = 0.02
        else:
            length = d + rel
        length = max(length, 0.03)
        x = _lead_note_di(m, length, d, v, b, vi, sf, r)
        x *= _release(len(x), length - rel, rel)
        i = int(round(max(0.0, t) * SR))
        if i >= n:
            continue
        e = min(len(di), i + len(x))
        di[i:e] += x[:e - i]
    di = di[:n]
    di = _pickup(di, 'bridge')
    return di * 0.42


_LEAD_GAIN = 0.21


def render_lead(events, total_len=None, seed=5):
    """mono lead guitar. events: (t, midi, dur, vel, bend_semis, vibrato_semis, slide_from_midi|None)."""
    di = render_lead_di(events, total_len=total_len, seed=seed)
    y = amp_sim(di, 'lead', variant=0.0) * _LEAD_GAIN
    return _norm_peak(y)


# =====================================================================================================
#  public: clean
# =====================================================================================================
def render_clean(events, total_len=None, seed=9):
    """stereo clean/ambient guitar (arpeggios): soft pluck, slight compression, chorus + reverb."""
    r = np.random.default_rng(3000 + seed)
    ev = sorted([(float(t), int(m), float(d), float(v)) for (t, m, d, v) in events], key=lambda e: e[0])
    if not ev:
        return np.zeros((int((total_len or 1.0) * SR), 2))
    end = max(t + d for (t, m, d, v) in ev) + CLEAN_TAIL
    n = int((total_len if total_len is not None else end) * SR)
    buf = np.zeros((n + int(0.5 * SR), 2))
    for k, (t, m, d, v) in enumerate(ev):
        v = float(np.clip(v, 0.05, 1.0))
        ring = float(np.clip(d + 1.2, 0.4, 3.5))                                # let ring into the next notes
        nn = int(ring * SR)
        tt = _t(nn)
        f = np.full(nn, E.midi2hz(m) * 2 ** (r.normal(0, 2.0) / 1200.0))
        s = _string(f, v * 0.8, r, mute=False, pick_pos=0.24 + 0.06 * r.random(), sub=0.0, saw2=0.35, sharpen=0.008)
        s *= np.exp(-tt / 1.9) * v ** 1.3
        pk = _pick_transient(v * 0.7, False, r, bright=0.5)
        s[:len(pk)] += pk[:nn] * 0.4
        s = _lp_close(s, 5500.0, 1400.0, 0.5)
        s *= _release(nn, ring - 0.3, 0.3)
        s = _pickup(s, 'neck')
        p = float(np.clip(((m - 60) / 24.0) * 0.6 + r.normal(0, 0.1), -0.6, 0.6))   # spread by pitch
        i = int(round(max(0.0, t) * SR))
        if i >= n:
            continue
        e = min(len(buf), i + nn)
        buf[i:e] += E.pan(s, p)[:e - i]
    x = buf[:n] * 0.5
    x = E.eq(x, ('hp', 70, 0.8), ('peak', 240, 1.0, -1.0), ('peak', 3000, 1.0, 1.5), ('hs', 5500, 0.7, -2.0))
    x = E.compress(x, -22, 3.0, 0.008, 0.15, makeup_db=3.0)
    x = E.saturate(x, 1.3, 0.25)
    x = E.chorus(x, rate=0.55, depth=0.0028, mix=0.4, voices=2, base=0.011)
    x = E.reverb(x, decay=2.2, predelay=0.025, hf_damp=0.5, width=1.2, mix=0.28, lowcut=180, highcut=7000, seed=23)
    pk = float(np.abs(x).max()) + 1e-12
    if pk > 0.45:
        x = x * (0.42 / pk)
    return x


# =====================================================================================================
#  extras (one-shots, cached to ./cache/gtrB_*.wav)
# =====================================================================================================
def _cached(name, key, fn):
    path = os.path.join(CACHE, 'gtrB_%s_%s.wav' % (name, '_'.join(str(k).replace('.', 'p') for k in key)))
    if os.path.exists(path):
        try:
            return E.read(path)
        except Exception:
            pass
    y = fn()
    try:
        E.write(path, y)
    except Exception:
        pass
    return y


def feedback_swell(midi, dur=3.0, seed=0, harmonic=2.0):
    """note that swells in from silence and drifts into its feedback harmonic. mono, through the lead amp."""
    return _cached('fbswell', (midi, round(dur, 3), seed, harmonic), lambda: _feedback_swell(midi, dur, seed, harmonic))


def _feedback_swell(midi, dur, seed, harmonic):
    r = np.random.default_rng(4000 + seed)
    n = int((dur + 0.6) * SR)
    t = _t(n)
    f = _lead_freq(n, midi, dur, 0.0, 0.35, None, r)
    s_fund = _string(f, 0.8, r, sub=0.0, saw2=0.3, sharpen=0.0)
    ph, _ = _phase(f * harmonic, r.random())
    s_harm = np.sin(2 * np.pi * ph) + 0.25 * np.sin(4 * np.pi * ph)
    rise = np.clip(t / (0.45 * dur), 0, 1) ** 2
    xf = np.clip((t - 0.35 * dur) / (0.5 * dur), 0, 1)
    s = (s_fund * (1 - xf) * 0.6 + s_harm * xf) * rise * 0.9
    s = _lp_close(s, 5000, 2500, 1.0)
    s *= _release(n, dur, 0.4)
    di = _pickup(s, 'bridge') * 0.42
    y = amp_sim(di, 'lead', gate_db=-70) * _LEAD_GAIN
    return _norm_peak(y)


def pick_scrape(dur=1.0, seed=0, f_start=2200.0, f_end=170.0):
    """pick scraped down the low string: descending comb-filtered noise with a ratchet, through the rhythm amp."""
    return _cached('scrape', (round(dur, 3), seed, int(f_start), int(f_end)), lambda: _pick_scrape(dur, seed, f_start, f_end))


def _pick_scrape(dur, seed, f_start, f_end):
    r = np.random.default_rng(5000 + seed)
    n = int((dur + 0.8) * SR)
    t = _t(n)
    # fixed-pitch feedback comb on noise, then time-warped so its pitch glides
    base_f = 1000.0
    m = int(n * 2.5)
    nz = r.standard_normal(m)
    nz = E.eq(nz, ('hp', 300, 0.7), ('lp', 7000, 0.7))
    d = int(SR / base_f)
    comb = nz.copy()
    for _ in range(3):
        comb[d:] += 0.72 * comb[:-d]
    curve = np.clip(t / dur, 0, 1) ** 0.8
    f_scr = f_start * (f_end / f_start) ** curve
    pos = np.cumsum(f_scr / base_f)
    pos = np.clip(pos, 0, m - 2)
    y = np.interp(pos, np.arange(m), comb)
    ratchet = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * np.cumsum(70 * (1 - 0.5 * curve)) / SR))
    y = y * ratchet * 0.35 + nz[:n] * 0.08
    env = np.clip(t / 0.03, 0, 1) * _release(n, dur, 0.15)
    y *= env
    di = _pickup(y, 'bridge') * 0.30
    out = amp_sim(di, 'rhythm', gate_db=-60) * _RHY_GAIN
    return _norm_peak(out)


def pinch_harmonic(midi, dur=1.5, seed=0, partial=None):
    """squealing pinch harmonic: fundamental choked, 3rd/4th/5th partial dominates, fast vibrato. mono."""
    return _cached('pinch', (midi, round(dur, 3), seed, partial), lambda: _pinch_harmonic(midi, dur, seed, partial))


def _pinch_harmonic(midi, dur, seed, partial):
    r = np.random.default_rng(6000 + seed)
    if partial is None:
        partial = int(r.choice([3, 4, 5]))
    n = int((dur + 0.5) * SR)
    t = _t(n)
    f = _lead_freq(n, midi, dur, 0.0, 0.5, None, r)
    s_h = _string(f * partial, 1.0, r, sub=0.0, saw2=0.4, sharpen=0.02)
    s_f = _string(f, 1.0, r, sub=0.0, saw2=0.3, sharpen=0.01)
    xf = np.clip(t / 0.03, 0, 1)
    s = s_h * (0.15 + 0.85 * xf) * (0.3 + 0.7 * np.exp(-t / 2.0)) + s_f * (1 - xf) * 0.6 + s_f * 0.06 * np.exp(-t / 0.3)
    pk = _pick_transient(1.0, True, r, bright=1.4)
    s[:len(pk)] += pk[:n] * 1.2
    s = _lp_close(s, 8000, 3500, 1.5)
    s *= _release(n, dur, 0.08)
    di = _pickup(s, 'bridge') * 0.42
    y = amp_sim(di, 'lead', gain_db=4.0) * _LEAD_GAIN
    return _norm_peak(y)


def dive_bomb(midi, dur=2.5, seed=0, depth_semis=26.0):
    """whammy dive: pitch falls depth_semis with increasing wobble, string flutter at the bottom. mono."""
    return _cached('dive', (midi, round(dur, 3), seed, depth_semis), lambda: _dive_bomb(midi, dur, seed, depth_semis))


def _dive_bomb(midi, dur, seed, depth_semis):
    r = np.random.default_rng(7000 + seed)
    n = int((dur + 0.6) * SR)
    t = _t(n)
    u = np.clip(t / dur, 0, 1)
    fall = depth_semis * (u ** 1.6)
    wob = (0.15 + 1.2 * u) * np.sin(2 * np.pi * (4.5 + 2.0 * u) * t)
    f = E.midi2hz(midi) * 2 ** (-(fall + wob) / 12.0)
    s = _string(f, 1.0, r, sub=0.0, saw2=0.5, sharpen=0.01)
    s *= (0.35 + 0.65 * np.exp(-t / 1.4))
    flutter = E.lp(r.standard_normal(n), 300, order=2) * (u ** 3) * 0.5
    s = s * (1 + flutter) + flutter * 0.4
    pk = _pick_transient(1.0, False, r)
    s[:len(pk)] += pk[:n]
    s = _lp_close(s, 6500, 1800, 1.2)
    s *= _release(n, dur + 0.15, 0.3)
    di = _pickup(s, 'bridge') * 0.42
    y = amp_sim(di, 'lead') * _LEAD_GAIN
    return _norm_peak(y)


__all__ = ['render_rhythm', 'render_rhythm_pair', 'render_lead', 'render_clean', 'render_rhythm_di', 'render_lead_di',
           'amp_sim', 'feedback_swell', 'pick_scrape', 'pinch_harmonic', 'dive_bomb']
