"""metal_drums.py - modern metal drum kit for the METAL remix.

Kick/snare = synthesized layers (sub body + beater click / body + wires + crack) stacked with the
FluidR3 GM "Power" kit (program 16 on channel 9) one-shots; toms + cymbals come from the Power kit and are
re-shaped (added sub + pitch drop on toms, brightening on cymbals). Everything is 48 kHz float64.

Public API
    hit(name, vel=1.0, seed=0) -> mono (kick/snare/toms) or stereo (cymbals/hats) one-shot, peak ~0.5 at vel=1
    render_kit(events, total_len=None) -> stereo   events = [(t_sec, name, vel), ...]
    NAMES  list of accepted piece names;  PAN / LEVEL_DB  per-piece placement constants.
"""
import os, sys, subprocess
import numpy as np
from scipy import signal
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(HERE, '..'))
import engine as E

SR = E.SR
CACHE = os.path.join(HERE, 'cache')
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
KIT_PROGRAM = 16          # FluidR3 "Power" kit
GM_VEL = 110              # velocity used for the cached soundfont one-shots

NAMES = ['kick', 'snare', 'snare_ghost', 'tom_hi', 'tom_mid', 'tom_floor', 'hat_closed', 'hat_open',
         'hat_pedal', 'ride', 'ride_bell', 'crash1', 'crash2', 'china', 'splash', 'rim']
ALIASES = {'crash': 'crash1', 'hat': 'hat_closed', 'hh': 'hat_closed', 'hat_o': 'hat_open', 'tom_lo': 'tom_floor',
           'tom_low': 'tom_floor', 'floor': 'tom_floor', 'ghost': 'snare_ghost', 'bell': 'ride_bell', 'sidestick': 'rim'}
GM_NOTE = {'kick': 36, 'snare': 38, 'snare2': 40, 'rim': 37, 'tom_floor': 41, 'tom_mid': 45, 'tom_hi': 48,
           'crash1': 49, 'crash2': 57, 'china': 52, 'ride': 51, 'ride_bell': 53, 'splash': 55,
           'hat_closed': 42, 'hat_pedal': 44, 'hat_open': 46}
# stereo placement (-1 = L, +1 = R) from the drummer's perspective, hats left / ride right
PAN = {'kick': 0.0, 'snare': 0.0, 'snare_ghost': 0.0, 'rim': 0.0, 'tom_hi': -0.35, 'tom_mid': 0.1, 'tom_floor': 0.5,
       'hat_closed': -0.4, 'hat_open': -0.4, 'hat_pedal': -0.4, 'ride': 0.45, 'ride_bell': 0.45,
       'crash1': -0.55, 'crash2': 0.55, 'china': 0.65, 'splash': -0.3}
# mix levels (dB) relative to a full-velocity kick, applied inside render_kit
LEVEL_DB = {'kick': 0.0, 'snare': -0.5, 'snare_ghost': -9.0, 'rim': -8.0, 'tom_hi': -3.0, 'tom_mid': -2.5,
            'tom_floor': -2.0, 'hat_closed': -13.0, 'hat_open': -11.0, 'hat_pedal': -17.0, 'ride': -11.0,
            'ride_bell': -9.0, 'crash1': -8.0, 'crash2': -8.0, 'china': -7.0, 'splash': -10.0}
VEL_LAYERS = np.array([0.15, 0.3, 0.45, 0.6, 0.78, 0.9, 1.0])   # timbre layers (quantized), level is continuous
N_RR = 4                                                        # round-robin variants per layer

_gm_cache = {}
_hit_cache = {}


# ----------------------------------------------------------------------------------------------------------
# soundfont one-shots
# ----------------------------------------------------------------------------------------------------------
def _render_gm_note(note, vel=GM_VEL, program=KIT_PROGRAM, dur=6.0, out_path=None):
    """Tiny MIDI renderer: ONE drum note on channel 9 with a program_change (kit select) -> stereo array."""
    out_path = out_path or os.path.join(CACHE, f'gm_p{program}_n{note}_v{vel}.wav')
    mid = mido.MidiFile(ticks_per_beat=960); tr = mido.MidiTrack(); mid.tracks.append(tr)
    tempo = mido.bpm2tempo(120); tr.append(mido.MetaMessage('set_tempo', tempo=tempo))
    tr.append(mido.Message('program_change', program=program, channel=9, time=0))
    tr.append(mido.Message('control_change', control=91, value=0, channel=9, time=0))   # reverb send off
    tr.append(mido.Message('control_change', control=93, value=0, channel=9, time=0))   # chorus send off
    tr.append(mido.Message('note_on', note=int(note), velocity=int(vel), channel=9, time=0))
    tr.append(mido.Message('note_off', note=int(note), velocity=0, channel=9,
                           time=int(mido.second2tick(0.5, 960, tempo))))
    tr.append(mido.MetaMessage('end_of_track', time=int(mido.second2tick(dur - 0.5, 960, tempo))))
    midp = out_path[:-4] + '.mid'; mid.save(midp)
    subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', '1.0', '-r', str(SR), '-F', out_path,
                    SF2, midp], check=True, capture_output=True)
    return E.read(out_path)


def _gm(name):
    """Cached, onset-trimmed, tail-trimmed, peak-normalised Power-kit sample (stereo)."""
    if name in _gm_cache: return _gm_cache[name]
    os.makedirs(CACHE, exist_ok=True)
    note = GM_NOTE[name]
    path = os.path.join(CACHE, f'gm_p{KIT_PROGRAM}_n{note}_v{GM_VEL}.wav')
    x = E.read(path) if os.path.exists(path) else _render_gm_note(note, out_path=path)
    x = E.stereo(x)
    a = np.abs(x).max(1); pk = a.max() + 1e-12
    i0 = int(np.argmax(a > pk * 2e-3)); i0 = max(0, i0 - 8)
    idx = np.where(a > pk * 10 ** (-66 / 20))[0]; i1 = min(len(x), (idx[-1] if len(idx) else len(x)) + int(0.03 * SR))
    x = x[i0:i1] / pk
    x = E.fade(x, 0.0, 0.02)
    _gm_cache[name] = x
    return x


# ----------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------
def _hold_decay(n, hold, tau):
    """gate-like envelope: 1 for `hold` seconds, then exponential decay with time constant tau."""
    t = E.t_axis(n); return np.where(t < hold, 1.0, np.exp(-(t - hold) / tau))


def _env(x, hold, tau):
    """apply _hold_decay to a mono or stereo array."""
    e = _hold_decay(len(x), hold, tau); return x * e if x.ndim == 1 else x * e[:, None]


def _norm(x):
    return x / (np.abs(x).max() + 1e-12)


def _fit(x, n):
    """pad/trim a mono or stereo array to n samples."""
    if len(x) >= n: return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:]); return np.concatenate([x, pad], 0)


def _resample(x, factor):
    """play `x` at speed `factor` (pitch and length change together) via linear interpolation."""
    n = len(x); m = int(n / factor); pos = np.arange(m) * factor
    if x.ndim == 1: return np.interp(pos, np.arange(n), x)
    return np.stack([np.interp(pos, np.arange(n), x[:, c]) for c in range(x.shape[1])], 1)


def _vel_timbre(x, vel):
    """lower velocity = darker: moving low-pass + high shelf cut (nothing happens at vel >= 1)."""
    v = float(np.clip(vel, 0.02, 1.0))
    if v >= 0.99: return x
    fc = 2200 + 17000 * v ** 1.4
    return E.eq(x, ('lp', min(fc, 19000), 0.6), ('hs', 5000, 0.7, -8.0 * (1 - v)))


def _vel_gain(vel):
    return float(np.clip(vel, 0.0, 1.2)) ** 1.4


# ----------------------------------------------------------------------------------------------------------
# synthesized layers
# ----------------------------------------------------------------------------------------------------------
def _kick_synth(vel, rng):
    """sub body (fast pitch drop to ~58 Hz) + shell knock + beater click (3-6 kHz), gated short."""
    n = int(0.30 * SR); t = E.t_axis(n)
    f_end = 55.0 + 7.0 * vel
    f = f_end * (1 + 2.0 * np.exp(-t / 0.014) + 0.45 * np.exp(-t / 0.05))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.06) * (1 - np.exp(-t / 0.0006))
    knock = np.sin(2 * np.pi * 160 * t + 0.3) * np.exp(-t / 0.012) * 0.2
    body = E.saturate(body + knock, 1.6, 0.5)
    # beater click: 3-6 kHz noise burst + two short pings + a little 2 kHz "thwack"
    click_n = rng.standard_normal(n) * np.exp(-t / 0.0045)
    click_n = _norm(E.eq(click_n, ('bp', 4300, 0.9), ('hp', 2600, 0.7)))
    thwack = _norm(E.eq(rng.standard_normal(n) * np.exp(-t / 0.002), ('bp', 2000, 1.2))) * 0.35
    ping = np.sin(2 * np.pi * 3900 * t) * np.exp(-t / 0.0035) + 0.5 * np.sin(2 * np.pi * 6100 * t) * np.exp(-t / 0.0018)
    click = (click_n * 1.0 + ping * 0.6 + thwack) * (0.4 + 0.6 * vel ** 2) * (1 - np.exp(-t / 0.0003))
    y = body + click * 1.4
    y *= _hold_decay(n, 0.075, 0.032)
    return y


def _snare_synth(vel, rng):
    """shell body (~196 Hz + overtones, slight pitch drop) + snare-wire noise + 2-4 kHz crack + stick."""
    n = int(0.36 * SR); t = E.t_axis(n)
    f0 = 196.0 * (1 + 0.4 * np.exp(-t / 0.012)); ph = 2 * np.pi * np.cumsum(f0) / SR
    body = (np.sin(ph) * np.exp(-t / 0.08) + 0.55 * np.sin(2 * np.pi * 338 * t) * np.exp(-t / 0.05)
            + 0.3 * np.sin(2 * np.pi * 470 * t) * np.exp(-t / 0.03)) * (1 - np.exp(-t / 0.0008))
    wires = rng.standard_normal(n)
    wires = E.eq(wires, ('hp', 900, 0.7), ('hp', 500, 0.7), ('peak', 6500, 0.8, 3.0), ('lp', 12500, 0.7))
    wires = _norm(wires) * np.exp(-t / 0.1) * (1 - np.exp(-t / 0.003)) * (1 + 0.25 * np.sin(2 * np.pi * 196 * t))
    crack = rng.standard_normal(n) * np.exp(-t / 0.008)
    crack = _norm(E.eq(crack, ('bp', 3000, 0.9), ('peak', 2400, 1.5, 4.0))) * (0.35 + 0.65 * vel ** 2)
    stick = np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.0045) * 0.3
    y = body * 1.3 + wires * 1.1 + crack * 0.75 + stick
    return y


# ----------------------------------------------------------------------------------------------------------
# piece builders (each returns a peak-normalised one-shot at the requested velocity layer)
# ----------------------------------------------------------------------------------------------------------
def _build_kick(vel, rng):
    syn = _kick_synth(vel, rng)
    gm = E.mono(_gm('kick')); gm = E.eq(gm, ('lp', 1500, 0.7), ('hp', 45, 0.7))
    gm = _fit(gm, len(syn)) * _hold_decay(len(syn), 0.06, 0.035)
    y = syn + 0.4 * _norm(gm)
    y = E.eq(y, ('hp', 32, 0.7), ('peak', 62, 1.0, 2.5), ('peak', 200, 1.2, -3.0), ('peak', 380, 1.0, -7.0),
             ('peak', 4300, 1.2, 3.0), ('hs', 11000, 0.7, -4.0))
    y = E.saturate(y, 1.2, 0.25)
    y = E.fade(y, 0.0, 0.01)
    return _norm(y)


def _build_snare(vel, rng, ghost=False):
    syn = _snare_synth(vel * (0.6 if ghost else 1.0), rng)
    gm = _fit(E.mono(_gm('snare')), len(syn)) + 0.6 * _fit(E.mono(_gm('snare2')), len(syn))
    gm = E.eq(gm, ('hp', 130, 0.7))
    y = syn + 0.9 * _norm(gm)
    # compression: 6 ms attack lets the crack through, body is levelled; saturation clips the crack peak
    y = E.compress(y, thresh_db=-12, ratio=3.0, attack=0.006, release=0.05, knee=4.0, makeup_db=4.0)
    y = E.saturate(y, 2.2, 0.5)
    y = E.eq(y, ('hp', 90, 0.7), ('peak', 200, 1.5, 3.0), ('peak', 520, 1.0, -3.5), ('peak', 3500, 1.0, 3.0),
             ('hs', 8000, 0.7, 2.0))
    if ghost:
        y = E.eq(y, ('lp', 5500, 0.7), ('peak', 3000, 1.0, -3.0)) * _hold_decay(len(y), 0.02, 0.07)
    else:
        y *= _hold_decay(len(y), 0.07, 0.085)
    y = E.fade(y, 0.0, 0.02)
    return _norm(y)


_TOM = {'tom_floor': (78.0, 0.2, 0.22, 0.3), 'tom_mid': (108.0, 0.16, 0.21, 0.22), 'tom_hi': (148.0, 0.12, 0.17, 0.15)}


def _build_tom(name, vel, rng):
    f_t, tau_sub, tau_gate, click_amt = _TOM[name]
    gm = E.mono(_gm(name)); n = min(len(gm), int(0.9 * SR)); gm = gm[:n]; t = E.t_axis(n)
    f = f_t * (1 + 0.7 * np.exp(-t / 0.045)); ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph) * np.exp(-t / tau_sub) * (1 - np.exp(-t / 0.001))
    click = rng.standard_normal(n) * np.exp(-t / 0.0022)
    click = _norm(E.eq(click, ('bp', 3200, 1.0), ('hp', 1800, 0.7))) * (0.3 + 0.7 * vel ** 2)
    y = _norm(gm) * 1.0 + sub * 0.8 + click * click_amt
    y = E.eq(y, ('hp', 42, 0.7), ('peak', 450, 1.2, -3.0), ('peak', 4000, 1.0, 3.0), ('hs', 7000, 0.7, 2.0))
    y = E.saturate(y, 1.3, 0.35)
    y *= _hold_decay(n, 0.1, tau_gate)
    y = E.fade(y, 0.0, 0.15)
    return _norm(y)


def _build_cymbal(name, vel, rng):
    gm = _gm(name)
    if name in ('hat_closed', 'hat_pedal', 'hat_open'):
        y = E.eq(gm, ('hp', 420, 0.7), ('hs', 8000, 0.7, 4.0), ('peak', 10500, 1.0, 2.0))
        if name == 'hat_closed': y = _env(y[:int(0.45 * SR)], 0.015, 0.075)
        if name == 'hat_pedal': y = _env(y[:int(0.35 * SR)], 0.01, 0.05)
    elif name in ('ride', 'ride_bell'):
        y = E.eq(gm, ('hp', 380, 0.7), ('hs', 7000, 0.7, 3.5), ('peak', 3200, 1.0, 1.5))
    elif name == 'rim':
        y = E.eq(gm[:int(0.4 * SR)], ('hp', 300, 0.7), ('peak', 2000, 1.0, 2.0))
    else:  # crashes / china / splash
        y = E.eq(gm, ('hp', 340, 0.7), ('hs', 7000, 0.7, 4.0), ('peak', 3000, 1.0, 1.5), ('peak', 700, 1.0, -2.0))
    y = E.saturate(y, 1.15, 0.25)
    y = E.fade(y, 0.0, 0.02)
    return _norm(y)


def _build(name, vel, seed):
    rng = np.random.default_rng(1000 + seed)
    if name == 'kick': return _build_kick(vel, rng)
    if name == 'snare': return _build_snare(vel, rng)
    if name == 'snare_ghost': return _build_snare(vel, rng, ghost=True)
    if name in _TOM: return _build_tom(name, vel, rng)
    return _build_cymbal(name, vel, rng)


def _canon(name):
    name = ALIASES.get(name, name)
    if name not in NAMES: raise ValueError(f'unknown drum piece {name!r}; known: {NAMES}')
    return name


def _variant(name, vel, rr):
    """peak-normalised one-shot for a quantized velocity layer + round-robin index (cached in memory)."""
    layer = float(VEL_LAYERS[np.argmin(np.abs(VEL_LAYERS - np.clip(vel, 0, 1)))])
    key = (name, layer, rr)
    if key in _hit_cache: return _hit_cache[key]
    y = _build(name, layer, rr * 17 + int(layer * 100))
    y = _vel_timbre(y, layer)
    r = np.random.default_rng(NAMES.index(name) * 10007 + int(round(layer * 100)) * 11 + rr)
    cents = 25.0 if name in ('kick', 'snare', 'snare_ghost', 'rim') or name in _TOM else 12.0
    y = _resample(y, 2 ** (r.uniform(-cents, cents) / 1200)) * E.db(r.uniform(-0.6, 0.6))
    y = _norm(y)
    _hit_cache[key] = y
    return y


# ----------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------
def hit(name, vel=1.0, seed=0):
    """One-shot. mono for kick/snare/toms, stereo for cymbals/hats. Peak ~0.5 at vel=1 (lower vel = darker + quieter)."""
    name = _canon(name)
    y = _variant(name, vel, int(seed) % N_RR)
    return y * 0.5 * _vel_gain(vel)


def render_kit(events, total_len=None, seed=0):
    """events: list of (t_sec, name, vel). Returns stereo float64, peak ~0.5.
    Panning, velocity layers, round-robin, kick choke (double-bass stays tight), hat-open choke,
    kick+snare punch bus, parallel compression, overhead/room reverb."""
    ev = []
    for e in events:
        t, name, vel = e[0], _canon(e[1]), float(e[2])
        if t >= 0: ev.append((float(t), name, vel))
    ev.sort(key=lambda z: z[0])
    tail = 3.5
    last = max([t for t, _, _ in ev], default=0.0)
    n = int(round((total_len if total_len is not None else last + tail) * SR))
    n = max(n, 1)
    bus = {k: np.zeros((n, 2)) for k in ('kick', 'snare', 'toms', 'hats', 'cyms')}
    rng = np.random.default_rng(seed)

    # choke lookups: next kick after a kick; next closed/pedal hat after an open hat
    kick_t = np.array([t for t, nm, _ in ev if nm == 'kick'])
    hat_close_t = np.array([t for t, nm, _ in ev if nm in ('hat_closed', 'hat_pedal')])

    def place(buf, y, i0, pan_):
        if y.ndim == 1: y = E.pan(y, pan_)
        else:
            a = (np.clip(pan_, -1, 1) + 1) / 2 * np.pi / 2
            y = y * np.array([np.cos(a), np.sin(a)]) * 1.2
        e = min(n, i0 + len(y))
        if e > i0: buf[i0:e] += y[:e - i0]

    counters = {}
    for t, name, vel in ev:
        i0 = int(round(t * SR))
        if i0 >= n: continue
        prev = counters.get(name, -1)
        rr = int(rng.integers(0, N_RR - 1)); rr = rr + 1 if rr >= prev else rr   # random, never the same twice in a row
        counters[name] = rr
        y = _variant(name, vel, rr) * (_vel_gain(vel) * E.db(LEVEL_DB[name]))
        if name == 'kick':
            nxt = kick_t[kick_t > t + 1e-6]
            if len(nxt):
                allow = int((nxt[0] - t) * SR) + int(0.003 * SR)
                if allow < len(y): y = E.fade(y[:allow], 0.0, 0.008)
        elif name == 'hat_open':
            nxt = hat_close_t[hat_close_t > t + 1e-6]
            if len(nxt):
                allow = int((nxt[0] - t) * SR)
                if allow < len(y): y = E.fade(y[:allow], 0.0, 0.01)
        pan_ = PAN[name]
        if name in ('kick',): b = bus['kick']
        elif name in ('snare', 'snare_ghost', 'rim'): b = bus['snare']
        elif name in _TOM: b = bus['toms']
        elif name.startswith('hat'): b = bus['hats']
        else: b = bus['cyms']
        place(b, y, i0, pan_)

    # ---- kick + snare punch bus: slow-attack compression keeps transients, trims sustain, adds grit
    ks = bus['kick'] + bus['snare']
    ks_c = E.compress(ks, thresh_db=-14, ratio=4.0, attack=0.014, release=0.09, knee=4.0, makeup_db=2.5)
    ks_c = E.saturate(ks_c, 1.5, 0.3)
    shells = ks_c + bus['toms']
    # ---- parallel ("New York") compression of the whole close-mic kit
    close = shells + bus['hats'] + bus['cyms']
    smash = E.compress(close, thresh_db=-32, ratio=8.0, attack=0.001, release=0.06, knee=2.0, makeup_db=12.0)
    smash = E.eq(smash, ('lp', 7500, 0.7), ('hp', 60, 0.7))
    # ---- overhead / room reverb (kick barely sent)
    send = bus['snare'] * 1.0 + bus['toms'] * 0.8 + bus['cyms'] * 0.45 + bus['hats'] * 0.35 + bus['kick'] * 0.18
    room = E.reverb(send, decay=0.85, predelay=0.012, hf_damp=0.5, width=1.0, mix=1.0, lowcut=260, highcut=7500,
                    early=0.7, seed=3)
    out = close + smash * E.db(-9.0) + room * E.db(-13.0)
    out = E.eq(out, ('hp', 28, 0.7), ('hs', 12000, 0.7, 1.5))
    out = out * 0.42
    out = E.limiter(out, ceiling_db=-5.0, lookahead=0.003, release=0.08)
    if total_len is not None: out = _fit(out, int(round(total_len * SR)))
    return out


def demo_events(bars=2, start=0.0, bpm=110.0):
    """A test pattern: crash on 1, snare on 2+4, hats 8ths, double-bass 16ths, 32nd tom fill at the end."""
    beat = 60.0 / bpm; ev = []
    for b in range(bars):
        t0 = start + b * 4 * beat
        ev.append((t0, 'crash1' if b % 2 == 0 else 'china', 0.9))
        for k in range(16):
            ev.append((t0 + k * beat / 4, 'kick', 0.95 if k % 4 == 0 else 0.8))
        for k in range(8):
            ev.append((t0 + k * beat / 2, 'hat_closed', 0.8 if k % 2 == 0 else 0.55))
        ev.append((t0 + beat, 'snare', 1.0)); ev.append((t0 + 3 * beat, 'snare', 0.95))
        ev.append((t0 + 2.5 * beat, 'snare_ghost', 0.5))
        if b == bars - 1:
            fill = ['snare', 'snare', 'tom_hi', 'tom_hi', 'tom_mid', 'tom_mid', 'tom_floor', 'tom_floor']
            for k, nm in enumerate(fill):
                ev.append((t0 + 3.5 * beat + k * beat / 8, nm, 0.85))
    ev.append((start + bars * 4 * beat, 'crash2', 1.0)); ev.append((start + bars * 4 * beat, 'kick', 1.0))
    return ev
