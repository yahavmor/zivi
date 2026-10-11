"""Minimal DAW engine: synths, drums, effects, mixing. 48 kHz float stereo."""
import numpy as np, subprocess, os, json
from scipy import signal
from scipy.ndimage import maximum_filter1d
import soundfile as sf

SR = 48000
rng = np.random.default_rng(7)

# ---------- basic helpers ----------
def midi2hz(n): return 440.0 * 2 ** ((n - 69) / 12)
def db(x): return 10 ** (x / 20)
def stereo(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1: return np.stack([x, x], 1)
    return x
def mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(1)
def pan(x, p):  # p in [-1,1], constant power
    x = mono(x); a = (p + 1) / 2 * np.pi / 2
    return np.stack([x * np.cos(a), x * np.sin(a)], 1)
def gain(x, g_db): return np.asarray(x) * db(g_db)
def t_axis(n): return np.arange(n) / SR
def fade(x, fin=0.0, fout=0.0):
    x = np.array(x, dtype=np.float64, copy=True); n = len(x)
    if fin > 0:
        k = min(n, int(fin * SR)); w = np.linspace(0, 1, k)
        x[:k] = (x[:k].T * w).T
    if fout > 0:
        k = min(n, int(fout * SR)); w = np.linspace(1, 0, k)
        x[n - k:] = (x[n - k:].T * w).T
    return x
def env_exp(n, tau):  # exponential decay with time constant tau seconds
    return np.exp(-t_axis(n) / tau)
def adsr(n, a, d, s, r, gate):
    """a,d,r seconds; s level; gate = seconds note is held. total length n samples."""
    t = t_axis(n); e = np.zeros(n)
    ia, idd, ig = int(a * SR), int(d * SR), int(gate * SR)
    e[:ia] = np.linspace(0, 1, ia, endpoint=False) if ia > 0 else 1
    if idd > 0:
        seg = np.linspace(1, s, idd, endpoint=False); e[ia:ia + idd] = seg[:max(0, n - ia)]
    e[ia + idd:ig] = s
    if ig < n:
        ir = max(1, int(r * SR)); start = e[ig - 1] if ig > 0 else s
        tail = start * np.exp(-np.arange(n - ig) / (ir / 5.0))
        e[ig:] = tail
    return e

def read(path):
    x, sr = sf.read(path, dtype='float64')
    assert sr == SR, (path, sr)
    return x
def write(path, x, subtype='FLOAT'):
    sf.write(path, np.asarray(x, dtype=np.float32), SR, subtype=subtype)

# ---------- filters ----------
def biquad(kind, f0, Q=0.707, gain_db=0.0):
    """RBJ cookbook biquads. returns (b,a)"""
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f0 / SR; cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * Q)
    if kind == 'lp': b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'hp': b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'bp': b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'notch': b = [1, -2 * cw, 1]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'peak': b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == 'ls':
        b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    elif kind == 'hs':
        b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    else: raise ValueError(kind)
    b = np.array(b) / a[0]; a = np.array(a) / a[0]
    return b, a
def eq(x, *bands):
    """bands: tuples (kind, f0, Q, gain_db). works on mono or stereo"""
    y = np.array(x, dtype=np.float64, copy=True)
    for band in bands:
        kind, f0 = band[0], band[1]; Q = band[2] if len(band) > 2 else 0.707; g = band[3] if len(band) > 3 else 0.0
        b, a = biquad(kind, f0, Q, g)
        y = signal.lfilter(b, a, y, axis=0)
    return y
def lp(x, f, Q=0.707, order=1):
    y = x
    for _ in range(order): y = eq(y, ('lp', f, Q))
    return y
def hp(x, f, Q=0.707, order=1):
    y = x
    for _ in range(order): y = eq(y, ('hp', f, Q))
    return y
def svf_lp_sweep(x, cutoff_arr, res=0.3):
    """time-varying state-variable lowpass (mono). cutoff_arr in Hz per sample."""
    x = mono(x); n = len(x); y = np.zeros(n)
    g = np.tan(np.pi * np.clip(cutoff_arr, 20, SR * 0.45) / SR); k = 2 - 2 * res
    ic1 = ic2 = 0.0
    a1 = 1 / (1 + g * (g + k)); a2 = g * a1; a3 = g * a2
    for i in range(n):
        v3 = x[i] - ic2; v1 = a1[i] * ic1 + a2[i] * v3; v2 = ic2 + a2[i] * ic1 + a3[i] * v3
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2; y[i] = v2
    return y
def lp_sweep_blocks(x, cutoff_arr, Q=0.9, block=256):
    """time-varying lowpass via block-wise biquads (fast). mono/stereo."""
    x = np.asarray(x, dtype=np.float64); n = len(x); y = np.zeros_like(x)
    zi = None
    for s in range(0, n, block):
        e = min(n, s + block); fc = float(np.clip(cutoff_arr[min(n - 1, (s + e) // 2)], 30, 20000))
        b, a = biquad('lp', fc, Q)
        if zi is None: zi = np.zeros((2,) + x.shape[1:])
        y[s:e], zi = signal.lfilter(b, a, x[s:e], axis=0, zi=zi)
    return y

# ---------- dynamics ----------
def _env_follow(level, attack, release, hop):
    """peak follower on a control-rate signal (hop samples per point)."""
    a_att = np.exp(-hop / (attack * SR)); a_rel = np.exp(-hop / (release * SR))
    out = np.zeros_like(level); e = 0.0
    for i, v in enumerate(level):
        e = a_att * e + (1 - a_att) * v if v > e else a_rel * e + (1 - a_rel) * v
        out[i] = e
    return out
def compress(x, thresh_db=-18, ratio=3.0, attack=0.01, release=0.12, knee=6.0, makeup_db=0.0, sidechain=None, hop=48):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    sc = mono(sidechain if sidechain is not None else x)
    npts = n // hop + 1
    lvl = np.abs(sc[:npts * hop].copy()); lvl.resize(npts * hop); lvl = lvl.reshape(npts, hop).max(1)
    envl = _env_follow(lvl, attack, release, hop)
    ldb = 20 * np.log10(envl + 1e-9)
    over = ldb - thresh_db
    # soft knee
    gr = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, over * (1 - 1 / ratio), (over + knee / 2) ** 2 / (2 * knee) * (1 - 1 / ratio)))
    g = db(-gr + makeup_db)
    gfull = np.interp(np.arange(n), np.arange(npts) * hop, g)
    return (x.T * gfull).T
def limiter(x, ceiling_db=-1.0, lookahead=0.004, release=0.1, oversample=4):
    """true-peak look-ahead limiter: gain never exceeds what the (oversampled) peak allows."""
    from scipy.signal import resample_poly
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    x = np.asarray(x, dtype=np.float64); c = db(ceiling_db); n = len(x)
    xs = resample_poly(x, oversample, 1, axis=0)
    pk = np.abs(xs).max(1) if xs.ndim == 2 else np.abs(xs)
    pk = pk[:n * oversample].reshape(n, oversample).max(1)
    need = np.minimum(1.0, c / (pk + 1e-12))
    la = max(2, int(lookahead * SR))
    g1 = minimum_filter1d(need, size=2 * la + 1)
    # release: instant down, exponential up (control rate 4 samples)
    hop = 4; m = g1[:n // hop * hop].reshape(-1, hop).min(1); gm = np.empty_like(m); cur = 1.0
    r = 1 - np.exp(-hop / (release * SR))
    for i in range(len(m)):
        v = m[i]; cur = v if v < cur else cur + (v - cur) * r; gm[i] = cur
    g2 = np.repeat(gm, hop); g2 = np.concatenate([g2, np.full(n - len(g2), g2[-1])])
    g2 = np.minimum(g2, g1)
    g3 = uniform_filter1d(g2, size=la, mode='nearest')
    g3 = np.minimum(g3, g1)  # safety
    y = (x.T * g3).T
    return np.clip(y, -c * 1.02, c * 1.02)
def saturate(x, drive=1.0, mix=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.tanh(x * drive) / np.tanh(drive) if drive > 0 else x
    return x * (1 - mix) + y * mix
def softclip(x, drive=1.5): return np.tanh(x * drive) / drive

# ---------- effects ----------
_ir_cache = {}
def make_ir(decay=1.8, predelay=0.02, hf_damp=0.4, width=1.0, early=0.5, seed=1):
    key = (decay, predelay, hf_damp, width, early, seed)
    if key in _ir_cache: return _ir_cache[key]
    r = np.random.default_rng(seed); n = int((predelay + decay * 1.2) * SR)
    t = t_axis(n)
    # band-split noise: lows decay long, highs shorter
    bands = [(0, 300, 1.0), (300, 1200, 0.95), (1200, 3500, 0.8 - 0.3 * hf_damp), (3500, 8000, 0.6 - 0.4 * hf_damp), (8000, 20000, 0.4 - 0.3 * hf_damp)]
    ir = np.zeros((n, 2))
    for lo, hi, dmul in bands:
        nz = r.standard_normal((n, 2))
        sos = signal.butter(2, [max(lo, 20), min(hi, SR * 0.49)], btype='band', fs=SR, output='sos')
        nz = signal.sosfilt(sos, nz, axis=0)
        e = np.exp(-t * 6.9 / (decay * dmul))
        ir += (nz.T * e).T
    # early reflections
    for d, g in [(0.007, 0.6), (0.013, 0.5), (0.021, 0.4), (0.029, 0.35), (0.037, 0.3), (0.051, 0.25)]:
        i = int(d * SR)
        if i < n: ir[i, 0] += g * early * (1 if d != 0.013 else -1); ir[i + 3, 1] += g * early * (1 if d != 0.021 else -1)
    ir[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
    # stereo width
    mid = ir.mean(1); side = (ir[:, 0] - ir[:, 1]) / 2 * width
    ir = np.stack([mid + side, mid - side], 1)
    pre = np.zeros((int(predelay * SR), 2)); ir = np.vstack([pre, ir])
    ir /= np.sqrt((ir ** 2).sum()) / 1.0
    _ir_cache[key] = ir
    return ir
def reverb(x, decay=1.8, predelay=0.02, hf_damp=0.4, width=1.0, mix=0.3, lowcut=150, highcut=9000, early=0.5, seed=1):
    x = stereo(x); ir = make_ir(decay, predelay, hf_damp, width, early, seed)
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    wet = hp(wet, lowcut); wet = lp(wet, highcut)
    # normalize wet level roughly to input level
    return x * (1 - mix) + wet * mix
def delay(x, time=0.25, feedback=0.4, mix=0.3, pingpong=False, lowpass=5000, highpass=200, spread=0.0):
    x = stereo(x); n = len(x); d = int(time * SR)
    if d <= 0: return x
    out = np.zeros((n + d * 12, 2)); buf = np.zeros_like(out); buf[:n] = x
    b_lp, a_lp = biquad('lp', lowpass); b_hp, a_hp = biquad('hp', highpass)
    cur = buf.copy()
    wet = np.zeros_like(out)
    for k in range(1, 12):
        cur = signal.lfilter(b_lp, a_lp, cur, axis=0); cur = signal.lfilter(b_hp, a_hp, cur, axis=0)
        cur = cur * feedback if k > 1 else cur
        if pingpong:
            cur = cur[:, ::-1]
        shifted = np.zeros_like(cur); shifted[d:] = cur[:-d]
        if k == 1 and pingpong:
            shifted[:, 1] *= 0.0  # first tap hard left, then alternate
            shifted[:, 0] = shifted[:, 0] * 1.0
        wet += shifted; cur = shifted
        if np.abs(cur).max() < 1e-5: break
    y = out; y[:n] = x
    return y[:n] * 1.0 + wet[:n] * mix
def chorus(x, rate=0.6, depth=0.004, mix=0.4, voices=2, base=0.012):
    x = stereo(x); n = len(x); t = t_axis(n); out = np.zeros_like(x)
    for v in range(voices):
        ph = v * 2 * np.pi / voices
        for c in range(2):
            d = base + depth * (1 + np.sin(2 * np.pi * rate * t + ph + c * np.pi / 2)) / 2
            idx = np.arange(n) - d * SR
            out[:, c] += np.interp(idx, np.arange(n), x[:, c], left=0)
    return x * (1 - mix) + out / voices * mix
def tremolo_pan(x, rate=5.5, depth=0.5):
    x = stereo(x); t = t_axis(len(x)); m = np.sin(2 * np.pi * rate * t)
    l = 1 - depth / 2 + depth / 2 * m; r = 1 - depth / 2 - depth / 2 * m
    return np.stack([x[:, 0] * l, x[:, 1] * r], 1)
def widen(x, amount=0.3, delay_ms=8):
    x = stereo(x); d = int(delay_ms / 1000 * SR); y = x.copy()
    y[d:, 1] += amount * x[:-d, 0]; y[d:, 0] += amount * x[:-d, 1] * -0.5
    return y
def duck(x, trigger_times, depth=0.5, attack=0.005, hold=0.03, release=0.25, length=None):
    """sidechain-style ducking envelope applied at trigger times (seconds)."""
    x = np.asarray(x, dtype=np.float64); n = len(x); env = np.ones(n)
    for tt in trigger_times:
        i = int(tt * SR)
        if i >= n: continue
        k = int((attack + hold + release * 3) * SR); seg = np.ones(k); tt_ = t_axis(k)
        seg = np.where(tt_ < attack, 1 - depth * tt_ / attack, np.where(tt_ < attack + hold, 1 - depth, 1 - depth * np.exp(-(tt_ - attack - hold) / (release / 3))))
        e = min(n, i + k); env[i:e] = np.minimum(env[i:e], seg[:e - i])
    return (x.T * env).T
def tape_stop(x, start, length=0.6, curve=2.0):
    """slow down to zero speed from 'start' seconds over 'length' seconds."""
    x = stereo(x); n = len(x); i0 = int(start * SR); m = int(length * SR)
    if i0 >= n: return x
    tt = np.linspace(0, 1, m); speed = (1 - tt) ** curve
    pos = i0 + np.cumsum(speed)
    pos = pos[pos < n - 1]
    seg = np.stack([np.interp(pos, np.arange(n), x[:, c]) for c in range(2)], 1)
    seg = (seg.T * np.linspace(1, 0.0, len(seg)) ** 0.5).T
    return np.vstack([x[:i0], seg])
def resample_speed(x, factor):
    """play at speed factor (pitch and time change together, like a turntable)."""
    x = stereo(x); n = len(x); m = int(n / factor)
    pos = np.linspace(0, n - 1, m)
    return np.stack([np.interp(pos, np.arange(n), x[:, c]) for c in range(2)], 1)
def reverse(x): return np.asarray(x)[::-1].copy()
def bitcrush(x, bits=8, down=4):
    x = np.asarray(x, dtype=np.float64); q = 2 ** (bits - 1)
    y = np.round(x * q) / q
    if down > 1:
        y = np.repeat(y[::down], down, axis=0)[:len(x)]
    return y
def telephone(x, drive=3.0):
    y = eq(mono(x), ('hp', 350, 0.8), ('hp', 350, 0.8), ('lp', 3200, 0.8), ('lp', 3200, 0.8), ('peak', 1800, 1.2, 4))
    y = saturate(y, drive)
    return y
def vinyl_noise(dur, level_db=-38, crackle=25, seed=3):
    r = np.random.default_rng(seed); n = int(dur * SR)
    hiss = r.standard_normal((n, 2)) * 0.5
    hiss = eq(hiss, ('lp', 6000), ('hp', 300), ('peak', 1500, 0.5, 3))
    # crackle: sparse impulses
    cr = np.zeros((n, 2)); k = int(crackle * dur)
    idx = r.integers(0, n - 50, k); amp = r.uniform(0.2, 1.0, k) * r.choice([-1, 1], k)
    for i, a in zip(idx, amp):
        w = r.integers(3, 25); cr[i:i + w, r.integers(0, 2)] += a * np.hanning(w) * 3
    cr = eq(cr, ('lp', 7000), ('hp', 200))
    # slow rumble
    rum = eq(r.standard_normal((n, 2)), ('lp', 120), ('lp', 120)) * 2
    y = hiss * 0.5 + cr + rum * 0.3
    y *= db(level_db) / (np.abs(y).max() + 1e-9) * 3
    return y

# ---------- oscillators ----------
def sine(freq, n, phase=0.0):
    f = np.broadcast_to(freq, n) if np.isscalar(freq) else freq
    ph = 2 * np.pi * np.cumsum(f) / SR + phase
    return np.sin(ph)
def saw_bl(freq, n, nharm=None):
    """band-limited saw via additive synthesis (static freq)."""
    t = t_axis(n); nh = int(min(nharm or 200, (SR * 0.45) / freq)); y = np.zeros(n)
    for k in range(1, nh + 1): y += np.sin(2 * np.pi * freq * k * t) / k
    return y * 2 / np.pi
def square_bl(freq, n):
    t = t_axis(n); nh = int((SR * 0.45) / freq); y = np.zeros(n)
    for k in range(1, nh + 1, 2): y += np.sin(2 * np.pi * freq * k * t) / k
    return y * 4 / np.pi
def tri(freq, n):
    t = t_axis(n); return 2 / np.pi * np.arcsin(np.sin(2 * np.pi * freq * t))
def noise(n, seed=None):
    r = rng if seed is None else np.random.default_rng(seed); return r.standard_normal(n)

# ---------- instruments ----------
def rhodes(note, dur, vel=0.8, release=0.5, bright=1.0):
    """FM electric piano (DX-style): carrier + 1:1 modulator (body) + 14:1 tine."""
    f = midi2hz(note); n = int((dur + release) * SR); t = t_axis(n)
    env = adsr(n, 0.004, 1.2, 0.35, release, dur)
    kb = np.clip((note - 40) / 50, 0, 1)
    idx_body = (1.6 + 1.2 * vel) * (1.3 - kb) * bright
    mod_env = np.exp(-t / (0.35 + 0.4 * (1 - kb)))
    body = np.sin(2 * np.pi * f * t + idx_body * mod_env * np.sin(2 * np.pi * f * t))
    tine_env = np.exp(-t / 0.06) * vel ** 2 * 1.2 * bright
    tine = np.sin(2 * np.pi * f * t + tine_env * np.sin(2 * np.pi * f * 14 * t)) * np.exp(-t / 0.5)
    bark = np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.12) * vel ** 3 * 0.25
    y = (body * 0.85 + tine * 0.25 + bark) * env * vel
    y = softclip(y, 1.2)
    return y
def rhodes_chord(notes, dur, vel=0.8, strum=0.012, release=0.6, bright=1.0):
    n = int((dur + release) * SR) + int(strum * len(notes) * SR); out = np.zeros(n)
    for i, nt in enumerate(notes):
        y = rhodes(nt, dur, vel * (0.9 + 0.1 * rng.random()), release, bright)
        s = int(i * strum * SR); out[s:s + len(y)] += y
    return out
def pad(notes, dur, vel=0.5, attack=0.9, release=1.5, cutoff=2200, detune=7):
    n = int((dur + release) * SR); out = np.zeros((n, 2))
    for i, nt in enumerate(notes):
        f = midi2hz(nt)
        for j, cents in enumerate([-detune, 0, detune]):
            ff = f * 2 ** (cents / 1200)
            y = saw_bl(ff, n, nharm=60) * 0.5 + tri(ff * 2.0, n) * 0.08
            p = [-0.7, 0.0, 0.7][j] * (1 if i % 2 == 0 else -1)
            out += pan(y, p)
    env = adsr(n, attack, 0.5, 0.85, release, dur)
    cut = cutoff * (0.35 + 0.65 * env) 
    out = lp_sweep_blocks(out, cut, Q=0.8)
    out = (out.T * env).T * vel / max(1, len(notes)) * 1.2
    out = chorus(out, 0.3, 0.003, 0.35)
    return out
def bass(note, dur, vel=0.9, glide_from=None, glide=0.08, release=0.08, sub=0.55, growl=0.55, cutoff=1100):
    f1 = midi2hz(note); n = int((dur + release) * SR); t = t_axis(n)
    if glide_from is not None:
        f0 = midi2hz(glide_from); k = int(glide * SR); fr = np.full(n, f1); fr[:k] = np.geomspace(f0, f1, k)
    else: fr = np.full(n, f1)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    s = np.sin(ph) * sub + np.sin(2 * ph) * 0.12
    # upper: saw through lowpass with envelope
    sw = ((ph / np.pi) % 2 - 1); sw = lp(sw, cutoff, 0.9, 2)
    env = adsr(n, 0.004, 0.25, 0.7, release, dur)
    envf = adsr(n, 0.002, 0.18, 0.3, release, dur)
    y = (s * env + sw * envf * growl) * vel
    y = saturate(y, 1.6, 0.6)
    y = eq(y, ('hp', 28, 0.7), ('peak', 700, 1.0, 2.0))
    return y
def bass808(note, dur, vel=1.0, bend=1.9, bend_time=0.05, drive=2.5):
    f = midi2hz(note); n = int(dur * SR); t = t_axis(n)
    fr = f * (1 + (bend - 1) * np.exp(-t / bend_time))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    env = np.exp(-t / (dur / 2.2)); env[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    y = np.sin(ph) * env
    y = saturate(y, drive, 0.8) * vel
    y = eq(y, ('hp', 25))
    return y
def pluck_lead(note, dur, vel=0.7, release=0.3, vib=0.0):
    f = midi2hz(note); n = int((dur + release) * SR); t = t_axis(n)
    fr = f * (1 + vib * 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    y = np.sin(ph) * 0.8 + np.sin(2 * ph) * 0.25 * np.exp(-t / 0.4) + np.sin(3 * ph) * 0.1 * np.exp(-t / 0.2)
    env = adsr(n, 0.01, 0.3, 0.6, release, dur)
    return y * env * vel
def bell(note, dur=2.0, vel=0.5):
    f = midi2hz(note); n = int(dur * SR); t = t_axis(n)
    y = np.sin(2 * np.pi * f * t + 2.0 * np.exp(-t / 0.4) * np.sin(2 * np.pi * f * 3.5 * t)) * np.exp(-t / (dur / 4))
    return y * vel
def whistle(dur=0.5, f=2900.0, trill=42.0, vel=0.6):
    n = int(dur * SR); t = t_axis(n)
    fr = f * (1 + 0.03 * np.sin(2 * np.pi * trill * t)); ph = 2 * np.pi * np.cumsum(fr) / SR
    am = 1 + 0.6 * np.sin(2 * np.pi * trill * t + 1)
    y = np.sin(ph) * am + 0.3 * np.sin(2 * ph) * am
    env = adsr(n, 0.01, 0.05, 0.9, 0.05, dur - 0.05)
    y = y * env + noise(n, 11) * 0.03 * env
    y = eq(y, ('bp', f, 2.0), ('hs', 4000, 0.7, 6))
    return y / (np.abs(y).max() + 1e-9) * vel

# ---------- drums ----------
def kick(vel=1.0, tone=52, punch=1.0, dur=0.5, seed=0):
    n = int(dur * SR); t = t_axis(n)
    fr = tone * (1 + 3.0 * punch * np.exp(-t / 0.035)) ; ph = 2 * np.pi * np.cumsum(fr) / SR
    y = np.sin(ph) * np.exp(-t / 0.16)
    click = noise(n, seed + 1) * np.exp(-t / 0.004) * 0.6; click = hp(click, 1500)
    y = y + click * punch
    y = saturate(y, 1.8, 0.7); y = eq(y, ('hp', 30), ('peak', 3500, 1.0, 2))
    return y * vel * 0.95
def snare(vel=1.0, dur=0.35, seed=0):
    n = int(dur * SR); t = t_axis(n)
    body = (np.sin(2 * np.pi * 185 * t) * 0.7 + np.sin(2 * np.pi * 330 * t) * 0.3) * np.exp(-t / 0.07)
    nz = noise(n, seed + 2); nz = eq(nz, ('hp', 700), ('lp', 9000), ('peak', 2500, 1, 3)) * np.exp(-t / 0.12)
    y = body * 0.8 + nz * 0.9
    y = saturate(y, 1.5, 0.5); y = eq(y, ('peak', 200, 1.2, 3))
    return y / (np.abs(y).max() + 1e-9) * vel * 0.9
def clap(vel=1.0, dur=0.45, seed=0):
    n = int(dur * SR); t = t_axis(n); y = np.zeros(n)
    for i, d in enumerate([0, 0.011, 0.024, 0.034]):
        k = int(d * SR); e = np.exp(-(t - d) / (0.012 if i < 3 else 0.14)) * (t >= d)
        y += noise(n, seed + 10 + i) * e * (0.8 if i < 3 else 1.0)
    y = eq(y, ('bp', 1400, 0.6), ('peak', 900, 1.0, 4), ('hp', 500), ('hs', 6000, 0.7, -4))
    return y / (np.abs(y).max() + 1e-9) * vel * 0.8
def snap(vel=0.8, seed=0):
    n = int(0.12 * SR); t = t_axis(n)
    y = noise(n, seed + 20) * np.exp(-t / 0.015) + np.sin(2 * np.pi * 1100 * t) * np.exp(-t / 0.01) * 0.6
    y = eq(y, ('bp', 2200, 0.8), ('hp', 800))
    return y / (np.abs(y).max() + 1e-9) * vel
def hat(vel=0.6, open_=False, dur=None, seed=0):
    dur = dur or (0.45 if open_ else 0.07); n = int(dur * SR); t = t_axis(n)
    y = np.zeros(n)
    for r in [1.0, 1.342, 1.2312, 1.6532, 1.9523, 2.1]:
        y += np.sign(np.sin(2 * np.pi * 330 * r * t + r))
    y = y / 6 + noise(n, seed + 30) * 0.6
    y = eq(y, ('hp', 6500, 0.8), ('hp', 6500, 0.8), ('peak', 10000, 1.0, 3), ('lp', 16000))
    env = np.exp(-t / (0.18 if open_ else 0.022))
    return y * env / (np.abs(y * env).max() + 1e-9) * vel * 0.7
def rim(vel=0.7, seed=0):
    n = int(0.08 * SR); t = t_axis(n)
    y = np.sin(2 * np.pi * 1700 * t) * np.exp(-t / 0.006) + np.sin(2 * np.pi * 480 * t) * np.exp(-t / 0.02) * 0.5 + noise(n, seed + 40) * np.exp(-t / 0.003)
    y = eq(y, ('hp', 400))
    return y / (np.abs(y).max() + 1e-9) * vel * 0.8
def shaker(vel=0.4, dur=0.09, seed=0):
    n = int(dur * SR); t = t_axis(n)
    y = noise(n, seed + 50) * adsr(n, 0.012, 0.03, 0.3, 0.03, 0.05)
    y = eq(y, ('bp', 6500, 1.0), ('hp', 3500))
    return y / (np.abs(y).max() + 1e-9) * vel * 0.6
def crash(vel=0.6, dur=2.0, seed=0):
    n = int(dur * SR); t = t_axis(n)
    y = noise(n, seed + 60) * np.exp(-t / (dur / 3.5))
    y = eq(y, ('hp', 3000), ('peak', 7000, 0.7, 3), ('lp', 15000))
    return y / (np.abs(y).max() + 1e-9) * vel * 0.5
def reverse_cymbal(dur=1.5, vel=0.6):
    return reverse(crash(vel, dur)) * np.linspace(0.2, 1, int(dur * SR)) ** 2
def riser(dur=2.0, vel=0.4, f0=200, f1=4000):
    n = int(dur * SR); t = t_axis(n)
    nz = noise(n, 70)
    cut = np.geomspace(f0, f1, n)
    y = lp_sweep_blocks(hp(nz, 150), cut, Q=1.2)
    env = np.linspace(0, 1, n) ** 2
    return y * env / (np.abs(y * env).max() + 1e-9) * vel

# ---------- placement / mixing ----------
class Track:
    def __init__(self, name, length_s, stereo_=True):
        self.name = name; self.n = int(length_s * SR)
        self.buf = np.zeros((self.n, 2) if stereo_ else self.n)
    def put(self, x, t, gain_db=0.0, pan_=0.0):
        x = np.asarray(x, dtype=np.float64)
        if self.buf.ndim == 2:
            x = pan(x, pan_) if x.ndim == 1 else x
        i = int(round(t * SR))
        if i < 0: x = x[-i:]; i = 0
        if i >= self.n: return
        e = min(self.n, i + len(x)); self.buf[i:e] += x[:e - i] * db(gain_db)
    def get(self): return self.buf

def ffmpeg_filter(in_path, out_path, af):
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', in_path, '-af', af, '-ar', str(SR), out_path], check=True)
def measure(path):
    out = subprocess.run(['ffmpeg', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
    res = {}
    for line in out.splitlines()[-14:]:
        l = line.strip()
        if l.startswith('I:'): res['lufs'] = float(l.split()[1])
        if l.startswith('LRA:'): res['lra'] = float(l.split()[1])
        if l.startswith('Peak:'): res['tp'] = float(l.split()[1])
    return res

def scratch(x, segs, seed=0):
    """DJ scratch: segs = list of (start_frac, end_frac, duration_s); plays x between fractions (backwards if start>end) in duration_s."""
    x = mono(x); n = len(x); out = []
    for a, b, d in segs:
        m = int(d * SR); pos = np.linspace(a * (n - 1), b * (n - 1), m)
        seg = np.interp(pos, np.arange(n), x); out.append(fade(seg, 0.002, 0.002))
    return np.concatenate(out)
