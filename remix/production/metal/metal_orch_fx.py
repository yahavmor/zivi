"""metal_orch_fx.py - orchestral layer + cinematic FX one-shots for the METAL remix.

Orchestra = FluidSynth (FluidR3_GM) through midi_render.render, one GM program per call, then a gentle
per-family EQ and a big hall reverb.  Everything else is synthesized here in numpy (48 kHz float64).

Public API (all functions are pure; expensive FluidSynth renders are cached in ./cache/ofx_*.wav)
    orch(events, program, total_len=None, reverb_mix=0.35) -> stereo   events = [(t_sec, midi, dur_sec, vel0-1)]
    drone(root_midi, dur, seed=0)            -> stereo  dark detuned low drone, slowly moving lowpass + chorus
    riser(dur, f0=200, f1=6000, seed=0)      -> stereo  filtered-noise sweep + rising tone + shimmer, ends hard at dur
    impact(vel=1.0, seed=0)                  -> stereo  sub boom (90->35 Hz) + noise burst + metallic ring + tail
    sub_drop(dur, f0=80, f1=25)              -> mono    sine glide with fade (+ a little 2nd harmonic)
    downlifter(dur, seed=0)                  -> stereo  falling noise sweep + falling tone, decays over dur
    metal_hit(vel, pitch_hz=420, seed=0)     -> mono    inharmonic anvil/pipe (ratios 1, 2.76, 5.4, 8.93, 13.3 ...)
    reverse_cymbal(dur, seed=0)              -> stereo  reversed Power-kit crash + synthetic shimmer swell
    noise_bed(dur, seed=0)                   -> stereo  dark atmospheric noise texture with slow movement
    braam(root_midi, dur, seed=0)            -> stereo  brass-like cinematic blast (detuned saws + GM trombone/tuba)
    industrial_clank(seed=0)                 -> mono    cluster of 2-3 random metallic strikes in a small room
    timpani_roll(midi=36, dur=2.0, vel=0.9)  -> stereo  GM timpani (47) roll with crescendo + accent (cached)
    orchestra_hit(midi=48, vel=1.0)          -> stereo  GM orchestra hit (55) one-shot (cached)
    gm_oneshot(program, midi, dur, vel, tail)-> stereo  generic cached FluidSynth one-shot (raw, no FX)
    hall(x, mix=0.35, decay=3.0)             -> stereo  the shared big-hall reverb
Levels: one-shots peak at ~0.5 (times vel where a vel argument exists); orch() is normalised to peak 0.5.
"""
import os, sys, hashlib
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.normpath(os.path.join(HERE, '..'))
for _p in (PARENT, HERE):
    if _p not in sys.path: sys.path.insert(0, _p)
import engine as E
import midi_render as M

SR = E.SR
CACHE = os.path.join(HERE, 'cache')
BPM = 110.0; BEAT = 60.0 / BPM; BAR = 4 * BEAT
TWO_PI = 2 * np.pi
_mem = {}          # in-process cache of decoded one-shots


# ----------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------
def _n(dur): return max(1, int(round(float(dur) * SR)))
def _t(n): return np.arange(n) / SR
def _rng(seed): return np.random.default_rng(int(seed) & 0xFFFFFFFF)
def _norm(x, peak=0.5):
    x = np.asarray(x, dtype=np.float64); pk = np.abs(x).max()
    return x * (peak / pk) if pk > 1e-9 else x
def _fade(x, fin=0.0, fout=0.0): return E.fade(x, fin, fout)
def _fade_cos(x, fin=0.0, fout=0.0):
    """raised-cosine (sin^2) fades: much less energy in the last ms than a linear ramp, no slope step."""
    x = np.array(x, dtype=np.float64, copy=True); n = len(x)
    if fin > 0:
        k = min(n, int(fin * SR)); w = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
        x[:k] = (x[:k].T * w).T
    if fout > 0:
        k = min(n, int(fout * SR)); w = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        x[n - k:] = (x[n - k:].T * w).T
    return x
def _onset(n, tau=0.0004):
    """short exponential attack ramp (default 0.4 ms) so noise layers do not start with a step from silence."""
    return 1 - np.exp(-_t(n) / tau)
def _level(x, win=0.3, max_db=8.0, target=None):
    """slow automatic gain control: divides by the (win-second) RMS envelope so that sub-Hz level pulsing
    (beating between detuned voices) disappears while faster timbral movement is kept. gain limited to +-max_db."""
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, dtype=np.float64); n = len(x); w = max(8, min(n, int(win * SR)))
    m = x if x.ndim == 1 else x.mean(1)
    env = np.sqrt(uniform_filter1d(m ** 2, w, mode='nearest') + 1e-18)
    tgt = float(np.median(env)) if target is None else float(target)
    g = np.clip(tgt / env, E.db(-max_db), E.db(max_db))
    g = uniform_filter1d(g, max(8, w // 2), mode='nearest')
    return (x.T * g).T
def _phase(freq_arr): return TWO_PI * np.cumsum(freq_arr) / SR
def _lfo(n, rate, phase=0.0):
    """unipolar 0..1 sine LFO"""
    return 0.5 + 0.5 * np.sin(TWO_PI * rate * _t(n) + phase)
def _sweep(x, kind, cutoff_arr, Q=0.9, block=256):
    """block-wise time-varying biquad (lp/hp/bp) on mono or stereo (generalised E.lp_sweep_blocks)."""
    x = np.asarray(x, dtype=np.float64); n = len(x); y = np.zeros_like(x); zi = None
    for s in range(0, n, block):
        e = min(n, s + block); fc = float(np.clip(cutoff_arr[min(n - 1, (s + e) // 2)], 20, 20000))
        b, a = E.biquad(kind, fc, Q)
        if zi is None: zi = np.zeros((2,) + x.shape[1:])
        y[s:e], zi = signal.lfilter(b, a, x[s:e], axis=0, zi=zi)
    return y
def _additive(freq_arr, nharm, amps=None, phase0=0.0, rolloff=1.0):
    """saw-like additive oscillator with a time-varying fundamental (harmonics k=1..nharm, amp 1/k^rolloff)."""
    ph = _phase(freq_arr) + phase0; y = np.zeros(len(freq_arr))
    fmax = float(np.max(freq_arr))
    for k in range(1, nharm + 1):
        if fmax * k > SR * 0.45: break
        a = (amps[k - 1] if amps is not None else 1.0 / k ** rolloff)
        y += a * np.sin(k * ph)
    return y * 2 / np.pi
def _inharmonic(n, base_hz, partials, r, decay_scale=1.0, detune=0.0035, attack=0.001):
    """sum of inharmonic partials (ratio, amp, tau) with slight detuned pairs for beating. mono."""
    t = _t(n); y = np.zeros(n)
    for ratio, amp, tau in partials:
        f = base_hz * ratio
        if f > SR * 0.45: continue
        tau = tau * decay_scale
        y += amp * np.sin(TWO_PI * f * t + r.uniform(0, TWO_PI)) * np.exp(-t / tau)
        f2 = f * (1 + r.normal(0, detune))
        y += amp * 0.6 * np.sin(TWO_PI * f2 * t + r.uniform(0, TWO_PI)) * np.exp(-t / (tau * 0.8))
    return y * (1 - np.exp(-t / attack))


# ----------------------------------------------------------------------------------------------------------
# reverb
# ----------------------------------------------------------------------------------------------------------
def hall(x, mix=0.35, decay=3.0, predelay=0.035, hf_damp=0.55, width=1.25, lowcut=120, highcut=8500, seed=5):
    """big concert hall (2.5-3.5 s) used for the orchestra and the cinematic hits. returns stereo."""
    return E.reverb(E.stereo(x), decay=decay, predelay=predelay, hf_damp=hf_damp, width=width, mix=mix,
                    lowcut=lowcut, highcut=highcut, early=0.35, seed=seed)


# ----------------------------------------------------------------------------------------------------------
# FluidSynth wrappers (cached)
# ----------------------------------------------------------------------------------------------------------
def _cached_render(events, program, render_len, tag, gain=0.7):
    """midi_render.render with a content-hashed cache file under ./cache (raw FluidSynth output, stereo)."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(repr((sorted(events), int(program), round(float(render_len), 4), gain)).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, 'ofx_%s_p%d_%s.wav' % (tag, int(program), key))
    if path in _mem: return _mem[path]
    x = E.read(path) if os.path.exists(path) else M.render(events, int(program), path, bpm=BPM, gain=gain, total_len=float(render_len))
    n = _n(render_len); y = np.zeros((n, 2)); m = min(n, len(x)); y[:m] = E.stereo(x)[:m]; x = y   # exact length
    if len(x) < 20 * SR: _mem[path] = x       # keep short one-shots in memory
    return x


def gm_oneshot(program, midi, dur=1.0, vel=0.9, tail=3.0):
    """generic cached FluidSynth one-shot: one note of `dur` seconds + `tail` seconds of release. raw, stereo."""
    return _cached_render([(0.0, int(midi), float(dur), float(vel))], program, dur + tail, 'one')


def _gm_drum(note, program=16, vel=110, dur=6.0):
    """one Power-kit (channel 9) drum note, cached; onset-trimmed stereo."""
    import mido, subprocess
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'ofx_drum_p%d_n%d_v%d.wav' % (program, note, vel))
    if path in _mem: return _mem[path]
    if not os.path.exists(path):
        mid = mido.MidiFile(ticks_per_beat=960); tr = mido.MidiTrack(); mid.tracks.append(tr)
        tempo = mido.bpm2tempo(120); tr.append(mido.MetaMessage('set_tempo', tempo=tempo))
        tr.append(mido.Message('program_change', program=program, channel=9, time=0))
        tr.append(mido.Message('control_change', control=91, value=0, channel=9, time=0))
        tr.append(mido.Message('control_change', control=93, value=0, channel=9, time=0))
        tr.append(mido.Message('note_on', note=int(note), velocity=int(vel), channel=9, time=0))
        tr.append(mido.Message('note_off', note=int(note), velocity=0, channel=9, time=int(mido.second2tick(0.5, 960, tempo))))
        tr.append(mido.MetaMessage('end_of_track', time=int(mido.second2tick(dur - 0.5, 960, tempo))))
        midp = path[:-4] + '.mid'; mid.save(midp)
        subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', '1.0', '-r', str(SR), '-F', path,
                        M.SF2, midp], check=True, capture_output=True)
    x = E.stereo(E.read(path)); a = np.abs(x).max(1); pk = a.max() + 1e-12
    i0 = max(0, int(np.argmax(a > pk * 2e-3)) - 8); x = x[i0:]
    _mem[path] = x
    return x


_EQ_FAMILY = {
    'strings': (('hp', 70, 0.7), ('peak', 350, 1.0, -1.5), ('peak', 3200, 1.2, -1.5), ('hs', 9000, 0.7, 1.5)),
    'choir':   (('hp', 90, 0.7), ('peak', 600, 1.0, -1.0), ('peak', 2800, 1.2, -1.0), ('hs', 7000, 0.7, 1.0)),
    'brass':   (('hp', 50, 0.7), ('peak', 1800, 1.0, 1.0), ('lp', 12000, 0.7)),
    'hit':     (('hp', 40, 0.7), ('peak', 120, 1.0, 1.5)),
    'perc':    (('hp', 35, 0.7), ('peak', 90, 1.0, 2.0), ('lp', 10000, 0.7)),
    'default': (('hp', 60, 0.7),),
}
def _family(program):
    p = int(program)
    if p in (47, 116, 117, 118, 14): return 'perc'      # timpani / taiko / tubular bells: before the strings range
    if p == 55: return 'hit'
    if 40 <= p <= 51: return 'strings'
    if 52 <= p <= 54: return 'choir'
    if 56 <= p <= 63: return 'brass'
    return 'default'


def orch(events, program, total_len=None, reverb_mix=0.35, normalize=True, peak=0.5):
    """FluidSynth (one GM program) -> gentle family EQ -> big hall.  events = [(t, midi, dur, vel0-1)].
    Returns stereo; length = total_len if given, else last note end + 4 s of hall tail."""
    events = [(float(t), int(nn), float(d), float(v)) for (t, nn, d, v) in events]
    if not events: return np.zeros((_n(total_len if total_len is not None else 1.0), 2))
    last_end = max(t + d for (t, nn, d, v) in events)
    render_len = float(total_len) if total_len is not None else last_end + 4.0
    x = _cached_render(events, program, render_len, 'orch')
    x = E.eq(x, *_EQ_FAMILY[_family(program)])
    y = hall(x, mix=reverb_mix) if reverb_mix > 0 else x
    y = _fade_cos(y, 0, min(0.03, 0.25 * render_len))   # total_len may cut a ringing note: no end step
    if normalize: y = _norm(y, peak)
    return y


def orchestra_hit(midi=48, vel=1.0, reverb_mix=0.3):
    """GM program 55 one-shot (cached) + hall tail, peak ~0.5*vel. stereo, ~4 s."""
    x = gm_oneshot(55, midi, 0.6, 0.95, tail=3.4)
    x = E.eq(x, *_EQ_FAMILY['hit'])
    y = hall(x, mix=reverb_mix, decay=2.6)
    return _norm(E.fade(y, 0, 0.1), 0.5 * float(vel))


def timpani_roll(midi=36, dur=2.0, vel=0.9, rate=13.0, accent=True, reverb_mix=0.3):
    """GM timpani (47) roll: alternating strokes at `rate` Hz with a crescendo, accent stroke at `dur` (cached)."""
    evs = []; k = 0; t = 0.0
    while t < dur - 0.02:
        prog = t / max(dur, 1e-6)
        v = (0.3 + 0.7 * prog ** 1.3) * vel * (0.92 + 0.08 * (k % 2))
        evs.append((round(t, 4), int(midi) + (0 if k % 2 == 0 else 0), 0.09, float(np.clip(v, 0.05, 1.0))))
        t += 1.0 / rate * (1.0 + 0.02 * ((k * 7919) % 5 - 2) / 5.0); k += 1
    if accent: evs.append((round(dur, 4), int(midi), 0.8, float(min(1.0, vel))))
    x = _cached_render(evs, 47, dur + 3.0, 'timp')
    x = E.eq(x, *_EQ_FAMILY['perc'])
    y = hall(x, mix=reverb_mix, decay=2.6)
    return _norm(E.fade(y, 0, 0.15), 0.5 * float(vel))


# ----------------------------------------------------------------------------------------------------------
# synthesized one-shots
# ----------------------------------------------------------------------------------------------------------
def impact(vel=1.0, seed=0, dur=3.5):
    """cinematic hit: sub boom with pitch drop 90->35 Hz + noise burst + inharmonic metallic ring + hall tail.
    stereo, peak 0.55*vel (sub kept dry/centre)."""
    vel = float(np.clip(vel, 0.05, 1.5)); n = _n(dur); t = _t(n); r = _rng(100 + seed)
    # sub boom
    f = 38.0 + 52.0 * np.exp(-t / 0.11)
    sub = np.sin(_phase(f)) * np.exp(-t / 0.55) * (1 - np.exp(-t / 0.004))
    sub = E.saturate(sub, 1.4, 0.45)
    # low body thump + click (0.4 ms onset ramps: no step from silence at sample 0)
    on = _onset(n)
    body = E.lp(r.standard_normal(n), 240, 0.9, 2) * np.exp(-t / 0.22) * on
    click = E.hp(r.standard_normal(n), 2500) * np.exp(-t / 0.0025) * on
    # bright noise burst (stereo, decorrelated) - the 'crack' of the hit; scales ~linearly with vel
    burst = r.standard_normal((n, 2))
    burst = E.eq(burst, ('hp', 350, 0.7), ('lp', 7500, 0.7), ('peak', 1500, 0.8, 3.0)) * (np.exp(-t / 0.085) * on)[:, None]
    # metallic ring
    ring = _inharmonic(n, 140.0, [(1, 1.0, 0.5), (2.76, 0.6, 0.32), (5.40, 0.4, 0.2), (8.93, 0.25, 0.12), (11.34, 0.15, 0.08)], r)
    ring = E.hp(ring, 90)
    src = E.stereo(body * 1.6 + click * 0.9 * vel + ring * 0.16) + burst * (0.5 + 1.8 * vel)
    src = _norm(src, 0.5)
    tail = E.reverb(src, decay=2.2, predelay=0.012, hf_damp=0.6, width=1.3, mix=1.0, lowcut=110, highcut=7000, early=0.4, seed=seed + 3)
    y = E.stereo(sub) * 1.0 + src * 1.8 + tail * 0.75
    y = E.eq(y, ('hp', 22, 0.7))
    y = E.fade(y, 0, 0.25)
    return _norm(y, 0.55 * min(vel, 1.0))


def sub_drop(dur, f0=80.0, f1=25.0):
    """sine glide f0->f1 (most of the drop early), soft saturation for a little 2nd harmonic, fade. mono, peak 0.5."""
    n = _n(dur); t = _t(n)
    s = (1 - np.exp(-3.0 * t / dur)) / (1 - np.exp(-3.0))
    f = f0 * (f1 / f0) ** s
    y = np.sin(_phase(f))
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / (dur * 0.7))
    k = int(0.3 * n); env[n - k:] *= np.linspace(1, 0, k) ** 1.5
    y = E.saturate(y * env, 1.8, 0.3)
    y = E.hp(y, 18)
    y = _fade_cos(y, 0, min(0.02, 0.2 * dur))      # the 18 Hz highpass rings past the envelope's end
    return _norm(y, 0.5)


def riser(dur, f0=200.0, f1=6000.0, seed=0):
    """filtered-noise sweep f0->f1 + rising saw tone with accelerating tremolo + cymbal-like shimmer.
    stereo, builds to full level at `dur` and stops hard (15 ms fade). peak 0.5."""
    n = _n(dur); t = _t(n); r = _rng(200 + seed); prog = t / dur
    # noise sweep (decorrelated L/R), bandpass-ish: lowpass tracking the sweep, highpass a bit under it
    nz = r.standard_normal((n, 2))
    cut = np.geomspace(f0, f1, n)
    sw = _sweep(nz, 'lp', cut, Q=1.5)
    sw = _sweep(sw, 'hp', np.maximum(40, cut * 0.22), Q=0.8)
    sw = _sweep(sw, 'lp', cut * 2.5, Q=0.7)
    sw = sw / (np.sqrt((sw ** 2).mean()) + 1e-9) * 0.25
    # rising tone: saw 3 octaves up over the riser, with tremolo accelerating 4 -> 32 Hz
    ft0 = max(45.0, f0 * 0.4); ft1 = ft0 * 8
    ft = ft0 * (ft1 / ft0) ** (prog ** 1.25)
    tone = _additive(ft, 10)
    rate = 4.0 + 28.0 * prog ** 2
    gate = 0.55 + 0.45 * np.sin(_phase(rate))
    tone = tone * gate
    tone = E.stereo(E.lp(tone, 4000)) * 0.35
    # shimmer: very bright noise, late
    sh = E.hp(r.standard_normal((n, 2)), 5500, 0.7, 2) * (prog ** 4)[:, None] * 0.12
    env = (prog ** 2.2)[:, None]
    y = sw * env + tone * (prog ** 1.6)[:, None] + sh
    y = E.widen(y, 0.25, 9)
    y = E.hp(y, 40)
    y = E.fade(y, 0.02, 0.015)
    return _norm(y, 0.5)


def downlifter(dur, seed=0, f1=7000.0, f0=120.0):
    """falling noise sweep f1->f0 + falling tone, full at t=0 then decaying over `dur` (+1 s tail). stereo, peak 0.5."""
    n = _n(dur); t = _t(n); r = _rng(250 + seed); prog = t / dur
    nz = r.standard_normal((n, 2))
    cut = np.geomspace(f1, f0, n)
    sw = _sweep(nz, 'lp', cut, Q=1.4); sw = _sweep(sw, 'hp', np.maximum(40, cut * 0.2), Q=0.8)
    sw = sw / (np.sqrt((sw ** 2).mean()) + 1e-9) * 0.25
    ft = 600.0 * (55.0 / 600.0) ** (prog ** 0.8)
    tone = _additive(ft, 8); tone = E.stereo(E.lp(tone, 3000)) * 0.3
    rate = 24.0 - 20.0 * prog
    tone = tone * (0.6 + 0.4 * np.sin(_phase(rate)))[:, None]
    env = (np.exp(-t / (dur * 0.45)) * (1 - prog ** 3))[:, None]
    y = (sw + tone * 0.8) * env
    y = np.vstack([y, np.zeros((_n(1.0), 2))])
    y = y + E.reverb(y, decay=1.6, predelay=0.01, hf_damp=0.5, width=1.3, mix=1.0, lowcut=120, highcut=7000, seed=seed + 9) * 0.35
    y = E.hp(y, 40)
    y = E.fade(y, 0.003, 0.3)
    return _norm(y, 0.5)


_ANVIL = [(1.0, 1.0, 1.0), (2.76, 1.0, 0.55), (5.40, 0.8, 0.34), (8.93, 0.55, 0.22), (13.34, 0.35, 0.14), (18.64, 0.2, 0.08)]


def metal_hit(vel=1.0, pitch_hz=420.0, seed=0, dur=2.2):
    """inharmonic anvil/pipe strike: partials at 1, 2.76, 5.40, 8.93, 13.3, 18.6 x pitch_hz with detuned beating
    pairs, decays shortening with partial number and pitch, plus a noise click.  mono, peak 0.5*vel."""
    vel = float(np.clip(vel, 0.05, 1.5)); n = _n(dur); t = _t(n); r = _rng(300 + seed)
    scale = (420.0 / float(pitch_hz)) ** 0.5
    ring = _inharmonic(n, float(pitch_hz), _ANVIL, r, decay_scale=scale, attack=0.0008)
    on = _onset(n, 0.0003)
    click = r.standard_normal(n)
    click = E.eq(click, ('bp', 3200, 0.7), ('hp', 900)) * np.exp(-t / 0.006) * (0.4 + 0.9 * vel ** 2) * on
    thud = E.lp(r.standard_normal(n), 320, 0.8, 2) * np.exp(-t / 0.018) * 0.8 * on
    y = ring * (0.6 + 0.4 * vel) + click + thud
    y = E.saturate(y, 1.3 + 0.8 * vel, 0.4)
    y = E.eq(y, ('hp', 120, 0.7), ('hp', 120, 0.7))
    y = E.fade(y, 0, 0.12)
    return _norm(y, 0.5 * min(vel, 1.0))


def industrial_clank(seed=0, dur=1.8):
    """2-3 randomly pitched metallic strikes (pipe/anvil partials + heavy noise) within ~0.25 s in a small room.
    mono, peak 0.5."""
    r = _rng(500 + seed); n = _n(dur); t = _t(n); y = np.zeros(n)
    nhit = int(r.integers(2, 4)); t0 = 0.0
    for i in range(nhit):
        p = float(r.uniform(260, 900)); k = _n(t0); m = n - k
        if m < 480: break                      # strike would start past the end of a very short render
        rr = _rng(600 + seed * 10 + i)
        part = [(1.0, 1.0, 0.5), (2.76, 0.8, 0.3), (5.40, 0.7, 0.2), (8.93, 0.45, 0.12), (13.34, 0.3, 0.08), (18.64, 0.2, 0.05)]
        hit = _inharmonic(m, p, part, rr, decay_scale=float(rr.uniform(0.6, 1.3)), detune=0.006, attack=0.0006)
        nz = rr.standard_normal(m); tt = _t(m); on = _onset(m, 0.0003)
        nz = E.eq(nz, ('bp', float(rr.uniform(1500, 4000)), 0.6), ('hp', 500)) * np.exp(-tt / float(rr.uniform(0.008, 0.02))) * on
        thud = E.lp(rr.standard_normal(m), 260, 0.8, 2) * np.exp(-tt / 0.02) * on
        g = float(rr.uniform(0.5, 1.0)) * (1.0 if i == nhit - 1 else 0.7)
        if k > 0:                                                    # contact choke: the previous ring is damped to ~-10 dB
            k0 = max(0, k - 480)                                     # over the 10 ms before the new strike lands
            y[k0:] *= 0.3 + 0.7 * np.exp(-_t(n - k0) / 0.004)
        y[k:] += (hit * 0.8 + nz * 1.3 + thud * 0.8) * g
        t0 += float(rr.uniform(0.05, 0.14))
    y = E.saturate(y, 2.2, 0.5)
    y = E.eq(y, ('hp', 110, 0.7), ('peak', 2200, 1.0, 2.0))
    room = E.reverb(y, decay=0.9, predelay=0.006, hf_damp=0.5, width=1.0, mix=1.0, lowcut=150, highcut=6500, early=0.7, seed=seed + 21)
    y = y + E.mono(room) * 0.45
    y = E.fade(y, 0, 0.15)
    return _norm(y, 0.5)


def reverse_cymbal(dur, seed=0):
    """reversed FluidR3 Power-kit crash + a synthetic high shimmer swell, ending at `dur`. stereo, peak 0.5."""
    n = _n(dur); t = _t(n); prog = t / dur; r = _rng(700 + seed)
    try:
        cr = _gm_drum(49 if seed % 2 == 0 else 57)
    except Exception:
        cr = E.stereo(E.crash(0.6, 3.0))
    seg = np.zeros((n, 2)); m = min(n, len(cr)); seg[:m] = cr[:m]
    seg = seg[::-1].copy()
    seg = E.eq(seg, ('hp', 300, 0.7), ('hs', 6000, 0.7, 3.0))
    seg = seg / (np.abs(seg).max() + 1e-9)
    env = (prog ** 2.2)[:, None]
    sh = E.hp(r.standard_normal((n, 2)), 4500, 0.7, 2) * 0.25
    y = seg * env + sh * (prog ** 3.5)[:, None]
    y = E.widen(y, 0.2, 7)
    y = E.fade(y, 0.02, 0.012)
    return _norm(y, 0.5)


def drone(root_midi, dur, seed=0, movement=1.0):
    """dark detuned low drone: 4 detuned saws + 5th + sub sines through a slowly moving 2-stage lowpass
    (~160-700 Hz), subtle chorus, slow amplitude movement, a breath of filtered noise. stereo, peak 0.45."""
    n = _n(dur); t = _t(n); f = E.midi2hz(root_midi); r = _rng(400 + seed)
    out = np.zeros((n, 2))
    for cents, p, g in [(-11, -0.65, 1.0), (8, 0.65, 1.0), (-4, -0.25, 0.75), (13, 0.25, 0.75)]:
        ff = f * 2 ** (cents / 1200)
        nh = int(min(48, SR * 0.45 / ff))
        out += E.pan(E.saw_bl(ff, n, nharm=nh) * g, p)
    out += E.pan(E.saw_bl(f * 1.5 * 2 ** (5 / 1200), n, nharm=32) * 0.35, 0.0)
    # the fundamental is carried by stable centre sines; the detuned saws are thinned below ~1.3 f so their
    # four nearly-equal fundamentals cannot beat against each other (that beating made the level throb ~10 dB)
    out = E.eq(out, ('hp', f * 1.3, 0.7))
    ph1, ph2, ph3 = r.uniform(0, TWO_PI, 3)
    lfo = _lfo(n, 0.06, ph1); lfo2 = _lfo(n, 0.023, ph2)
    cut = 160.0 * 2 ** ((1.4 * lfo + 0.7 * lfo2) * movement)
    out = E.lp_sweep_blocks(out, cut, Q=1.15)
    out = E.lp_sweep_blocks(out, cut * 1.6, Q=0.7)
    out = E.chorus(out, rate=0.18, depth=0.0025, mix=0.3, voices=2, base=0.014)
    sub = np.sin(TWO_PI * f * t + 0.3) * 0.8 + np.sin(TWO_PI * f / 2 * t) * 0.3
    am = 1 - 0.14 * _lfo(n, 0.11, ph3) * movement
    air = E.eq(r.standard_normal((n, 2)), ('lp', 1400, 0.7), ('lp', 1400, 0.7), ('hp', 180, 0.7)) * 0.04
    y = out * 0.8 + E.stereo(sub) * 0.45 + air
    # residual beating of the upper harmonics (0.3-3 Hz) is levelled per channel, like a slow compressor on
    # the drone bus, so neither the mono sum nor the stereo image pulses; the timbral chorus movement remains
    y = np.stack([_level(y[:, c], win=0.12, max_db=9.0) for c in range(2)], 1)
    y = y * am[:, None]
    y = E.hp(y, 24)
    y = _fade_cos(y, min(1.5, dur * 0.3), min(2.0, dur * 0.35))
    return _norm(y, 0.45)


def noise_bed(dur, seed=0):
    """dark atmospheric texture: pink-ish noise through a slowly drifting lowpass (~250-900 Hz) + low rumble,
    two slow amplitude LFOs, decorrelated L/R. stereo, peak 0.35."""
    n = _n(dur); t = _t(n); r = _rng(800 + seed)
    nz = r.standard_normal((n, 2))
    dark = E.eq(nz, ('ls', 300, 0.7, 6.0), ('lp', 2500, 0.7), ('peak', 180, 0.8, 3.0))
    ph = r.uniform(0, TWO_PI, 4)
    cut = 250.0 * 2 ** (1.3 * _lfo(n, 0.045, ph[0]) + 0.6 * _lfo(n, 0.017, ph[1]))
    dark = E.lp_sweep_blocks(dark, cut, Q=0.9); dark = E.lp_sweep_blocks(dark, cut * 1.8, Q=0.7)
    rum = E.lp(r.standard_normal(n), 110, 0.8, 3); rum = E.eq(rum, ('peak', 55, 2.0, 4.0))
    rum = rum / (np.abs(rum).max() + 1e-9)
    am = (0.7 + 0.3 * _lfo(n, 0.08, ph[2])) * (0.65 + 0.35 * _lfo(n, 0.031, ph[3]))
    dark = dark / (np.abs(dark).max() + 1e-9)
    y = (dark * 0.8 + E.stereo(rum) * 0.5) * am[:, None]
    y = E.hp(y, 25)
    y = _fade_cos(y, min(2.0, dur * 0.3), min(2.0, dur * 0.3))
    return _norm(y, 0.35)


def braam(root_midi, dur, seed=0, gm_layer=True):
    """brass-like cinematic blast: detuned saws (root, 5th, octave, sub) with slow attack, opening lowpass,
    vibrato onset and saturation, layered with GM trombone+tuba, into the hall. stereo, length dur+3 s, peak 0.5."""
    tail = 3.0; n = _n(dur + tail); t = _t(n); f = E.midi2hz(root_midi); r = _rng(900 + seed)
    attack = float(min(0.7, dur * 0.35))
    env = E.adsr(n, attack, 0.35, 0.85, 1.3, dur)
    fenv = E.adsr(n, attack * 1.4, 0.6, 0.6, 1.5, dur)
    vib = 1 + 0.004 * np.sin(TWO_PI * 5.2 * t + r.uniform(0, TWO_PI)) * np.clip((t - attack) / 0.9, 0, 1)
    out = np.zeros((n, 2))
    for mult, cents, p, g, nh in [(1, -9, -0.6, 1.0, 40), (1, 7, 0.6, 1.0, 40), (1.5, 3, -0.2, 0.45, 30),
                                  (2, 4, 0.3, 0.5, 24), (2, -12, -0.3, 0.5, 24)]:
        fr = f * mult * 2 ** (cents / 1200) * vib
        out += E.pan(_additive(fr, nh, phase0=r.uniform(0, TWO_PI)) * g, p)
    cut = 180.0 + 2400.0 * fenv
    out = E.lp_sweep_blocks(out, cut, Q=1.1)
    out = (out.T * env).T
    out = E.saturate(out, 3.0, 0.8)
    out = E.eq(out, ('peak', 800, 1.0, 3.0), ('peak', 2200, 1.0, 2.0), ('lp', 4000, 0.7), ('lp', 6000, 0.7))
    sub = np.sin(TWO_PI * f / 2 * t) * env * 0.5 + np.sin(TWO_PI * f * t) * env * 0.3
    y = _norm(out, 0.5) + E.stereo(sub) * 0.5
    if gm_layer:
        evs = [(0.0, int(root_midi), float(dur), 0.95), (0.0, int(root_midi) + 7, float(dur), 0.85), (0.0, int(root_midi) + 12, float(dur), 0.8)]
        tb = _cached_render(evs, 57, dur + tail, 'braam')
        tu = _cached_render([(0.0, int(root_midi), float(dur), 0.95)], 58, dur + tail, 'braam')
        gm = _norm(tb, 0.5) + _norm(tu, 0.5) * 0.6
        gm = E.saturate(gm, 2.0, 0.6)
        gm = E.eq(gm, ('hp', 50), ('lp', 5000))
        y = y + _norm(gm, 0.5) * 0.6
    y = hall(y, mix=0.3, decay=2.8)
    y = E.eq(y, ('hp', 28, 0.7))
    y = E.fade(y, 0, 0.4)
    return _norm(y, 0.5)


__all__ = ['orch', 'drone', 'riser', 'impact', 'sub_drop', 'downlifter', 'metal_hit', 'reverse_cymbal', 'noise_bed',
           'braam', 'industrial_clank', 'timpani_roll', 'orchestra_hit', 'gm_oneshot', 'hall']
