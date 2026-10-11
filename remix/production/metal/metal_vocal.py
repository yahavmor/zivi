"""metal_vocal.py - vocal variants for the METAL remix.

Public API
----------
variant(name, kind, **kw) -> stereo float64 array (n, 2), or (array, lead_in_s) for kind='reverse_swell'
    name  : a key of vocal.WORDS / vocal.PHRASES (file ../vox/w_<name>.wav), or 'clean' (../vox/clean.wav)
    kind  : one of KINDS (below). Extra keyword arguments tune the variant (defaults are the song's settings).
    All outputs are peak-normalised to `peak` (default 0.5) so the song mixes them with plain dB gains.
    Results are memoised in RAM (same name/kind/kw -> same array) and every Rubber Band render is cached
    on disk in ./cache/rb_<name>_<semis><F|N>.wav, so repeated calls in the song cost nothing.

KINDS and signal paths (BEAT = 60/110 s; all chains start from the already-cleaned mono word):
    'raw'           the word as stored (for inspection)
    'clean'         metal lead-vocal chain: HPF 100 Hz (12 dB/oct), -1.5 dB mud cut @350, +3 dB presence @3 kHz,
                    +1.5 dB air shelf @9 kHz, light compression (-16 dB, 2.5:1, 8/120 ms), tanh saturation
                    (drive 1.4, 35 % mix). No reverb. Same length as the word.
    'hall'          clean + huge arena hall (engine.reverb decay 3.5 s, predelay 40 ms, hf_damp 0.5, lowcut 150,
                    highcut 8 kHz) at 40 % wet; the wet is ducked 12 dB while the voice speaks (5 ms attack /
                    150 ms release) so the words stay intelligible (whisper: 8/8 words on 'full' vs 2/8 un-ducked)
                    and the tail blooms in the gaps and after the word. Length = word + 3.64 s tail.
    'double'        two Rubber Band copies (-6 c / +6 c, formant preserved) through the clean chain, each with
                    a 12 ms base delay plus an independent slow (<0.3 Hz) random timing wander of +-4 ms (so
                    the inter-channel offset moves like two real takes instead of a fixed Haas delay), panned
                    L/R (+-0.9). Nominal onset is 12 ms into the array (inaudible, keeps everything causal).
    'demon'         Rubber Band -12 st formant-preserved (+ a -10 dB non-formant copy for chest/body) ->
                    HPF 80 (24 dB/oct, keeps the ~116 Hz fundamental, drops the sub), +2 dB @150,
                    compressor -> tanh drive 3.0 (85 %) -> 2nd-order LP 2.5 kHz -> light chorus
                    (0.35 Hz, 2.5 ms, 30 %).
    'shout_stack'   centre clean + octave-down (-12 st, formants NOT preserved = big chest, -8 dB, HPF 110,
                    LP 5 kHz) + octave-up (+12 st, formant preserved, -12 dB, HPF 300) -> tanh saturate(1.5).
    'megaphone'     clean -> 24 dB/oct band-pass 500-3000 Hz, +6 dB honk @1.6 kHz -> hard clip (x4) ->
                    horn band-limit (24 dB/oct LP 3.45k / HP 400) + band-limited hiss at -36 dB.
    'radio'         clean -> 24 dB/oct band-pass 300-3400, +3 dB @2 kHz, hard compression (6:1), drive ->
                    bitcrush 10 bit / 2x sample-hold -> LP 3.4k -> + flickering static at -42 dB.
    'stutter'       first `grid` beats (0.25 = 16th = 136.4 ms, 0.125 = 32nd = 68.2 ms) repeated `reps` times
                    with per-repeat decay, each repeat at the sample-exact grid position round(k*grid*BEAT*SR); by
                    default the last repeat is the whole word (the "landing"). Length = that position + word.
    'reverse_swell' reverse(reverb(reverse(word))) wet-only lead-in that crescendos into the word onset,
                    trimmed at its -55 dB start and cut exactly where the word begins (6 ms half-cosine declick on
                    the cut, so it can also be used standalone as a riser). Returns (array, lead_in)
                    with lead_in = len(array)/SR, so the caller places it at t - lead_in. with_word=True appends
                    the clean word after the swell (lead_in still points at the word onset).
    'glitch'        random 30-90 ms micro-slices of the clean word, shuffled, some repeated / reversed /
                    re-panned, 3 ms crossfade edges. Deterministic for a given `seed`.
    'slowmo'        engine.resample_speed(clean, 0.5): half speed, one octave down, 2x length.
    'tele'          engine.telephone(clean): 350-3200 band-pass + drive.

Helpers also exported: word(name) -> mono, word_path(name), clean_mono(name) -> mono, rb_shift(name, semis,
formant) -> mono (cached pitch shift), KINDS, BEAT, SR.
"""
import os
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORK = os.path.normpath(os.path.join(_HERE, '..'))
if _WORK not in sys.path:
    sys.path.insert(0, _WORK)

import numpy as np
from scipy import signal
import engine as E
import vocal as V

SR = E.SR
BPM = 110.0
BEAT = 60.0 / BPM
VOX_DIR = os.path.join(_WORK, 'vox')
CACHE = os.path.join(_HERE, 'cache')
os.makedirs(CACHE, exist_ok=True)

KINDS = ('raw', 'clean', 'hall', 'double', 'demon', 'shout_stack', 'megaphone', 'radio', 'stutter',
         'reverse_swell', 'glitch', 'slowmo', 'tele')


# ----------------------------------------------------------------------------- source material
_word_cache = {}


def word_path(name):
    if name == 'clean':
        return os.path.join(VOX_DIR, 'clean.wav')
    p = os.path.join(VOX_DIR, 'w_%s.wav' % name)
    if not os.path.exists(p):
        raise KeyError('unknown vocal slice %r (no %s)' % (name, p))
    return p


def word(name):
    """mono float64 of the stored (already cleaned/de-noised) slice."""
    if name not in _word_cache:
        _word_cache[name] = E.mono(E.read(word_path(name)))
    return _word_cache[name].copy()


def rb_shift(name, semis, formant=True):
    """Rubber Band R3 pitch shift of a stored slice, cached in ./cache. Returns mono, same length as the word."""
    out = os.path.join(CACHE, 'rb_%s_%+.3f%s.wav' % (name, semis, 'F' if formant else 'N'))
    args = ['-p', '%.4f' % semis] + (['-F'] if formant else [])
    y = V.rubberband(word_path(name), out, args)
    n = len(_word_cache[name]) if name in _word_cache else len(word(name))
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y))])
    return y[:n]


# ----------------------------------------------------------------------------- small helpers
def _name_seed(name):
    """stable per-name seed (str hash is randomised per process)."""
    return zlib.crc32(name.encode('utf-8')) % 1000


def _norm(x, peak=0.5):
    x = np.asarray(x, dtype=np.float64)
    m = np.abs(x).max()
    return x * (peak / m) if m > 1e-12 else x


def _stereo(x, peak):
    return E.stereo(_norm(x, peak))


def _wander(n, amp_s, rate_hz=0.3, seed=0):
    """slow random timing wander in samples, slope-limited by a 2nd-order lowpass at rate_hz."""
    hop = 480                                   # 10 ms control rate
    r = np.random.default_rng(seed)
    m_fix = int(12.0 * SR / hop)                # normalise over a fixed 12 s so short words don't get steep ramps
    m = max(m_fix, n // hop + 2)
    nz = r.standard_normal(m + 600)
    sos = signal.butter(2, rate_hz, fs=SR / hop, output='sos')
    w = signal.sosfilt(sos, nz)[600:600 + m]
    w = w / (np.abs(w[:m_fix]).max() + 1e-12) * amp_s * SR
    return np.interp(np.arange(n), np.arange(m) * hop, w)


def _vdelay(x, d_samples, extra):
    """variable fractional delay of mono x (delay curve in samples); output is len(x)+extra."""
    n = len(x) + extra
    xp = np.concatenate([x, np.zeros(extra)])
    d = np.concatenate([d_samples, np.full(extra, d_samples[-1])]) if len(d_samples) < n else d_samples[:n]
    idx = np.arange(n) - d
    return np.interp(idx, np.arange(n), xp, left=0.0, right=0.0)


def _env(x, ms=10.0):
    k = max(1, int(ms / 1000 * SR))
    e = np.abs(x)
    return signal.fftconvolve(e, np.ones(k) / k, mode='same')


def _duck(wet, key, depth_db=6.0, attack=0.005, release=0.15, hop=48):
    """gain-duck `wet` (n,) or (n,2) by the envelope of mono `key`: -depth_db while the key is within 12 dB of its
    own peak, recovering with `release` in the gaps (the classic vocal-reverb trick: the tail blooms between the
    words instead of washing over the consonants)."""
    wet = np.asarray(wet, dtype=np.float64)
    n = len(wet)
    k = np.abs(E.mono(key))[:n]
    if len(k) < n:
        k = np.concatenate([k, np.zeros(n - len(k))])
    npts = n // hop + 1
    lvl = k.copy()
    lvl.resize(npts * hop)
    lvl = lvl.reshape(npts, hop).max(1)
    env = E._env_follow(lvl, attack, release, hop)
    ctl = np.clip(env / (env.max() + 1e-12) / 0.25, 0.0, 1.0)
    g = np.interp(np.arange(n), np.arange(npts) * hop, E.db(-depth_db * ctl))
    return (wet.T * g).T


# ----------------------------------------------------------------------------- the lead-vocal chain
def clean_chain(x, hpf=100.0, presence_db=3.0, mud_db=-1.5, air_db=1.5, comp=True, sat_mix=0.35, peak=0.5):
    y = E.eq(x, ('hp', hpf, 0.707), ('peak', 350, 1.0, mud_db), ('peak', 3000, 1.0, presence_db),
             ('hs', 9000, 0.7, air_db))
    if comp:
        y = E.compress(y, thresh_db=-16, ratio=2.5, attack=0.008, release=0.12, knee=6.0)
    y = E.saturate(y, 1.4, sat_mix)
    return _norm(y, peak)


def clean_mono(name, **kw):
    return clean_chain(word(name), **kw)


# ----------------------------------------------------------------------------- variants
def v_raw(name, peak=0.5):
    return _stereo(word(name), peak)


def v_clean(name, peak=0.5, **kw):
    return E.stereo(clean_mono(name, peak=peak, **kw))


def v_hall(name, decay=3.5, predelay=0.04, mix=0.4, hf_damp=0.5, duck_db=12.0, duck_release=0.15, peak=0.5):
    d = clean_mono(name)
    tail = int((decay + predelay + 0.1) * SR)
    x = np.concatenate([d, np.zeros(tail)])
    wet = E.reverb(x, decay=decay, predelay=predelay, hf_damp=hf_damp, width=1.0, mix=1.0, lowcut=150,
                   highcut=8000)
    if duck_db:
        wet = _duck(wet, x, duck_db, 0.005, duck_release)
    y = E.stereo(x) * (1.0 - mix) + wet * mix
    return _norm(y, peak)


def v_double(name, cents=6.0, offset_ms=12.0, wander_ms=4.0, pan=0.9, seed=0, peak=0.5):
    a = clean_chain(rb_shift(name, -cents / 100.0, True), peak=1.0)
    b = clean_chain(rb_shift(name, +cents / 100.0, True), peak=1.0)
    n = len(a)
    extra = int((offset_ms + wander_ms) / 1000 * SR) + 8
    sd = _name_seed(name) + seed * 7919
    base = offset_ms / 1000 * SR
    dl = base + _wander(n, wander_ms / 1000, 0.3, sd + 1)
    dr = base + _wander(n, wander_ms / 1000, 0.3, sd + 2)
    yl = _vdelay(a, dl, extra)
    yr = _vdelay(b, dr, extra)
    y = E.pan(yl, -pan) + E.pan(yr, +pan)
    return _norm(y, peak)


def v_demon(name, semis=-12.0, drive=3.0, lp_hz=2500.0, body_db=-10.0, chorus_mix=0.3, peak=0.5):
    x = rb_shift(name, semis, True)
    if body_db is not None and body_db > -60:
        x = x + rb_shift(name, semis, False) * E.db(body_db)
    y = E.eq(x, ('hp', 80, 0.707), ('hp', 80, 0.707), ('peak', 150, 1.0, 2.0), ('peak', 400, 1.0, -1.0))
    y = E.compress(y, thresh_db=-18, ratio=3.0, attack=0.01, release=0.15, knee=6.0)
    y = _norm(y, 1.0)
    y = E.saturate(y, drive, 0.85)
    y = E.lp(y, lp_hz, 0.8, order=2)
    y = E.chorus(E.stereo(y), rate=0.35, depth=0.0025, mix=chorus_mix, voices=2, base=0.015)
    y = E.eq(y, ('hp', 40, 0.707))                 # DC block after the waveshaper
    return _norm(y, peak)


def v_shout_stack(name, down_db=-8.0, up_db=-12.0, drive=1.5, peak=0.5):
    c = clean_chain(word(name), peak=1.0)
    d = clean_chain(rb_shift(name, -12.0, False), hpf=110.0, peak=1.0)
    d = E.lp(d, 5000.0, 0.707, order=1)
    u = clean_chain(rb_shift(name, +12.0, True), hpf=300.0, peak=1.0)
    y = c + d * E.db(down_db) + u * E.db(up_db)
    y = E.saturate(y, drive, 1.0)
    y = E.eq(y, ('hp', 40, 0.707))                 # DC block after the waveshaper
    return _stereo(y, peak)


def v_megaphone(name, lo=500.0, hi=3000.0, clip_drive=4.0, noise_db=-36.0, seed=0, peak=0.5):
    x = clean_chain(word(name), peak=1.0)
    y = E.eq(x, ('hp', lo, 0.9), ('hp', lo, 0.9), ('lp', hi, 0.9), ('lp', hi, 0.9), ('peak', 1600, 1.5, 6.0))
    y = np.clip(_norm(y, 1.0) * clip_drive, -1.0, 1.0)
    y = E.eq(y, ('lp', hi * 1.15, 0.8), ('lp', hi * 1.15, 0.8), ('hp', lo * 0.8, 0.8))
    nz = E.eq(E.noise(len(y), seed + 101), ('hp', lo, 0.9), ('lp', hi, 0.9), ('peak', 1500, 0.6, 3.0))
    nz = _norm(nz, 1.0) * E.db(noise_db) * np.abs(y).max()
    nz = E.fade(nz, 0.005, 0.06)
    return _stereo(y + nz, peak)


def v_radio(name, lo=300.0, hi=3400.0, bits=10, down=2, noise_db=-42.0, seed=0, peak=0.5):
    x = clean_chain(word(name), peak=1.0)
    y = E.eq(x, ('hp', lo, 0.8), ('hp', lo, 0.8), ('lp', hi, 0.8), ('lp', hi, 0.8), ('peak', 2000, 1.2, 3.0))
    y = E.compress(y, thresh_db=-20, ratio=6.0, attack=0.004, release=0.08, knee=4.0)
    y = E.saturate(_norm(y, 1.0), 2.0, 0.6)
    y = E.bitcrush(y, bits, down)
    y = E.eq(y, ('lp', hi, 0.8), ('hp', 40, 0.707))
    r = np.random.default_rng(seed + 202)
    nz = E.eq(r.standard_normal(len(y)), ('hp', lo, 0.8), ('lp', hi, 0.8))
    flick = 1.0 + 0.5 * signal.sosfilt(signal.butter(2, 8.0, fs=SR, output='sos'), r.standard_normal(len(y))) * 20
    nz = _norm(nz * np.clip(flick, 0.2, 3.0), 1.0) * E.db(noise_db) * np.abs(y).max()
    nz = E.fade(nz, 0.005, 0.05)
    return _stereo(y + nz, peak)


def v_stutter(name, reps=4, grid=0.25, decay=0.08, full_last=True, slice_len=None, fade_ms=3.0, peak=0.5):
    x = clean_chain(word(name), peak=1.0)
    pos = [int(round(i * grid * BEAT * SR)) for i in range(reps)]      # sample-exact grid, no accumulated rounding
    cell = int(np.floor(grid * BEAT * SR + 1e-9))                       # shortest distance between two grid points
    sl = int(round((slice_len if slice_len else grid * BEAT) * SR))
    sl = max(1, min(sl, cell, len(x)))                                  # a slice never spills into the next cell
    s = E.fade(x[:sl], fade_ms / 1000, fade_ms / 1000)
    total = pos[-1] + len(x) if full_last else int(round(reps * grid * BEAT * SR))
    y = np.zeros(total)
    for i, p in enumerate(pos):
        if i == reps - 1 and full_last:
            y[p:p + len(x)] += x                 # the landing: whole word, full level
        else:
            y[p:p + sl] += s * (1.0 - decay) ** i
    return _stereo(y, peak)


def v_reverse_swell(name, decay=2.5, hf_damp=0.5, trim_db=-55.0, with_word=False, end_fade_ms=6.0, peak=0.5):
    w = clean_mono(name, peak=1.0)
    n = len(w)
    tail = int((decay * 1.2 + 0.1) * SR)
    xr = np.concatenate([w[::-1], np.zeros(tail)])
    wet = E.reverb(xr, decay=decay, predelay=0.0, hf_damp=hf_damp, width=1.0, mix=1.0, lowcut=150,
                   highcut=7000)
    sw = wet[::-1]                               # (tail + n, 2): [0:tail] is the lead-in, [tail:] overlaps the word
    lead = sw[:tail]
    env = _env(lead.mean(1), 20.0)
    thr = env.max() * E.db(trim_db)
    i0 = int(np.argmax(env > thr))
    lead = lead[i0:]
    lead = E.fade(lead, min(0.05, len(lead) / SR / 2), 0.0)
    k = int(end_fade_ms / 1000 * SR)
    if 1 < k < len(lead):                        # half-cosine declick: the swell stops at the onset without a step
        lead[-k:] *= (0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, k))))[:, None]
    lead_in = len(lead) / SR
    if with_word:
        lead = _norm(lead, np.abs(w).max() * 0.8)
        y = np.vstack([lead, E.stereo(w)])
    else:
        y = lead
    return _norm(y, peak), lead_in


def v_glitch(name, seed=0, lo_ms=30.0, hi_ms=90.0, repeat_p=0.3, reverse_p=0.15, pan_amt=0.4, fade_ms=3.0,
             peak=0.5):
    x = clean_chain(word(name), peak=1.0)
    r = np.random.default_rng(seed + _name_seed(name))
    n = len(x)
    cuts = [0]
    while cuts[-1] < n:
        cuts.append(cuts[-1] + int(r.uniform(lo_ms, hi_ms) / 1000 * SR))
    cuts[-1] = n
    slices = [x[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 64]
    order = list(range(len(slices)))
    r.shuffle(order)
    out = []
    k = int(fade_ms / 1000 * SR)
    for i in order:
        s = slices[i]
        if r.random() < reverse_p:
            s = s[::-1]
        reps = 2 if r.random() < repeat_p else 1
        for _ in range(reps):
            seg = E.fade(s, fade_ms / 1000, fade_ms / 1000) * r.uniform(0.7, 1.0)
            out.append(E.pan(seg, r.uniform(-pan_amt, pan_amt)))
    # overlap-add with the fade length as crossfade
    total = sum(len(o) for o in out) - k * (len(out) - 1)
    y = np.zeros((max(total, 1), 2))
    pos = 0
    for o in out:
        y[pos:pos + len(o)] += o
        pos += len(o) - k
    y = E.eq(y, ('hp', 90, 0.707))                 # the slice edges thump below the voice; keep it out of the kick's band
    return _norm(y, peak)


def v_slowmo(name, factor=0.5, peak=0.5):
    y = E.resample_speed(E.stereo(clean_mono(name)), factor)
    y = E.lp(y, 9000.0, 0.707, order=1)
    return _norm(y, peak)


def v_tele(name, drive=3.0, peak=0.5):
    return _stereo(E.telephone(clean_mono(name), drive), peak)


_KINDS = {
    'raw': v_raw, 'clean': v_clean, 'hall': v_hall, 'double': v_double, 'demon': v_demon,
    'shout_stack': v_shout_stack, 'megaphone': v_megaphone, 'radio': v_radio, 'stutter': v_stutter,
    'reverse_swell': v_reverse_swell, 'glitch': v_glitch, 'slowmo': v_slowmo, 'tele': v_tele,
}

_mem = {}


def variant(name, kind, **kw):
    """stereo (n,2) float64 for the word/phrase `name` processed as `kind`; (array, lead_in_s) for reverse_swell."""
    if kind not in _KINDS:
        raise KeyError('unknown vocal kind %r; choose from %s' % (kind, KINDS))
    key = (name, kind, tuple(sorted(kw.items())))
    if key not in _mem:
        _mem[key] = _KINDS[kind](name, **kw)
    y = _mem[key]
    if isinstance(y, tuple):
        return y[0].copy(), y[1]
    return y.copy()


if __name__ == '__main__':
    import time
    t0 = time.time()
    for nm in ('meham3', 'p2'):
        for k in KINDS:
            y = variant(nm, k)
            if isinstance(y, tuple):
                y, li = y
                print('%-8s %-14s len %.3fs peak %.3f lead_in %.3fs' % (nm, k, len(y) / SR, np.abs(y).max(), li))
            else:
                print('%-8s %-14s len %.3fs peak %.3f' % (nm, k, len(y) / SR, np.abs(y).max()))
    print('total %.1fs' % (time.time() - t0))
