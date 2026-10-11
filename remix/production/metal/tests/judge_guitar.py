"""judge_guitar.py -- objective A/B comparison of metal guitar modules against the SPEC.md targets.

Usage:  python3 tests/judge_guitar.py [module ...] [--full] [--no-extras]
        default modules: metal_guitar_A metal_guitar_B   (pass `metal_guitar` to judge the merged module)
        --full : also time each public function with the real song event lists (riffs_3.json, ~113 s)

Material (identical for every module, 110 BPM, Drop C):
  R1  8 palm-muted C2 chugs at 16ths, then an open C5 power chord (root C2) held 2 s        -> render_rhythm
  R0  one isolated palm-muted C2 chug                                                      -> render_rhythm
  P   R1 riff + Eb5 / F5 chords + G2 chugs, double tracked                                 -> render_rhythm_pair
  L   lead phrase: G4 (vib) -> slide Bb4 -> C5 bend +2 (vib) -> long D5 (vib)              -> render_lead
  C   C-minor arpeggio, 16 notes                                                           -> render_clean
  X   extras (feedback_swell, pick_scrape, pinch_harmonic, dive_bomb)                       -> sanity only

Measurements per module (all PSDs = Welch, band energy = PSD summed over the band, dB rel. 250-500 Hz):
  bands 20-60 .. 8-16k on the chug region, the chord region, the pair and the lead
  thump      = E(65-200) / E(250-500) on the chugs               target  +1 .. +8 dB   (SPEC: strong 80-200)
  scoop      = PSD density 400-700 vs mean of 200-400 & 700-1200 target  -2 .. -9 dB   (SPEC: dip 400-600)
  presence   = E(2-4k) / E(250-500) on the chord                 target  -6 .. +3 dB   (SPEC: presence 2-4k)
  rolloff    = density 6-8k vs 2-4k                              target  < -18 dB      (SPEC: steep > 5.5 kHz)
  fizz       = density 8-12k vs 2-4k                             target  < -35 dB
  chug decay = ms from envelope peak to -20 dB (isolated chug)   target  50 .. 170 ms
  chug gap   = body RMS vs the 20 ms before the next chug        target  >= 6 dB
  sustain    = chord level at +1.5 s rel. its peak               target  -14 .. -0.5 dB
  density    = spectral flatness 200-4000 Hz of the chord (0.3-1.3 s), peak count < 5 kHz within 40 dB
  crest      = 20 log10(peak / rms) over the active part         target  8 .. 16 dB (rhythm), 7 .. 15 (lead)
  pair       = L/R waveform corr (target 0 .. 0.7, i.e. not 1), envelope corr (>= 0.6), S/M ratio (+-3 dB)
  lead       = HPS pitch tracking: bend reached, vibrato depth / rate, sustain at 1.0 / 1.5 s, bands
  clean      = peak, bands, L/R corr, crest, ring-out
  DC, NaN, clipping, peak, render time (and --full song timing)
Each target miss adds its distance (in the metric's unit) to a penalty; lower penalty = better. Spectrogram
panels are saved to tests/judge_<module>_rhythm.png and tests/judge_<module>_leadclean.png -- look at them.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, '..'))
import importlib
import numpy as np
from scipy import signal
import engine as E

SR = E.SR
BEAT = 60 / 110.0
S16 = BEAT / 4
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
FULL = '--full' in sys.argv
NO_EXTRAS = '--no-extras' in sys.argv
MODULES = [a for a in sys.argv[1:] if not a.startswith('--')] or ['metal_guitar_A', 'metal_guitar_B']


# ------------------------------------------------------------------------------------------- material
def riff_r1():
    ev = [(i * S16, 36, S16 * 0.8, 1.0 if i % 4 == 0 else 0.9, True) for i in range(8)]
    ev.append((8 * S16, 36, 2.0, 1.0, False))
    return ev, 8 * S16, 4.0


def riff_pair():
    ev, t_ch, _ = riff_r1()
    t = t_ch + 2.1
    ev.append((t, 39, 0.8, 0.95, False)); t += 0.85
    ev.append((t, 41, 0.8, 0.95, False)); t += 0.85
    ev += [(t + i * S16, 43, S16 * 0.8, 0.9, True) for i in range(4)]
    t += 4 * S16
    ev.append((t, 46, 1.2, 0.9, False))
    return ev, 7.0


def lead_phrase():
    # (t, midi, dur, vel, bend_semis, vibrato, slide_from)
    return [(0.0, 67, 0.5, 0.9, 0.0, 0.3, None),
            (0.5, 70, 0.5, 0.9, 0.0, 0.0, 67),
            (1.0, 72, 1.5, 0.95, 2.0, 0.4, None),
            (2.6, 74, 2.2, 0.9, 0.0, 0.35, None)], 5.5


def clean_arp():
    notes = [48, 55, 60, 63, 67, 63, 60, 55] * 2
    return [(i * 0.25, m, 0.5, 0.55 + 0.1 * (i % 3 == 0)) for i, m in enumerate(notes)], 7.0


# ------------------------------------------------------------------------------------------- measurement helpers
def mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(1)


def welch(x, nperseg=8192):
    return signal.welch(mono(x), SR, nperseg=min(nperseg, len(x)))


def band_energies(x, nperseg=8192):
    f, p = welch(x, nperseg)
    ref = p[(f >= 250) & (f < 500)].sum() + 1e-30
    return {'%d-%d' % b: float(10 * np.log10(p[(f >= b[0]) & (f < b[1])].sum() / ref + 1e-30)) for b in BANDS}


def density_db(f, p, lo, hi):
    sel = (f >= lo) & (f < hi)
    return float(10 * np.log10(p[sel].mean() + 1e-30))


def rhythm_tone(x):
    """thump / scoop / presence / rolloff / fizz of a segment."""
    f, p = welch(x)

    def E_(lo, hi):
        return p[(f >= lo) & (f < hi)].sum() + 1e-30
    return dict(thump=float(10 * np.log10(E_(65, 200) / E_(250, 500))),
                scoop=float(density_db(f, p, 400, 700) - 0.5 * (density_db(f, p, 200, 400) + density_db(f, p, 700, 1200))),
                presence=float(10 * np.log10(E_(2000, 4000) / E_(250, 500))),
                rolloff=float(density_db(f, p, 6000, 8000) - density_db(f, p, 2000, 4000)),
                fizz=float(density_db(f, p, 8000, 12000) - density_db(f, p, 2000, 4000)))


def rms_env(x, win=0.005):
    k = max(1, int(win * SR))
    return np.sqrt(np.convolve(mono(x) ** 2, np.ones(k) / k, 'same'))


def decay_ms(x, levels=(-20, -40)):
    e = 20 * np.log10(rms_env(x) + 1e-12)
    i0 = int(np.argmax(e)); pk = e[i0]
    out = {}
    for L in levels:
        idx = np.where(e[i0:] <= pk + L)[0]
        out[str(L)] = float(idx[0] / SR * 1000) if len(idx) else None
    return out, i0 / SR


def level_rel_peak(x, t0, offsets):
    e = rms_env(x, 0.02)
    seg = e[int(t0 * SR):int((t0 + 2.0) * SR)]
    pk = seg.max() + 1e-12
    return {str(o): float(20 * np.log10(e[int((t0 + o) * SR)] / pk + 1e-12)) for o in offsets}


def chug_gap(x, times):
    """mean dB dip between the body of chug i and the 20 ms before chug i+1."""
    m = mono(x); dips = []
    for a, b in zip(times[:-1], times[1:]):
        body = m[int((a + 0.008) * SR):int((a + 0.06) * SR)]
        gap = m[int((b - 0.022) * SR):int((b - 0.002) * SR)]
        dips.append(10 * np.log10(np.mean(body ** 2) / (np.mean(gap ** 2) + 1e-20)))
    return float(np.mean(dips)), float(np.min(dips))


def harmonic_density(x, f_line=None):
    """spectral flatness in 200-4000 Hz (Welch 16384) + number of spectral peaks < 5 kHz within 40 dB of max
    + line-to-floor contrast at multiples of f_line (dB)."""
    f, p = welch(x, 16384)
    sel = (f >= 200) & (f < 4000)
    ps = p[sel] + 1e-30
    flat = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
    pdb = 10 * np.log10(p + 1e-30)
    s5 = f < 5000
    pk, _ = signal.find_peaks(pdb[s5], prominence=6, height=pdb[s5].max() - 40)
    out = dict(flatness=flat, flatness_db=float(10 * np.log10(flat)), n_peaks=int(len(pk)))
    if f_line:
        df = f[1]; ks = np.arange(int(np.ceil(300 / f_line)), int(1500 / f_line))
        lines, mids = [], []
        for k in ks:
            i = int(round(k * f_line / df)); j = int(round((k + 0.5) * f_line / df))
            lines.append(pdb[i - 1:i + 2].max()); mids.append(pdb[j - 1:j + 2].min())
        out['line_floor_db'] = float(np.median(lines) - np.median(mids))
    return out


def crest_db(x):
    m = mono(x); e = rms_env(m, 0.02); act = e > e.max() * 10 ** (-40 / 20)
    pk = np.abs(np.asarray(x)).max(); rms = np.sqrt(np.mean(m[act] ** 2)) + 1e-12
    return float(20 * np.log10(pk / rms))


def basic(x):
    x = np.asarray(x)
    m = mono(x)
    sub = E.lp(m, 20, order=2)
    return dict(nan=bool(~np.isfinite(x).all()), peak=float(np.abs(x).max()), rms=float(np.sqrt(np.mean(m ** 2))),
                dc=float(m.mean()), sub20_dbfs=float(20 * np.log10(np.sqrt(np.mean(sub ** 2)) + 1e-12)),
                clipped=int((np.abs(x) > 0.99).sum()), crest=crest_db(x), len_s=float(len(x) / SR))


def stereo_stats(x):
    L, R = x[:, 0], x[:, 1]
    wf = float(np.corrcoef(L, R)[0, 1])
    eL, eR = rms_env(L, 0.02), rms_env(R, 0.02)
    env = float(np.corrcoef(eL, eR)[0, 1])
    lo = float(np.corrcoef(E.lp(L, 150, order=2), E.lp(R, 150, order=2))[0, 1])
    M = (L + R) / 2; S = (L - R) / 2
    sm = float(10 * np.log10(np.mean(S ** 2) / (np.mean(M ** 2) + 1e-20) + 1e-20))
    bal = float(20 * np.log10((np.sqrt(np.mean(L ** 2)) + 1e-12) / (np.sqrt(np.mean(R ** 2)) + 1e-12)))
    return dict(corr_wave=wf, corr_env=env, corr_low150=lo, side_mid_db=sm, lr_balance_db=bal, width=float(1 - abs(wf)))


def track_pitch(x, fmin=150, fmax=1500, hop=0.01, win=0.05, nh=4):
    """harmonic-product-spectrum f0 tracker (robust on distorted tones). returns (times, f0 Hz)."""
    x = mono(x); n = len(x); k = int(win * SR); h = int(hop * SR); nfft = 1 << 16
    fr = np.fft.rfftfreq(nfft, 1 / SR); df = fr[1]
    i_lo, i_hi = int(fmin / df), int(fmax / df)
    ts, fs = [], []
    w = np.hanning(k)
    for i in range(0, n - k, h):
        seg = x[i:i + k]
        if np.sqrt(np.mean(seg ** 2)) < 2e-3:
            continue
        P = np.log(np.abs(np.fft.rfft(seg * w, nfft)) + 1e-9)
        hps = np.zeros(i_hi - i_lo)
        for m in range(1, nh + 1):
            hps += P[i_lo * m:i_hi * m:m][:i_hi - i_lo]
        j = int(np.argmax(hps)) + i_lo
        ts.append(i / SR + win / 2); fs.append(fr[j])
    return np.array(ts), np.array(fs)


def lead_pitch_report(y, events):
    ts, fs = track_pitch(y)
    out = {}
    if len(ts) == 0:
        return out
    cents = lambda f, m: 1200 * np.log2(f / E.midi2hz(m))
    # note 3: C5 bend +2 with vibrato
    t0, m, d = events[2][0], events[2][1], events[2][2]
    a = (ts > t0 + 0.02) & (ts < t0 + 0.09); b = (ts > t0 + 0.9) & (ts < t0 + d - 0.1)
    if a.any() and b.any():
        out['bend_start_semis'] = float(np.median(cents(fs[a], m)) / 100)
        out['bend_end_semis'] = float(np.median(cents(fs[b], m)) / 100)
    # note 4: long D5 with vibrato: p-p depth (cents) and rate
    t0, m, d = events[3][0], events[3][1], events[3][2]
    s = (ts > t0 + 0.6) & (ts < t0 + 1.8)
    if s.sum() > 20:
        c = cents(fs[s], m); c = c - np.median(c)
        out['vib_pp_cents'] = float(np.percentile(c, 95) - np.percentile(c, 5))
        out['vib_mean_cents'] = float(np.median(cents(fs[s], m)))
        cc = c - c.mean(); sp = np.abs(np.fft.rfft(cc * np.hanning(len(cc)), 4096)); fr = np.fft.rfftfreq(4096, 0.01)
        sel = (fr > 2) & (fr < 12)
        out['vib_rate_hz'] = float(fr[sel][np.argmax(sp[sel])])
    # note 2: slide from G4 to Bb4
    t0, m = events[1][0], events[1][1]
    a = (ts > t0 + 0.0) & (ts < t0 + 0.05); b = (ts > t0 + 0.3) & (ts < t0 + 0.48)
    if a.any() and b.any():
        out['slide_start_semis'] = float(np.median(cents(fs[a], m)) / 100)
        out['slide_end_semis'] = float(np.median(cents(fs[b], m)) / 100)
    # sustain of the long note
    e = rms_env(y, 0.02); t0 = events[3][0]
    pk = e[int(t0 * SR):int((t0 + 0.3) * SR)].max() + 1e-12
    out['sustain_1p0_db'] = float(20 * np.log10(e[int((t0 + 1.0) * SR)] / pk))
    out['sustain_1p5_db'] = float(20 * np.log10(e[int((t0 + 1.5) * SR)] / pk))
    return out


# ------------------------------------------------------------------------------------------- scoring
def miss(v, lo=None, hi=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 10.0
    if lo is not None and v < lo:
        return float(lo - v)
    if hi is not None and v > hi:
        return float(v - hi)
    return 0.0


def score_rhythm(R):
    P = {}
    P['nan/clip'] = 100.0 * (R['chug_basic']['nan'] or R['chord_basic']['nan']) + 1.0 * (R['r1_basic']['clipped'] > 0)
    P['peak'] = 10 * miss(R['r1_basic']['peak'], 0.3, 0.6)
    P['dc'] = 10 * miss(abs(R['r1_basic']['dc']), None, 0.003)
    P['thump'] = miss(R['tone_chug']['thump'], 1.0, 8.0)
    P['scoop'] = miss(R['tone_chord']['scoop'], -9.0, -2.0)
    P['presence'] = miss(R['tone_chord']['presence'], -6.0, 3.0)
    P['rolloff'] = 0.5 * miss(R['tone_chord']['rolloff'], None, -18.0)
    P['fizz'] = 0.5 * miss(R['tone_chord']['fizz'], None, -35.0)
    P['decay20'] = 0.05 * miss(R['chug_decay']['-20'], 50.0, 170.0)
    P['decay40'] = 0.02 * miss(R['chug_decay']['-40'], None, 320.0)
    P['gap'] = miss(R['chug_gap_mean'], 6.0, None)
    P['sustain'] = miss(R['chord_level']['1.5'], -14.0, -0.5)
    P['crest'] = miss(R['r1_basic']['crest'], 8.0, 16.0)
    P['pair_wave'] = 5 * miss(R['pair']['corr_wave'], -0.2, 0.7)
    P['pair_env'] = 5 * miss(R['pair']['corr_env'], 0.6, None)
    P['pair_sm'] = miss(abs(R['pair']['side_mid_db']), None, 3.0)
    P['pair_bal'] = miss(abs(R['pair']['lr_balance_db']), None, 2.0)
    P['time'] = 0.2 * miss(R['t_pair_s'] / R['pair_basic']['len_s'] * 113.0, None, 60.0)   # extrapolated song time
    return P


def score_lead(R):
    P = {}
    P['nan/clip'] = 100.0 * R['basic']['nan'] + 1.0 * (R['basic']['clipped'] > 0)
    P['peak'] = 10 * miss(R['basic']['peak'], 0.3, 0.6)
    P['dc'] = 10 * miss(abs(R['basic']['dc']), None, 0.003)
    p = R['pitch']
    P['bend'] = 2 * miss(p.get('bend_end_semis'), 1.7, 2.4) + miss(p.get('bend_start_semis'), -0.4, 0.5)
    P['vib_depth'] = 0.02 * miss(p.get('vib_pp_cents'), 20.0, 70.0)
    P['vib_rate'] = miss(p.get('vib_rate_hz'), 4.5, 6.5)
    P['slide'] = miss(p.get('slide_start_semis'), -3.4, -1.0) + miss(p.get('slide_end_semis'), -0.4, 0.4)
    P['sustain'] = 0.5 * miss(p.get('sustain_1p5_db'), -12.0, 0.5)
    P['lowcut'] = 0.5 * miss(R['bands']['60-120'], None, -12.0)
    P['rolloff'] = 0.5 * miss(R['tone']['rolloff'], None, -15.0)
    P['crest'] = miss(R['basic']['crest'], 7.0, 15.0)
    return P


def score_clean(R):
    P = {}
    P['nan/clip'] = 100.0 * R['basic']['nan'] + 1.0 * (R['basic']['clipped'] > 0)
    P['peak'] = 10 * miss(R['basic']['peak'], 0.25, 0.6)
    P['dc'] = 10 * miss(abs(R['basic']['dc']), None, 0.003)
    P['corr'] = 5 * miss(R['stereo']['corr_wave'], 0.2, 0.9)
    P['crest'] = miss(R['basic']['crest'], 10.0, 22.0)
    P['hf'] = 0.3 * miss(R['bands']['8000-16000'], None, -30.0)
    P['lowmud'] = 0.3 * miss(R['bands']['60-120'], None, 0.0)
    return P


# ------------------------------------------------------------------------------------------- spectrograms
def spec_panel(ax, x, title, fmax=12000, dyn=85):
    x = mono(x)
    f, tt, S = signal.spectrogram(x, SR, nperseg=2048, noverlap=1536, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    sel = f <= fmax
    ax.pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - dyn, vmax=Sdb.max())
    ax.set_yscale('symlog', linthresh=250); ax.set_ylim(40, fmax)
    ax.set_yticks([50, 100, 200, 500, 1000, 2000, 4000, 8000]); ax.set_yticklabels(['50', '100', '200', '500', '1k', '2k', '4k', '8k'])
    ax.set_title(title, fontsize=9); ax.set_ylabel('Hz')


def wave_panel(ax, x, title):
    x = mono(x); t = np.arange(len(x)) / SR
    ax.plot(t, x, lw=0.3, color='#333'); ax.set_xlim(0, t[-1]); ax.set_ylim(-0.7, 0.7); ax.set_title(title, fontsize=9)


def save_rhythm_png(mod, r1, r0, pair, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(14, 15))
    wave_panel(ax[0], r1, mod + ': R1 waveform (8 chugs + C5 chord)')
    spec_panel(ax[1], r1, mod + ': R1 spectrogram')
    spec_panel(ax[2], r0[:int(0.6 * SR)], mod + ': single chug (first 0.6 s)', fmax=12000)
    spec_panel(ax[3], pair[:, 0], mod + ': pair L')
    spec_panel(ax[4], pair[:, 1], mod + ': pair R')
    ax[4].set_xlabel('s')
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def save_leadclean_png(mod, lead, clean, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(14, 12))
    wave_panel(ax[0], lead, mod + ': lead waveform')
    spec_panel(ax[1], lead, mod + ': lead (G4 vib, slide Bb4, C5 bend+2 vib, D5 long vib)', fmax=12000)
    wave_panel(ax[2], clean, mod + ': clean waveform (mono sum)')
    spec_panel(ax[3], clean[:, 0], mod + ': clean L', fmax=12000)
    ax[3].set_xlabel('s')
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


# ------------------------------------------------------------------------------------------- run one module
def judge(modname):
    G = importlib.import_module(modname)
    R = {'module': modname}
    tag = modname.replace('metal_guitar', 'gtr').strip('_') or 'gtr'
    # ---- rhythm
    ev1, t_ch, L1 = riff_r1()
    t = time.time(); r1 = G.render_rhythm(ev1, seed=0, total_len=L1); R['t_r1_s'] = time.time() - t
    t = time.time(); r0 = G.render_rhythm([(0.0, 36, S16 * 0.8, 1.0, True)], seed=0, total_len=1.0); R['t_r0_s'] = time.time() - t
    evp, Lp = riff_pair()
    t = time.time(); pair = G.render_rhythm_pair(evp, total_len=Lp); R['t_pair_s'] = time.time() - t
    r1 = np.asarray(r1); r0 = np.asarray(r0); pair = np.asarray(pair)
    assert r1.ndim == 1 and r0.ndim == 1 and pair.ndim == 2 and pair.shape[1] == 2, (r1.shape, r0.shape, pair.shape)
    assert len(r1) == int(L1 * SR) and len(pair) == int(Lp * SR), (len(r1), len(pair))
    chug_seg = r1[:int(t_ch * SR)]
    chord_seg = r1[int((t_ch + 0.3) * SR):int((t_ch + 1.3) * SR)]
    R['r1_basic'] = basic(r1); R['chug_basic'] = basic(chug_seg); R['chord_basic'] = basic(chord_seg); R['pair_basic'] = basic(pair)
    R['bands_chug'] = band_energies(chug_seg); R['bands_chord'] = band_energies(chord_seg); R['bands_pair'] = band_energies(pair)
    R['tone_chug'] = rhythm_tone(chug_seg); R['tone_chord'] = rhythm_tone(chord_seg)
    R['chug_decay'], R['chug_peak_t'] = decay_ms(r0)
    R['chug_level_rel_peak'] = {k: float(v) for k, v in zip(['50ms', '100ms', '150ms', '200ms', '300ms'],
                                                           [20 * np.log10(rms_env(r0)[int(R['chug_peak_t'] * SR + o * SR / 1000)] / (rms_env(r0).max() + 1e-12) + 1e-12) for o in (50, 100, 150, 200, 300)])}
    R['chug_gap_mean'], R['chug_gap_min'] = chug_gap(r1, [e[0] for e in ev1[:8]])
    R['chord_level'] = level_rel_peak(r1, t_ch, (0.5, 1.0, 1.5, 1.9))
    R['density'] = harmonic_density(chord_seg, f_line=E.midi2hz(36) / 2)
    R['pair'] = stereo_stats(pair)
    R['pair_tone'] = rhythm_tone(pair)
    # ---- lead
    evl, Ll = lead_phrase()
    t = time.time(); lead = np.asarray(G.render_lead(evl, total_len=Ll)); R['t_lead_s'] = time.time() - t
    assert lead.ndim == 1 and len(lead) == int(Ll * SR)
    R['lead'] = dict(basic=basic(lead), bands=band_energies(lead), tone=rhythm_tone(lead), pitch=lead_pitch_report(lead, evl))
    # ---- clean
    evc, Lc = clean_arp()
    t = time.time(); clean = np.asarray(G.render_clean(evc, total_len=Lc)); R['t_clean_s'] = time.time() - t
    assert clean.ndim == 2 and clean.shape[1] == 2 and len(clean) == int(Lc * SR)
    R['clean'] = dict(basic=basic(clean), bands=band_energies(clean), stereo=stereo_stats(clean))
    e = rms_env(clean, 0.02); R['clean']['ring_after_last_db'] = float(20 * np.log10(e[int(5.0 * SR)] / (e.max() + 1e-12) + 1e-12))
    # ---- extras
    if not NO_EXTRAS:
        X = {}
        for name, call in [('feedback_swell', lambda: G.feedback_swell(55, 2.5)), ('pick_scrape', lambda: G.pick_scrape(1.0)),
                           ('pinch_harmonic', lambda: G.pinch_harmonic(48, 1.2)), ('dive_bomb', lambda: G.dive_bomb(48, 2.0))]:
            t = time.time()
            try:
                y = np.asarray(call()); X[name] = basic(y); X[name]['t_s'] = time.time() - t
            except Exception as ex:
                X[name] = {'error': repr(ex)}
        R['extras'] = X
    # ---- edge cases (contract)
    try:
        z = G.render_rhythm([], total_len=1.0); zl = G.render_lead([], total_len=1.0); zc = G.render_clean([], total_len=1.0)
        s = G.render_rhythm([(1.5, 36, 3.0, 1.0, False), (2.5, 36, 0.1, 1.0, True)], total_len=2.0)
        R['contract'] = dict(empty_ok=bool(len(z) == SR and len(zl) == SR and zc.shape == (SR, 2)), short_total_len_ok=bool(len(s) == 2 * SR),
                             no_total_len_len_s=float(len(G.render_rhythm([(0.0, 36, 1.0, 1.0, False)])) / SR))
    except Exception as ex:
        R['contract'] = {'error': repr(ex)}
    # ---- scores
    R['score_rhythm'] = score_rhythm(R); R['score_lead'] = score_lead(R['lead']); R['score_clean'] = score_clean(R['clean'])
    R['penalty'] = {k: float(sum(R['score_' + k].values())) for k in ('rhythm', 'lead', 'clean')}
    # ---- files
    E.write(os.path.join(HERE, 'judge_%s_r1.wav' % tag), r1)
    E.write(os.path.join(HERE, 'judge_%s_pair.wav' % tag), pair)
    E.write(os.path.join(HERE, 'judge_%s_lead.wav' % tag), lead)
    E.write(os.path.join(HERE, 'judge_%s_clean.wav' % tag), clean)
    save_rhythm_png(modname, r1, r0, pair, os.path.join(HERE, 'judge_%s_rhythm.png' % tag))
    save_leadclean_png(modname, lead, clean, os.path.join(HERE, 'judge_%s_leadclean.png' % tag))
    # ---- full song timing
    if FULL:
        R['full'] = full_timing(G)
    return R


def full_timing(G):
    import glob
    cands = sorted(glob.glob(os.path.join(ROOT, 'riffs*.json')))
    pick = [c for c in cands if c.endswith('riffs.json')] or [c for c in cands if 'placeholder' not in c] or cands
    if not pick:
        return {'error': 'no riffs json'}
    RIFFS = json.load(open(pick[-1]))
    ORDER = ['intro', 'first_entry', 'riffA', 'riffB', 'build', 'breakdown', 'solo', 'climax', 'outro']
    BAR = 4 * BEAT; b = 1; start = {}
    for s in ORDER:
        start[s] = b; b += RIFFS['sections'][s]['bars']
    LEN = (b - 1) * BAR + 6.0
    gtr, lead, clean = [], [], []
    for s in ORDER:
        sec = RIFFS['sections'][s]; t0 = (start[s] - 1) * BAR
        for e in sec.get('guitar', []):
            gtr.append((t0 + e['beat'] * BEAT, int(e['root']), float(e['dur']) * BEAT, float(e['vel']), bool(e.get('mute', False))))
        for e in sec.get('lead', []):
            lead.append((t0 + e['beat'] * BEAT, int(e['midi']), float(e['dur']) * BEAT, float(e['vel']), float(e.get('bend', 0) or 0), float(e.get('vibrato', 0) or 0), e.get('slide_from')))
        for e in sec.get('clean', []):
            clean.append((t0 + e['beat'] * BEAT, int(e['midi']), float(e['dur']) * BEAT, float(e['vel'])))
    out = {'riffs': os.path.basename(pick[-1]), 'len_s': LEN, 'n_gtr': len(gtr), 'n_lead': len(lead), 'n_clean': len(clean)}
    t = time.time(); y = G.render_rhythm_pair(gtr, total_len=LEN); out['t_pair_s'] = time.time() - t; out['pair'] = basic(y); out['pair_stereo'] = stereo_stats(np.asarray(y)); out['pair_bands'] = band_energies(y)
    if lead:
        t = time.time(); y = G.render_lead(lead, total_len=LEN); out['t_lead_s'] = time.time() - t; out['lead'] = basic(y)
    if clean:
        t = time.time(); y = G.render_clean(clean, total_len=LEN); out['t_clean_s'] = time.time() - t; out['clean'] = basic(y)
    out['t_total_s'] = out['t_pair_s'] + out.get('t_lead_s', 0) + out.get('t_clean_s', 0)
    return out


# ------------------------------------------------------------------------------------------- report
def fmt_bands(b):
    return ' '.join('%s:%+5.1f' % (k.replace('000', 'k').replace('k-', 'k-').replace('16k', '16k'), v) for k, v in b.items() if k != '250-500')


def print_report(results):
    names = [r['module'] for r in results]
    w = 30
    print('\n' + '=' * 110)
    print('RHYTHM'.ljust(w) + ''.join(n.ljust(26) for n in names))
    rows = [
        ('peak / rms / crest R1', lambda r: '%.3f / %.3f / %4.1f dB' % (r['r1_basic']['peak'], r['r1_basic']['rms'], r['r1_basic']['crest'])),
        ('DC / <20Hz dBFS / clipped', lambda r: '%+.5f / %5.1f / %d' % (r['r1_basic']['dc'], r['r1_basic']['sub20_dbfs'], r['r1_basic']['clipped'])),
        ('thump chug 65-200/250-500', lambda r: '%+5.1f dB  [tgt +1..+8]' % r['tone_chug']['thump']),
        ('thump chord', lambda r: '%+5.1f dB' % r['tone_chord']['thump']),
        ('scoop chord 400-700', lambda r: '%+5.1f dB  [tgt -9..-2]' % r['tone_chord']['scoop']),
        ('scoop chug', lambda r: '%+5.1f dB' % r['tone_chug']['scoop']),
        ('presence chord 2-4k', lambda r: '%+5.1f dB  [tgt -6..+3]' % r['tone_chord']['presence']),
        ('rolloff 6-8k vs 2-4k', lambda r: '%+5.1f dB  [tgt < -18]' % r['tone_chord']['rolloff']),
        ('fizz 8-12k vs 2-4k', lambda r: '%+5.1f dB  [tgt < -35]' % r['tone_chord']['fizz']),
        ('chug -20 / -40 dB (ms)', lambda r: '%s / %s  [tgt 50..170]' % (('%.0f' % r['chug_decay']['-20']) if r['chug_decay']['-20'] else 'never', ('%.0f' % r['chug_decay']['-40']) if r['chug_decay']['-40'] else 'never')),
        ('chug lvl @50/100/150/200ms', lambda r: '%.0f/%.0f/%.0f/%.0f dB' % tuple(r['chug_level_rel_peak'][k] for k in ('50ms', '100ms', '150ms', '200ms'))),
        ('chug gap mean / min', lambda r: '%4.1f / %4.1f dB  [tgt >= 6]' % (r['chug_gap_mean'], r['chug_gap_min'])),
        ('chord lvl @0.5/1.0/1.5/1.9s', lambda r: '%.1f/%.1f/%.1f/%.1f dB' % tuple(r['chord_level'][k] for k in ('0.5', '1.0', '1.5', '1.9'))),
        ('density flat(dB) / peaks<5k', lambda r: '%5.1f / %d' % (r['density']['flatness_db'], r['density']['n_peaks'])),
        ('line/floor 300-1500 (dB)', lambda r: '%5.1f' % r['density'].get('line_floor_db', float('nan'))),
        ('pair corr wave/env/low150', lambda r: '%.2f / %.2f / %.2f' % (r['pair']['corr_wave'], r['pair']['corr_env'], r['pair']['corr_low150'])),
        ('pair S/M dB / L-R bal dB', lambda r: '%+.1f / %+.1f' % (r['pair']['side_mid_db'], r['pair']['lr_balance_db'])),
        ('pair peak / rms / crest', lambda r: '%.3f / %.3f / %4.1f' % (r['pair_basic']['peak'], r['pair_basic']['rms'], r['pair_basic']['crest'])),
        ('time r1 / r0 / pair (s)', lambda r: '%.1f / %.1f / %.1f' % (r['t_r1_s'], r['t_r0_s'], r['t_pair_s'])),
        ('extrap. song pair time (s)', lambda r: '%.0f' % (r['t_pair_s'] / r['pair_basic']['len_s'] * 113)),
        ('PENALTY rhythm', lambda r: '%.2f' % r['penalty']['rhythm']),
    ]
    for lbl, fn in rows:
        print(lbl.ljust(w) + ''.join(fn(r).ljust(26) for r in results))
    for r in results:
        print('  bands chug  %-16s %s' % (r['module'], fmt_bands(r['bands_chug'])))
        print('  bands chord %-16s %s' % (r['module'], fmt_bands(r['bands_chord'])))
        print('  bands pair  %-16s %s' % (r['module'], fmt_bands(r['bands_pair'])))
        print('  penalties   %-16s %s' % (r['module'], ' '.join('%s=%.2f' % (k, v) for k, v in r['score_rhythm'].items() if v > 0)))
    print('-' * 110)
    print('LEAD'.ljust(w) + ''.join(n.ljust(26) for n in names))
    rows = [
        ('peak / rms / crest', lambda r: '%.3f / %.3f / %4.1f dB' % (r['lead']['basic']['peak'], r['lead']['basic']['rms'], r['lead']['basic']['crest'])),
        ('DC / clipped', lambda r: '%+.5f / %d' % (r['lead']['basic']['dc'], r['lead']['basic']['clipped'])),
        ('bend start -> end (semis)', lambda r: '%+.2f -> %+.2f  [tgt ~0 -> +2]' % (r['lead']['pitch'].get('bend_start_semis', float('nan')), r['lead']['pitch'].get('bend_end_semis', float('nan')))),
        ('vibrato p-p cents / rate', lambda r: '%.0f c / %.1f Hz  [tgt 20-70 / 5.5]' % (r['lead']['pitch'].get('vib_pp_cents', float('nan')), r['lead']['pitch'].get('vib_rate_hz', float('nan')))),
        ('vibrato mean offset (c)', lambda r: '%+.0f' % r['lead']['pitch'].get('vib_mean_cents', float('nan'))),
        ('slide start -> end (semis)', lambda r: '%+.2f -> %+.2f  [tgt -3 -> 0]' % (r['lead']['pitch'].get('slide_start_semis', float('nan')), r['lead']['pitch'].get('slide_end_semis', float('nan')))),
        ('sustain @1.0 / 1.5 s', lambda r: '%.1f / %.1f dB' % (r['lead']['pitch'].get('sustain_1p0_db', float('nan')), r['lead']['pitch'].get('sustain_1p5_db', float('nan')))),
        ('rolloff / fizz', lambda r: '%+.1f / %+.1f dB' % (r['lead']['tone']['rolloff'], r['lead']['tone']['fizz'])),
        ('time (s)', lambda r: '%.1f' % r['t_lead_s']),
        ('PENALTY lead', lambda r: '%.2f' % r['penalty']['lead']),
    ]
    for lbl, fn in rows:
        print(lbl.ljust(w) + ''.join(fn(r).ljust(26) for r in results))
    for r in results:
        print('  bands lead  %-16s %s' % (r['module'], fmt_bands(r['lead']['bands'])))
        print('  penalties   %-16s %s' % (r['module'], ' '.join('%s=%.2f' % (k, v) for k, v in r['score_lead'].items() if v > 0)))
    print('-' * 110)
    print('CLEAN'.ljust(w) + ''.join(n.ljust(26) for n in names))
    rows = [
        ('peak / rms / crest', lambda r: '%.3f / %.3f / %4.1f dB' % (r['clean']['basic']['peak'], r['clean']['basic']['rms'], r['clean']['basic']['crest'])),
        ('DC / clipped', lambda r: '%+.5f / %d' % (r['clean']['basic']['dc'], r['clean']['basic']['clipped'])),
        ('L/R corr wave / env', lambda r: '%.2f / %.2f  [tgt 0.2..0.9]' % (r['clean']['stereo']['corr_wave'], r['clean']['stereo']['corr_env'])),
        ('S/M dB', lambda r: '%+.1f' % r['clean']['stereo']['side_mid_db']),
        ('ring 1 s after last note', lambda r: '%.1f dB' % r['clean']['ring_after_last_db']),
        ('time (s)', lambda r: '%.1f' % r['t_clean_s']),
        ('PENALTY clean', lambda r: '%.2f' % r['penalty']['clean']),
    ]
    for lbl, fn in rows:
        print(lbl.ljust(w) + ''.join(fn(r).ljust(26) for r in results))
    for r in results:
        print('  bands clean %-16s %s' % (r['module'], fmt_bands(r['clean']['bands'])))
        print('  penalties   %-16s %s' % (r['module'], ' '.join('%s=%.2f' % (k, v) for k, v in r['score_clean'].items() if v > 0)))
    print('-' * 110)
    for r in results:
        if 'extras' in r:
            print('EXTRAS %-16s %s' % (r['module'], ' | '.join('%s: %s' % (k, ('ERR ' + v['error']) if 'error' in v else 'pk %.3f dc %+.4f %.1fs' % (v['peak'], v['dc'], v['t_s'])) for k, v in r['extras'].items())))
        print('CONTRACT %-14s %s' % (r['module'], r['contract']))
        if 'full' in r:
            f = r['full']
            print('FULL %-18s %s' % (r['module'], json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items() if not isinstance(v, dict)})))
            for k in ('pair', 'lead', 'clean'):
                if k in f:
                    print('     %-5s peak %.3f rms %.3f crest %.1f dc %+.5f clipped %d' % (k, f[k]['peak'], f[k]['rms'], f[k]['crest'], f[k]['dc'], f[k]['clipped']))
            if 'pair_stereo' in f:
                print('     pair corr wave %.2f env %.2f  S/M %+.1f dB' % (f['pair_stereo']['corr_wave'], f['pair_stereo']['corr_env'], f['pair_stereo']['side_mid_db']))
    print('=' * 110)
    print('WINNERS (lowest penalty): ' + '  '.join('%s: %s' % (k, min(results, key=lambda r: r['penalty'][k])['module']) for k in ('rhythm', 'lead', 'clean')))


if __name__ == '__main__':
    results = []
    for m in MODULES:
        t = time.time()
        print('judging %s ...' % m, flush=True)
        results.append(judge(m))
        print('  done in %.1fs' % (time.time() - t), flush=True)
    print_report(results)
    with open(os.path.join(HERE, 'judge_guitar_results.json'), 'w') as fh:
        json.dump(results, fh, indent=1, default=float)
    print('results -> tests/judge_guitar_results.json ; spectrograms -> tests/judge_<module>_{rhythm,leadclean}.png')
