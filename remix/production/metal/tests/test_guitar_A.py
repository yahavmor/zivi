"""Smoke test for metal_guitar_A: renders short rhythm / lead / clean / extras, checks NaN / DC / clipping,
prints peak / RMS / crest, band energies relative to 250-500 Hz, chug decay time, chord sustain,
pitch tracking of bends / vibrato / slides, DI tuning, and saves spectrogram PNGs.
Run: python3 tests/test_guitar_A.py [--full]   (--full also times a 113 s song-sized render)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
import soundfile as sf
from scipy import signal
import engine as E
import metal_guitar_A as G

SR = E.SR
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
BEAT = 60 / 110; S16 = BEAT / 4
OUT = HERE


def band_energies(x):
    f, p = signal.welch(x, SR, nperseg=8192)
    ref = p[(f >= 250) & (f < 500)].sum()
    return {b: 10 * np.log10(p[(f >= b[0]) & (f < b[1])].sum() / ref + 1e-20) for b in BANDS}


def fmt_bands(be):
    return ' '.join('%d-%d:%+.1f' % (lo if lo < 1000 else lo // 1000, hi if hi < 1000 else hi // 1000, v) if False else
                    '%s:%+5.1f' % (('%g' % (lo / 1000) + 'k' if lo >= 1000 else str(lo)) + '-' + ('%g' % (hi / 1000) + 'k' if hi >= 1000 else str(hi)), v)
                    for (lo, hi), v in be.items())


def stats(name, x):
    x = np.asarray(x)
    assert np.isfinite(x).all(), name + ': NaN/inf'
    m = E.mono(x) if x.ndim == 2 else x
    pk = np.abs(x).max(); rms = np.sqrt(np.mean(m ** 2)); dc = m.mean()
    clip = int((np.abs(x) >= 0.99).sum())
    print('  %-22s shape %-13s peak %.3f rms %.3f crest %4.1f dB dc %+.5f clip %d' % (name, x.shape, pk, rms, 20 * np.log10(pk / (rms + 1e-9)), dc, clip))
    assert abs(dc) < 0.01, name + ': DC offset'
    assert pk <= 0.85, name + ': peak too high'
    assert clip == 0, name + ': clipping'
    return m


def envelope_db(x, win=0.005):
    k = int(win * SR)
    e = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, 'same'))
    return 20 * np.log10(e + 1e-9)


def decay_times(x, levels=(-20, -40)):
    """time (s) after the peak at which the RMS envelope has dropped by `levels` dB."""
    e = envelope_db(x); i0 = int(np.argmax(e)); pk = e[i0]
    res = {}
    for L in levels:
        idx = np.where(e[i0:] <= pk + L)[0]
        res[L] = (idx[0] / SR) if len(idx) else None
    return res, pk


def pitch_track(x, hop=0.02, win=0.05, fmin=60, fmax=1500):
    """autocorrelation pitch per frame -> (times, f0)."""
    n = len(x); W = int(win * SR); H = int(hop * SR); ts = []; fs = []
    lag_min = int(SR / fmax); lag_max = int(SR / fmin)
    for s in range(0, n - W, H):
        seg = x[s:s + W] * np.hanning(W)
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        ac = signal.fftconvolve(seg, seg[::-1])[W - 1:]
        ac = ac / (ac[0] + 1e-12)
        k = lag_min + int(np.argmax(ac[lag_min:lag_max]))
        if ac[k] < 0.3:
            continue
        # parabolic interpolation
        if 1 <= k < len(ac) - 1:
            a, b, c = ac[k - 1], ac[k], ac[k + 1]; d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12); k = k + d
        ts.append((s + W / 2) / SR); fs.append(SR / k)
    return np.array(ts), np.array(fs)


def spectrogram_png(x, path, title, fmax=10000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, tt, S = signal.spectrogram(x, SR, nperseg=2048, noverlap=1536, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].plot(np.arange(len(x)) / SR, x, lw=0.4, color='#444')
    ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(-0.7, 0.7); ax[0].set_ylabel('amp'); ax[0].set_title(title)
    sel = f <= fmax
    ax[1].pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 80, vmax=Sdb.max())
    ax[1].set_ylabel('Hz'); ax[1].set_xlabel('s')
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    print('  spectrogram ->', path)


def riff_events():
    """~6.5 s: 8 x 16th C2 chugs, C5 ring 1.8 s, Eb/F chords, G2 chugs, Bb2 ring."""
    ev = []; t = 0.0
    for i in range(8):
        ev.append((t + i * S16, 36, S16 * 0.8, 1.0 if i % 4 == 0 else 0.85, True))
    t += 8 * S16
    ev.append((t, 36, 1.8, 1.0, False)); t += 1.9
    ev.append((t, 39, 0.8, 0.95, False)); t += 0.85
    ev.append((t, 41, 0.8, 0.95, False)); t += 0.85
    for i in range(4):
        ev.append((t + i * S16, 43, S16 * 0.8, 0.9, True))
    t += 4 * S16
    ev.append((t, 46, 1.2, 0.9, False))
    return ev


def lead_events():
    """G4 vib, Bb4 bend +2, slide G4->C5, long D5 with vibrato, fast run."""
    ev = [(0.0, 67, 0.9, 0.9, 0.0, 0.35, None),
          (1.0, 70, 1.2, 0.95, 2.0, 0.4, None),
          (2.4, 72, 0.9, 0.9, 0.0, 0.3, 67),
          (3.5, 74, 2.5, 1.0, 0.0, 0.45, None)]
    t = 6.2
    for i, m in enumerate([72, 70, 67, 65, 63, 60, 58, 55]):
        ev.append((t + i * 0.14, m, 0.14, 0.85, 0.0, 0.0, None))
    ev.append((t + 8 * 0.14, 48, 1.6, 1.0, 0.0, 0.3, None))
    return ev


def clean_events():
    ev = []; t = 0.0
    for rep in range(2):
        for i, m in enumerate([48, 55, 60, 63, 67, 63, 60, 55]):
            ev.append((t + i * S16 * 2, m, S16 * 2 * 1.6, 0.65 + 0.1 * (i % 2)))
        t += 16 * S16
    return ev


def song_sized_events():
    gtr = []; lead = []; clean = []
    roots = [36, 36, 39, 41, 43, 44, 46, 48]; t = 0.0; bar = 0
    while t < 112.0:
        root = roots[bar % len(roots)]
        if bar % 4 == 3:
            gtr.append((t, root, BEAT * 3.5, 0.95, False))
        else:
            for i in range(16):
                gtr.append((t + i * S16, root, S16 * 0.8, 0.85 + 0.15 * (i % 4 == 0), i % 8 != 7))
        if 36 < bar < 52:
            for i in range(4):
                lead.append((t + i * BEAT, 60 + (bar * 3 + i * 2) % 14, BEAT * 0.9, 0.9, 2.0 if i == 3 else 0.0, 0.3, None if i else 58))
        if bar < 8 or 28 <= bar < 36:
            for i in range(8):
                clean.append((t + i * S16 * 2, [48, 55, 60, 63][i % 4] + 12 * (i // 4), S16 * 3, 0.6))
        t += 4 * BEAT; bar += 1
    return gtr, lead, clean


def timing():
    print('== song-sized timing (113 s)')
    gtr, lead, clean = song_sized_events()
    print('  events: rhythm %d lead %d clean %d' % (len(gtr), len(lead), len(clean)))
    t0 = time.time(); yp = G.render_rhythm_pair(gtr, total_len=113.0); t1 = time.time() - t0
    stats('song rhythm pair', yp); print('  rhythm pair %.1fs' % t1)
    t0 = time.time(); yl = G.render_lead(lead, total_len=113.0); t2 = time.time() - t0
    stats('song lead', yl); print('  lead %.1fs' % t2)
    t0 = time.time(); yc = G.render_clean(clean, total_len=113.0); t3 = time.time() - t0
    stats('song clean', yc); print('  clean %.1fs' % t3)
    print('  TOTAL %.1fs' % (t1 + t2 + t3))


def main():
    full = '--full' in sys.argv
    if '--timing-only' in sys.argv:
        timing(); return
    print('== rhythm: DI tuning check (C2 string)')
    s = G.string_note(E.midi2hz(36), int(1.0 * SR), 0.9, False, seed=3, click=False, settle=False)
    f, p = signal.welch(s, SR, nperseg=65536)
    k = np.argmax(p[(f > 40) & (f < 100)]); fpk = f[(f > 40) & (f < 100)][k]
    sel = (f > 40) & (f < 100); fi = np.where(sel)[0][k]
    # parabolic refine of the fundamental peak
    y0, y1, y2 = np.log(p[fi - 1]), np.log(p[fi]), np.log(p[fi + 1]); d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    fpk = f[fi] + d * (f[1] - f[0])
    print('  C2 string fundamental: %.3f Hz (target 65.406, error %+.1f cents)' % (fpk, 1200 * np.log2(fpk / 65.406)))
    for midi in (41, 48, 60, 72):
        s2 = G.string_note(E.midi2hz(midi), int(0.6 * SR), 0.9, False, seed=3, click=False, settle=False)
        ts, fz = pitch_track(s2[int(0.1 * SR):], fmin=60, fmax=1500)
        print('  midi %d: tracked %.2f Hz (target %.2f, %+.1f cents)' % (midi, np.median(fz), E.midi2hz(midi), 1200 * np.log2(np.median(fz) / E.midi2hz(midi))))

    print('== rhythm riff (mono, seed 1)')
    t0 = time.time(); ev = riff_events()
    di = G.render_rhythm_di(ev, seed=1)
    y = G.render_rhythm(ev, seed=1)
    print('  render time %.1fs' % (time.time() - t0))
    stats('rhythm DI', di); m = stats('rhythm amp', y)
    print('  DI  bands  ', fmt_bands(band_energies(di)))
    print('  AMP bands  ', fmt_bands(band_energies(m)))
    chug = m[:int(8 * S16 * SR)]; chord = m[int(8 * S16 * SR):int((8 * S16 + 1.8) * SR)]
    print('  CHUG bands ', fmt_bands(band_energies(chug)))
    print('  CHORD bands', fmt_bands(band_energies(chord)))
    # single chug decay
    one = G.render_rhythm([(0.1, 36, S16 * 0.8, 1.0, True)], seed=1, total_len=1.0)
    dt, pk = decay_times(one)
    print('  single chug: peak %.3f, -20 dB after %s ms, -40 dB after %s ms' % (np.abs(one).max(), None if dt[-20] is None else '%.0f' % (dt[-20] * 1000), None if dt[-40] is None else '%.0f' % (dt[-40] * 1000)))
    e_one = envelope_db(one); i_pk = int(np.argmax(e_one))
    print('  chug envelope rel. peak @ +50ms %+.1f  +100ms %+.1f  +150ms %+.1f  +200ms %+.1f  +300ms %+.1f dB' % tuple(e_one[i_pk + int(x * SR)] - e_one[i_pk] for x in (0.05, 0.1, 0.15, 0.2, 0.3)))
    # chord sustain
    ch = G.render_rhythm([(0.1, 36, 3.0, 1.0, False)], seed=1, total_len=3.6)
    e_ch = envelope_db(ch, 0.02); i_pk = int(np.argmax(e_ch))
    print('  C5 chord envelope rel. peak @ 0.5s %+.1f  1.0s %+.1f  1.5s %+.1f  2.0s %+.1f  2.5s %+.1f dB' % tuple(e_ch[i_pk + int(x * SR)] - e_ch[i_pk] for x in (0.5, 1.0, 1.5, 2.0, 2.5)))
    sf.write(os.path.join(OUT, 'gtrA_rhythm.wav'), np.stack([di, m], 1).astype(np.float32), SR)
    spectrogram_png(m, os.path.join(OUT, 'gtrA_rhythm_spec.png'), 'metal_guitar_A rhythm (amp+cab) — chugs C2, C5 ring, Eb5, F5, G2 chugs, Bb5')
    spectrogram_png(di, os.path.join(OUT, 'gtrA_rhythm_di_spec.png'), 'metal_guitar_A rhythm DI')

    print('== rhythm pair (stereo)')
    t0 = time.time(); yp = G.render_rhythm_pair(ev); print('  render time %.1fs' % (time.time() - t0))
    stats('rhythm pair', yp)
    L, R = yp[:, 0], yp[:, 1]
    print('  L/R correlation %.3f   (want clearly < 0.9 for width)' % (np.corrcoef(L, R)[0, 1]))
    print('  L bands ', fmt_bands(band_energies(L)))
    sf.write(os.path.join(OUT, 'gtrA_rhythm_pair.wav'), yp.astype(np.float32), SR)

    print('== lead')
    t0 = time.time(); lev = lead_events()
    ldi = G.render_lead_di(lev); yl = G.render_lead(lev); print('  render time %.1fs' % (time.time() - t0))
    stats('lead DI', ldi); ml = stats('lead amp', yl)
    print('  LEAD bands ', fmt_bands(band_energies(ml)))
    ts, fz = pitch_track(ldi, fmin=80, fmax=1200)
    def seg(a, b):
        sel = (ts >= a) & (ts < b); return fz[sel]
    g4 = E.midi2hz(67); bb4 = E.midi2hz(70); c5 = E.midi2hz(72); d5 = E.midi2hz(74)
    s1 = seg(0.4, 0.9); print('  G4 vibrato: pitch range %.1f..%.1f Hz = %+.2f..%+.2f semis (depth set 0.35)' % (s1.min(), s1.max(), 12 * np.log2(s1.min() / g4), 12 * np.log2(s1.max() / g4)))
    s2a = seg(1.0, 1.1); s2b = seg(1.6, 2.1)
    print('  Bb4 bend: start %+.2f semis, end %+.2f semis (set +2.0, vib 0.4 on top)' % (12 * np.log2(np.median(s2a) / bb4), 12 * np.log2(np.median(s2b) / bb4)))
    s3a = seg(2.4, 2.45); s3b = seg(2.7, 3.2)
    print('  slide G4->C5: start %+.2f semis from C5 (expect ~-5), settled %+.2f' % (12 * np.log2(np.median(s3a) / c5) if len(s3a) else np.nan, 12 * np.log2(np.median(s3b) / c5)))
    e_l = envelope_db(ml, 0.02); i0 = int(3.5 * SR); i_pk = i0 + int(np.argmax(e_l[i0:i0 + int(0.3 * SR)]))
    print('  D5 long note envelope rel. peak @ 0.5s %+.1f 1.0s %+.1f 1.5s %+.1f 2.0s %+.1f dB' % tuple(e_l[i_pk + int(x * SR)] - e_l[i_pk] for x in (0.5, 1.0, 1.5, 2.0)))
    sf.write(os.path.join(OUT, 'gtrA_lead.wav'), np.stack([ldi, ml], 1).astype(np.float32), SR)
    spectrogram_png(ml, os.path.join(OUT, 'gtrA_lead_spec.png'), 'metal_guitar_A lead — G4 vib, Bb4 bend+2, slide to C5, D5 long, run, C3')

    print('== clean')
    t0 = time.time(); cev = clean_events(); yc = G.render_clean(cev); print('  render time %.1fs' % (time.time() - t0))
    mc = stats('clean', yc)
    print('  CLEAN bands', fmt_bands(band_energies(mc)))
    print('  L/R correlation %.3f' % np.corrcoef(yc[:, 0], yc[:, 1])[0, 1])
    sf.write(os.path.join(OUT, 'gtrA_clean.wav'), yc.astype(np.float32), SR)
    spectrogram_png(mc, os.path.join(OUT, 'gtrA_clean_spec.png'), 'metal_guitar_A clean arpeggio (chorus+reverb)')

    print('== extras')
    ex = []
    for name, fn in [('feedback_swell', lambda: G.feedback_swell(48, 3.0)), ('pick_scrape', lambda: G.pick_scrape(1.2)),
                     ('pinch_harmonic', lambda: G.pinch_harmonic(43)), ('dive_bomb', lambda: G.dive_bomb(55, 2.0))]:
        t0 = time.time(); x = fn(); stats(name + ' (%.1fs)' % (time.time() - t0), x); ex.append(x); ex.append(np.zeros(int(0.3 * SR)))
    exa = np.concatenate(ex)
    sf.write(os.path.join(OUT, 'gtrA_extras.wav'), exa.astype(np.float32), SR)
    spectrogram_png(exa, os.path.join(OUT, 'gtrA_extras_spec.png'), 'metal_guitar_A extras — feedback swell, pick scrape, pinch harmonic, dive bomb')

    if full:
        timing()
    print('OK')


if __name__ == '__main__':
    main()
