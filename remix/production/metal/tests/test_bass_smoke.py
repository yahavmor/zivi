"""Smoke test for metal_bass.render_bass: renders a short riff, checks NaN/DC/clipping, prints
peak/RMS/crest, band energies relative to 250-500 Hz, per-note peaks (compressor evenness),
and saves a spectrogram PNG. Run: python3 tests/test_bass_smoke.py [--full]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
import soundfile as sf
from scipy import signal
import metal_bass as B
import engine as E

SR = E.SR
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
S16 = 60 / 110 / 4


def band_energies(x):
    f, p = signal.welch(x, SR, nperseg=8192)
    ref = p[(f >= 250) & (f < 500)].sum()
    out = {}
    for lo, hi in BANDS:
        e = p[(f >= lo) & (f < hi)].sum()
        out[(lo, hi)] = 10 * np.log10(e / ref + 1e-20)
    return out


def riff():
    """~5 s: 16th chugs on C2, accented, a ringing Eb2, slide C2->F2, G2, Bb2 mutes, octave C3."""
    ev = []
    t = 0.0
    for i in range(8):                       # 2 beats of 16th palm mutes on C2
        ev.append((t + i * S16, 36, S16 * 0.75, 0.95 if i % 4 == 0 else 0.8, None, True))
    t += 8 * S16
    ev.append((t, 39, 0.9, 0.9, None, False))                # ringing Eb2
    t += 1.0
    ev.append((t, 41, 0.5, 0.9, 36, False))                  # slide C2 -> F2
    t += 0.55
    for i in range(4):
        ev.append((t + i * S16, 43, S16 * 0.75, 0.85, None, True))  # G2 mutes
    t += 4 * S16
    ev.append((t, 46, 0.4, 0.7, None, True))                 # softer Bb2 mute
    t += 0.45
    ev.append((t, 48, 0.8, 0.9, None, False))                # C3 ring
    t += 0.9
    ev.append((t, 36, 1.2, 1.0, None, False))                # long low C2 ring-out
    return ev


def full_song_events():
    """~113 s worth of typical metal bass activity to time the renderer."""
    ev = []
    beat = 60 / 110
    roots = [36, 36, 39, 41, 43, 44, 46, 48]
    t = 0.0
    bar = 0
    while t < 112.0:
        root = roots[bar % len(roots)]
        if bar % 4 == 3:
            ev.append((t, root, beat * 3.5, 0.9, root - 2, False))
        else:
            for i in range(16):
                ev.append((t + i * S16, root, S16 * 0.75, 0.8 + 0.15 * (i % 4 == 0), None, i % 8 != 7))
        t += 4 * beat
        bar += 1
    return ev


def spectrogram_png(x, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, tt, S = signal.spectrogram(x, SR, nperseg=2048, noverlap=1536, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].plot(np.arange(len(x)) / SR, x, lw=0.4, color='#444')
    ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(-0.7, 0.7); ax[0].set_ylabel('amp'); ax[0].set_title(title)
    sel = f <= 8000
    im = ax[1].pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 90, vmax=Sdb.max())
    ax[1].set_yscale('symlog', linthresh=200); ax[1].set_ylim(30, 8000)
    ax[1].set_yticks([50, 65, 100, 150, 250, 500, 1000, 2000, 4000, 8000])
    ax[1].set_yticklabels(['50', '65', '100', '150', '250', '500', '1k', '2k', '4k', '8k'])
    ax[1].set_ylabel('Hz'); ax[1].set_xlabel('s')
    fig.colorbar(im, ax=ax[1], label='dB')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def report(name, x, events=None):
    assert not np.isnan(x).any(), 'NaN'
    assert not np.isinf(x).any(), 'inf'
    peak = np.abs(x).max(); rms = np.sqrt(np.mean(x ** 2)); dc = x.mean()
    sub20 = E.lp(x, 20, order=2); dc_lf = np.sqrt(np.mean(sub20 ** 2))
    print(f'[{name}] len={len(x)/SR:.2f}s peak={peak:.3f} ({20*np.log10(peak+1e-12):.1f} dBFS) rms={rms:.3f} '
          f'({20*np.log10(rms+1e-12):.1f} dBFS) crest={20*np.log10(peak/(rms+1e-12)):.1f} dB '
          f'DC={dc:+.5f} (<20Hz rms {20*np.log10(dc_lf+1e-12):.1f} dBFS) clipped={(np.abs(x)>0.99).sum()}')
    be = band_energies(x)
    print('  band energy rel 250-500:  ' + '  '.join(f'{lo}-{hi}:{v:+5.1f}' for (lo, hi), v in be.items()))
    if events is not None:
        pk = []
        for (t, m, d, v, sf_, mu) in events:
            i0 = int(t * SR); i1 = int((t + min(d, 0.25)) * SR)
            pk.append(np.abs(x[i0:i1]).max())
        pk = np.array(pk)
        print(f'  per-note peak: min={pk.min():.3f} max={pk.max():.3f} spread={20*np.log10(pk.max()/pk.min()):.1f} dB')
    return be


if __name__ == '__main__':
    ev = riff()
    t0 = time.time()
    di = B.render_di(ev)
    y = B.render_bass(ev)
    dt = time.time() - t0
    print(f'render time (riff, {len(ev)} notes, {len(y)/SR:.1f}s): {dt:.2f}s')
    report('DI', di / (np.abs(di).max() + 1e-9) * 0.5, ev)
    be = report('BASS', y, ev)
    assert 0.3 <= np.abs(y).max() <= 0.6, 'peak out of 0.3-0.6'
    assert abs(y.mean()) < 1e-3, 'DC offset'
    out = os.path.join(HERE, 'bass_smoke.wav'); sf.write(out, y.astype(np.float32), SR, subtype='FLOAT')
    out_di = os.path.join(HERE, 'bass_smoke_di.wav'); sf.write(out_di, (di / (np.abs(di).max() + 1e-9) * 0.5).astype(np.float32), SR, subtype='FLOAT')
    png = os.path.join(HERE, 'bass_smoke_spec.png'); spectrogram_png(y, png, 'metal_bass smoke riff (final)')
    png_di = os.path.join(HERE, 'bass_smoke_di_spec.png'); spectrogram_png(di / (np.abs(di).max() + 1e-9) * 0.5, png_di, 'metal_bass DI')
    print('wrote', out, png, png_di)
    # single-note spectra: open C2 vs muted C2
    for mute in (False, True):
        y1 = B.render_bass([(0.0, 36, 1.0, 0.9, None, mute)], total_len=1.4)
        report('C2 ' + ('mute' if mute else 'open'), y1)
    if '--full' in sys.argv:
        evf = full_song_events()
        t0 = time.time(); yf = B.render_bass(evf, total_len=113.0); dt = time.time() - t0
        print(f'FULL render: {len(evf)} notes, {len(yf)/SR:.1f}s audio in {dt:.1f}s')
        report('FULL', yf)
