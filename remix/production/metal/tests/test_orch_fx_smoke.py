"""Smoke test for metal_orch_fx.py: renders every one-shot + an orchestra chord, checks NaN/DC/clipping,
prints peak / RMS / band energies (dB relative to 250-500 Hz) and saves spectrogram montages.
Run: python3 tests/test_orch_fx_smoke.py            (add --full to also time a 113 s orchestra render)"""
import os, sys, time
import numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, '..'))
import engine as E
import metal_orch_fx as O
SR = E.SR
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def band_db(x):
    m = E.mono(x); f, P = signal.welch(m, SR, nperseg=8192)
    e = []
    for lo, hi in BANDS:
        sel = (f >= lo) & (f < hi); e.append(P[sel].sum() + 1e-20)
    ref = e[3]
    return [10 * np.log10(v / ref) for v in e]


def check(name, x, expect_stereo=None, pk_lo=0.25, pk_hi=0.62):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim in (1, 2), (name, x.shape)
    if expect_stereo is True: assert x.ndim == 2 and x.shape[1] == 2, (name, x.shape)
    if expect_stereo is False: assert x.ndim == 1, (name, x.shape)
    assert np.isfinite(x).all(), (name, 'NaN/inf')
    pk = float(np.abs(x).max()); rms = float(np.sqrt((x ** 2).mean())); dc = float(E.mono(x).mean())
    assert pk_lo <= pk <= pk_hi, (name, 'peak', pk)
    assert abs(dc) < 0.004, (name, 'DC', dc)
    # block DC: no 100 ms window with a mean above 2% of peak (catches sub-sonic/offset problems)
    m = E.mono(x); w = int(0.1 * SR); nb = len(m) // w
    if nb > 0:
        bm = np.abs(m[:nb * w].reshape(nb, w).mean(1)).max()
        assert bm < 0.08 * pk, (name, 'block DC', bm, pk)
    b = band_db(x)
    print("  %-20s %-14s len %6.2fs peak %.3f rms %.4f dc %+.5f | " % (name, str(x.shape), len(x) / SR, pk, rms, dc)
          + ' '.join('%+6.1f' % v for v in b))
    return x


def montage(items, gap=0.4):
    out = []; marks = []
    for name, x in items:
        marks.append((name, sum(len(o) for o in out) / SR)); out.append(E.stereo(x)); out.append(np.zeros((int(gap * SR), 2)))
    return np.vstack(out), marks


def spectrogram_png(x, path, title, marks=(), fmax=16000):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    m = E.mono(x); f, t, S = signal.spectrogram(m, SR, nperseg=2048, noverlap=1536, window='hann')
    S = 10 * np.log10(S + 1e-12); S = np.clip(S, S.max() - 90, S.max())
    fig, ax = plt.subplots(figsize=(16, 6))
    sel = f <= fmax
    im = ax.pcolormesh(t, f[sel], S[sel], shading='auto', cmap='magma')
    ax.set_yscale('symlog', linthresh=200); ax.set_ylim(20, fmax); ax.set_xlabel('s'); ax.set_ylabel('Hz')
    for name, tt in marks: ax.axvline(tt, color='w', lw=0.5, alpha=0.6); ax.text(tt + 0.05, fmax * 0.75, name, color='w', fontsize=7, rotation=90)
    ax.set_title(title); fig.colorbar(im, ax=ax, label='dB'); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return path


def main():
    t0 = time.time()
    print("band dB rel 250-500:", ' '.join('%d-%d' % b for b in BANDS))
    shots = []
    shots.append(('impact', check('impact', O.impact(1.0), True)))
    check('impact v0.6', O.impact(0.6, seed=1), True, pk_lo=0.15, pk_hi=0.45)
    shots.append(('sub_drop', check('sub_drop', O.sub_drop(1.5, 70, 25), False)))
    shots.append(('riser', check('riser', O.riser(2 * O.BAR, 200, 6000), True)))
    shots.append(('downlifter', check('downlifter', O.downlifter(2.0), True)))
    shots.append(('metal_hit', check('metal_hit', O.metal_hit(0.7, 420), False, pk_lo=0.2, pk_hi=0.45)))
    check('metal_hit 600', O.metal_hit(1.0, 600, seed=2), False)
    shots.append(('clank', check('industrial_clank', O.industrial_clank(1), False)))
    check('industrial_clank2', O.industrial_clank(2), False)
    shots.append(('rev_cymbal', check('reverse_cymbal', O.reverse_cymbal(1.5), True)))
    shots.append(('drone', check('drone', O.drone(36, 6.0), True, pk_lo=0.3, pk_hi=0.5)))
    shots.append(('noise_bed', check('noise_bed', O.noise_bed(6.0), True, pk_lo=0.25, pk_hi=0.4)))
    shots.append(('braam', check('braam', O.braam(36, 2 * O.BAR), True)))
    shots.append(('timpani', check('timpani_roll', O.timpani_roll(36, 2.0, 0.9), True)))
    shots.append(('orch_hit', check('orchestra_hit', O.orchestra_hit(48, 1.0), True)))
    # orchestra: a Cm chord on strings, choir and brass through the hall
    BEAT = O.BEAT
    chord = [(0.0, 48, 4 * BEAT, 0.7), (0.0, 51, 4 * BEAT, 0.7), (0.0, 55, 4 * BEAT, 0.7), (0.0, 60, 4 * BEAT, 0.7), (4 * BEAT, 46, 4 * BEAT, 0.8), (4 * BEAT, 53, 4 * BEAT, 0.8)]
    for name, prog in [('strings', 48), ('choir', 52), ('brass', 61)]:
        shots.append(('orch_' + name, check('orch p%d' % prog, O.orch(chord, prog), True)))
    x = O.orch(chord, 48, total_len=4.0); assert x.shape == (O._n(4.0), 2), x.shape
    x = O.orch([], 48, total_len=2.0); assert x.shape == (O._n(2.0), 2) and np.abs(x).max() == 0
    print("  orch total_len / empty events OK")
    mont, marks = montage(shots)
    E.write(os.path.join(HERE, 'orchfx_oneshots.wav'), mont)
    p = spectrogram_png(mont, os.path.join(HERE, 'orchfx_oneshots_spec.png'), 'metal_orch_fx one-shots + orch chords', marks)
    print("spectrogram:", p)
    # second detailed view: the low end of impact / sub_drop / drone / braam
    low = [s for s in shots if s[0] in ('impact', 'sub_drop', 'drone', 'braam')]
    mont2, marks2 = montage(low)
    p2 = spectrogram_png(mont2, os.path.join(HERE, 'orchfx_lowend_spec.png'), 'impact / sub_drop / drone / braam (<2 kHz detail)', marks2, fmax=2000)
    print("spectrogram:", p2)
    if '--full' in sys.argv:
        rng = np.random.default_rng(3); evs = []
        for bar in range(52):
            t = bar * O.BAR; root = int(rng.choice([48, 51, 53, 55]))
            for n in (root, root + 3, root + 7, root + 12): evs.append((t, n, O.BAR * 0.95, 0.75))
        t1 = time.time(); y = O.orch(evs, 48, total_len=52 * O.BAR + 6.0); dt = time.time() - t1
        check('orch 113s strings', y, True)
        print("  full-song orchestra render: %.1fs (%d events)" % (dt, len(evs)))
        t1 = time.time(); y = O.orch(evs, 48, total_len=52 * O.BAR + 6.0); print("  cached re-render: %.1fs" % (time.time() - t1))
    print("ORCH_FX SMOKE OK in %.0fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
