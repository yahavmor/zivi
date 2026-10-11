"""Smoke test for metal_vocal.variant: renders every kind for 'meham3' and 'p2', checks NaN/DC/clipping,
lengths and peaks, prints RMS and band energies (relative to 250-500 Hz), runs structural checks per kind
(reverse-swell crescendo, stutter grid, double L/R decorrelation, hall decay, demon/slowmo pitch drop,
megaphone/radio band limits), times a song-sized workload, and saves spectrogram PNGs + demo WAVs in tests/.
Run: python3 tests/test_metal_vocal.py"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
from scipy import signal
import engine as E
import metal_vocal as MV

SR = E.SR
BEAT = MV.BEAT
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
NAMES = ('meham3', 'p2')
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print('  FAIL:', msg)


def band_energies(x):
    x = E.mono(x)
    f, p = signal.welch(x, SR, nperseg=8192)
    ref = p[(f >= 250) & (f < 500)].sum()
    return [10 * np.log10(p[(f >= lo) & (f < hi)].sum() / ref + 1e-20) for lo, hi in BANDS]


def rms_db(x):
    return 20 * np.log10(np.sqrt((np.asarray(x) ** 2).mean()) + 1e-12)


def f0_median(x, fmin=50, fmax=500):
    """median YIN-ish f0 via vocal.yin_f0 (voiced frames only)."""
    import vocal as V
    t, f, v = V.yin_f0(E.mono(x), fmin=fmin, fmax=fmax)
    return float(np.median(f[v])) if v.sum() > 3 else float('nan')


def onset_times(x, thr_db=-20):
    e = signal.fftconvolve(np.abs(E.mono(x)), np.ones(48) / 48, 'same')
    e = e / e.max()
    on = (e > E.db(thr_db)).astype(int)
    d = np.diff(np.concatenate([[0], on]))
    return np.where(d > 0)[0] / SR


def spectrogram_png(x, path, title, labels=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = E.mono(x)
    f, tt, S = signal.spectrogram(m, SR, nperseg=2048, noverlap=1536, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].plot(np.arange(len(x)) / SR, x[:, 0], lw=0.3, color='#2a6fdb', label='L')
    ax[0].plot(np.arange(len(x)) / SR, -x[:, 1], lw=0.3, color='#db2a2a', label='R (flipped)')
    ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(-0.6, 0.6); ax[0].set_ylabel('amp'); ax[0].set_title(title)
    ax[0].legend(loc='upper right', fontsize=7)
    sel = f <= 10000
    ax[1].pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 90, vmax=Sdb.max())
    ax[1].set_yscale('symlog', linthresh=200); ax[1].set_ylim(40, 10000); ax[1].set_ylabel('Hz'); ax[1].set_xlabel('s')
    ax[1].set_yticks([50, 100, 200, 500, 1000, 2000, 4000, 8000]); ax[1].set_yticklabels(['50', '100', '200', '500', '1k', '2k', '4k', '8k'])
    if labels:
        for t0, lab in labels:
            ax[1].axvline(t0, color='w', lw=0.5, alpha=0.6); ax[1].text(t0 + 0.03, 8500, lab, color='w', fontsize=8)
            ax[0].axvline(t0, color='k', lw=0.5, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    print('bands:', ' '.join('%4d-%-5d' % b for b in BANDS))
    t_cold = time.time()
    rendered = {}
    for nm in NAMES:
        w = MV.word(nm); n = len(w)
        print('\n=== %s  (word %.3fs, raw peak %.3f) ===' % (nm, n / SR, np.abs(w).max()))
        for k in MV.KINDS:
            t0 = time.time()
            y = MV.variant(nm, k)
            dt = time.time() - t0
            lead = None
            if isinstance(y, tuple):
                y, lead = y
            rendered[(nm, k)] = (y, lead)
            check(y.ndim == 2 and y.shape[1] == 2, '%s/%s not stereo' % (nm, k))
            check(np.isfinite(y).all(), '%s/%s has NaN/inf' % (nm, k))
            dc = np.abs(y.mean(0)).max()
            check(dc < 2e-3, '%s/%s DC %.4f' % (nm, k, dc))
            pk = np.abs(y).max()
            check(abs(pk - 0.5) < 0.01, '%s/%s peak %.3f (want 0.5)' % (nm, k, pk))
            check(pk <= 0.99, '%s/%s clipping' % (nm, k))
            be = band_energies(y)
            print('%-14s len %.3fs peak %.3f rms %6.1f dB  %5.2fs | %s' % (k, len(y) / SR, pk, rms_db(E.mono(y)), dt,
                  ' '.join('%+6.1f' % b for b in be)))
            # ---- length expectations
            if k in ('raw', 'clean', 'demon', 'shout_stack', 'megaphone', 'radio', 'tele'):
                check(len(y) == n, '%s/%s length %d != word %d' % (nm, k, len(y), n))
            elif k == 'hall':
                check(len(y) == n + int((3.5 + 0.04 + 0.1) * SR), '%s/hall length %d' % (nm, len(y)))
            elif k == 'double':
                check(n + int(0.016 * SR) <= len(y) <= n + int(0.02 * SR), '%s/double length %d' % (nm, len(y)))
            elif k == 'stutter':
                want = int(round(3 * 0.25 * BEAT * SR)) + n          # last repeat at the exact grid position + word
                check(len(y) == want, '%s/stutter length %d != %d' % (nm, len(y), want))
            elif k == 'reverse_swell':
                check(abs(len(y) / SR - lead) < 1e-6, '%s/reverse_swell lead_in %.4f != len %.4f' % (nm, lead, len(y) / SR))
            elif k == 'glitch':
                check(0.7 * n <= len(y) <= 2.2 * n, '%s/glitch length %d' % (nm, len(y)))
            elif k == 'slowmo':
                check(abs(len(y) - 2 * n) <= 2, '%s/slowmo length %d != %d' % (nm, len(y), 2 * n))
    t_cold = time.time() - t_cold

    # ---------------------------------------------------------------- structural checks (meham3)
    print('\n=== structural checks ===')
    nm = 'meham3'; w = MV.word(nm); n = len(w)
    clean, _ = rendered[(nm, 'clean')]
    # clean: HPF 100 -> 20-60 band far down; presence lift vs raw
    raw, _ = rendered[(nm, 'raw')]
    be_c, be_r = band_energies(clean), band_energies(raw)
    print('clean vs raw band delta: ' + ' '.join('%+5.1f' % (c - r) for c, r in zip(be_c, be_r)))
    # the stored words were already high-passed at 75 Hz, so check absolutes: nothing below 120 Hz but the f0
    check(be_c[0] < -40 and be_c[1] <= be_r[1] + 1.0, 'clean: low end not controlled (20-60 %.1f, 60-120 %.1f)' % (be_c[0], be_c[1]))
    check(be_c[6] > be_r[6] + 1.5, 'clean: 2-4k presence lift missing (delta %.1f)' % (be_c[6] - be_r[6]))
    # hall: tail decays about 60 dB over 3.5 s (T60); measure RMS 0.3 s after word end vs 2.8 s after
    hall, _ = rendered[(nm, 'hall')]
    a = rms_db(hall[n + int(0.3 * SR): n + int(0.5 * SR)]); b = rms_db(hall[n + int(2.8 * SR): n + int(3.0 * SR)])
    print('hall: tail RMS +0.3s %.1f dB, +2.8s %.1f dB -> slope %.1f dB/s (T60 3.5 s = -17 dB/s)' % (a, b, (b - a) / 2.5))
    check(-24 < (b - a) / 2.5 < -11, 'hall tail decay slope %.1f dB/s' % ((b - a) / 2.5))
    check(rms_db(hall[n + int(0.3 * SR): n + int(0.5 * SR)]) > -45, 'hall tail too quiet')
    hall0 = MV.variant(nm, 'hall', duck_db=0)
    side = lambda y: rms_db((y[:n, 0] - y[:n, 1]))          # the dry is mono: L-R is wet only
    print('hall: wet (L-R) during the word %.1f dB ducked vs %.1f dB un-ducked; tail +0.3 s %.1f vs %.1f dB' % (
        side(hall), side(hall0), a, rms_db(hall0[n + int(0.3 * SR): n + int(0.5 * SR)])))
    check(side(hall0) - side(hall) > 6, 'hall: wet is not ducked under the voice')
    # double: L/R decorrelated but envelopes aligned
    dbl, _ = rendered[(nm, 'double')]
    cc = np.corrcoef(dbl[:, 0], dbl[:, 1])[0, 1]
    eL = signal.fftconvolve(np.abs(dbl[:, 0]), np.ones(480) / 480, 'same'); eR = signal.fftconvolve(np.abs(dbl[:, 1]), np.ones(480) / 480, 'same')
    lag = (np.argmax(signal.correlate(eL - eL.mean(), eR - eR.mean(), 'full')) - (len(eL) - 1)) / SR * 1000
    print('double: L/R sample correlation %.3f, envelope lag %.1f ms, L rms %.1f dB R rms %.1f dB' % (cc, lag, rms_db(dbl[:, 0]), rms_db(dbl[:, 1])))
    check(cc < 0.9, 'double: L/R too correlated (%.3f)' % cc)
    check(abs(lag) < 15, 'double: L/R envelope lag %.1f ms' % lag)
    check(abs(rms_db(dbl[:, 0]) - rms_db(dbl[:, 1])) < 2, 'double: L/R level mismatch')
    # demon / slowmo / shout stack: pitch
    f_clean = f0_median(clean, 110, 480)
    f_demon = f0_median(rendered[(nm, 'demon')][0], 50, 300)
    f_slow = f0_median(rendered[(nm, 'slowmo')][0], 50, 300)
    print('f0 median: clean %.0f Hz, demon %.0f Hz (want ~%.0f), slowmo %.0f Hz (want ~%.0f)' % (f_clean, f_demon, f_clean / 2, f_slow, f_clean / 2))
    check(abs(np.log2(f_demon / (f_clean / 2))) < 0.15, 'demon f0 not an octave down')
    check(abs(np.log2(f_slow / (f_clean / 2))) < 0.15, 'slowmo f0 not an octave down')
    be_d = band_energies(rendered[(nm, 'demon')][0])
    check(be_d[7] < be_c[7] - 10, 'demon: 4-8k not lowpassed (%.1f vs clean %.1f)' % (be_d[7], be_c[7]))
    # shout stack: octave-up and octave-down energy visible: 60-120 band up vs clean, 4-8k not lost
    be_s = band_energies(rendered[(nm, 'shout_stack')][0])
    print('shout_stack vs clean band delta: ' + ' '.join('%+5.1f' % (s - c) for s, c in zip(be_s, be_c)))
    check(be_s[1] > be_c[1] + 2, 'shout_stack: no octave-down energy at 60-120 Hz')
    # megaphone / radio / tele: band limited
    for k, lo_band_max, hi_band_max in (('megaphone', -25, -12), ('radio', -25, -10), ('tele', -25, -10)):
        be = band_energies(rendered[(nm, k)][0])
        print('%s: 60-120 %.1f dB, 120-250 %.1f dB, 4-8k %.1f dB, 8-16k %.1f dB' % (k, be[1], be[2], be[7], be[8]))
        check(be[1] < lo_band_max and be[8] < hi_band_max, '%s not band-limited' % k)
    # stutter grid: onsets at multiples of a 16th
    def stutter_grid_ok(y, reps, grid, decay=0.08):
        """repeat k must equal repeat 0 scaled by (1-decay)^k at the sample-exact grid position round(k*grid*BEAT*SR),
        the fades must null the sample at every grid boundary, and the first onset (envelope > -24 dB) must sit
        within 4 ms of t=0."""
        step = int(round(grid * BEAT * SR)); s0 = y[:step]
        pos = [int(round(k * grid * BEAT * SR)) for k in range(reps)]
        ok = True
        for k in range(1, reps - 1):
            seg = y[pos[k]:pos[k] + step]
            ok &= np.allclose(seg, s0 * (1 - decay) ** k, atol=1e-9)
            ok &= np.abs(y[pos[k] - 1:pos[k] + 1]).max() < 1e-6
        ons = onset_times(y, -24)
        ok &= len(ons) > 0 and ons[0] < 0.004
        return ok, ons
    st, _ = rendered[(nm, 'stutter')]
    ok, ons = stutter_grid_ok(st, 4, 0.25)
    print('stutter meham3 16th: len %.3fs grid %.4f s, first onsets %s, grid-exact copies: %s' % (len(st) / SR, 0.25 * BEAT, np.round(ons[:4], 4), ok))
    check(ok, 'stutter 16th repeats are not exact grid copies')
    st32 = MV.variant('mu1', 'stutter', reps=8, grid=0.125)
    ok32, ons32 = stutter_grid_ok(st32, 8, 0.125)
    check(len(st32) == int(round(7 * 0.125 * BEAT * SR)) + len(MV.word('mu1')), 'stutter 32nd length')
    check(ok32, 'stutter 32nd repeats are not exact grid copies')
    print('stutter mu1 32nd: len %.3fs (7 x 68.2 ms + 155 ms word), onsets %s, grid-exact copies: %s' % (len(st32) / SR, np.round(ons32[:8], 4), ok32))
    # reverse swell: crescendo, ends loud, starts quiet
    sw, lead = rendered[(nm, 'reverse_swell')]
    first = rms_db(sw[:int(0.1 * SR)]); last = rms_db(sw[-int(0.1 * SR):])
    print('reverse_swell: lead_in %.3fs, first 100 ms %.1f dB, last 100 ms %.1f dB' % (lead, first, last))
    check(last - first > 25, 'reverse_swell not a crescendo (%.1f dB)' % (last - first))
    check(last > -20, 'reverse_swell ends too quiet')
    end_step = np.abs(sw[-1]).max() / np.abs(sw).max()
    check(end_step < 0.01, 'reverse_swell ends with a step (%.3f of peak) -> click when used standalone' % end_step)
    sw2, lead2 = MV.variant(nm, 'reverse_swell', with_word=True)
    i0 = int(round(lead2 * SR))
    check(len(sw2) == i0 + n, 'reverse_swell with_word length')
    e_before = rms_db(sw2[i0 - int(0.05 * SR):i0]); e_after = rms_db(sw2[i0:i0 + int(0.1 * SR)])
    print('reverse_swell with_word: lead_in %.3fs, 50 ms before onset %.1f dB, 100 ms after %.1f dB' % (lead2, e_before, e_after))
    # glitch: deterministic per seed, different per seed
    g1 = MV.variant(nm, 'glitch', seed=3); g2 = MV.variant(nm, 'glitch', seed=3); g3 = MV.variant(nm, 'glitch', seed=4)
    check(np.array_equal(g1, g2), 'glitch not deterministic')
    check(len(g1) != len(g3) or not np.allclose(g1, g3), 'glitch seed has no effect')
    # memo returns copies
    c1 = MV.variant(nm, 'clean'); c1[:] = 0; c2 = MV.variant(nm, 'clean')
    check(np.abs(c2).max() > 0.4, 'variant memo returned a shared array')

    # ---------------------------------------------------------------- song-like workload timing (warm disk cache)
    MV._mem.clear()
    calls = [('p2', 'hall'), ('mushe1', 'demon'), ('meham3', 'reverse_swell'), ('p2', 'double'), ('full', 'clean'), ('full', 'hall'),
             ('meham3', 'shout_stack'), ('meham3', 'hall'), ('mu1', 'stutter', dict(reps=4, grid=0.25)),
             ('ata1', 'clean'), ('mushe1', 'clean'), ('ata2', 'clean'), ('mushe2', 'clean'), ('meham1', 'clean'), ('meham1', 'hall'),
             ('p3', 'hall'), ('meham3', 'megaphone'), ('p12', 'radio'), ('mushe1', 'glitch', dict(seed=3)),
             ('mu1', 'stutter', dict(reps=8, grid=0.125)), ('p3a', 'clean'), ('meham3', 'demon'), ('ata2', 'demon'), ('mushe2', 'demon'),
             ('meham1', 'slowmo'), ('p2', 'megaphone'), ('mushe1', 'hall'), ('full', 'double'), ('ata1', 'double'), ('mushe1', 'double'),
             ('meham1', 'double'), ('mushe3', 'double'), ('ata3', 'double'), ('meham3', 'tele'), ('meham1', 'tele')]
    t0 = time.time()
    for c in calls:
        kw = c[2] if len(c) > 2 else {}
        MV.variant(c[0], c[1], **kw)
    t_song = time.time() - t0
    t0 = time.time()
    for c in calls * 3:
        kw = c[2] if len(c) > 2 else {}
        MV.variant(c[0], c[1], **kw)
    t_memo = time.time() - t0
    print('\ntiming: 26 test renders cold %.1fs; %d distinct song-style calls %.1fs (disk cache warm); %d memoised repeats %.2fs'
          % (t_cold, len(calls), t_song, 3 * len(calls), t_memo))
    check(t_song < 60, 'song workload too slow')

    # ---------------------------------------------------------------- demo wavs + spectrograms
    for nm in NAMES:
        parts, labels, pos = [], [], 0.0
        gap = np.zeros((int(0.25 * SR), 2))
        for k in MV.KINDS:
            if k == 'raw':
                continue
            y, _ = rendered[(nm, k)]
            labels.append((pos, k)); parts.append(y); parts.append(gap); pos += (len(y) + len(gap)) / SR
        demo = np.vstack(parts)
        wav = os.path.join(HERE, 'vocal_%s_demo.wav' % nm); png = os.path.join(HERE, 'vocal_%s_spec.png' % nm)
        E.write(wav, demo); spectrogram_png(demo, png, 'metal_vocal variants of %s' % nm, labels)
        print('wrote', wav, png)
    # zoomed: demon vs clean vs shout_stack vs double (meham3)
    y = np.vstack([rendered[('meham3', k)][0] for k in ('clean', 'shout_stack', 'demon', 'double')])
    lab = []; pos = 0.0
    for k in ('clean', 'shout_stack', 'demon', 'double'):
        lab.append((pos, k)); pos += len(rendered[('meham3', k)][0]) / SR
    png = os.path.join(HERE, 'vocal_meham3_stack_spec.png'); spectrogram_png(y, png, 'meham3: clean / shout_stack / demon / double', lab)
    print('wrote', png)

    print('\n%d failures' % len(FAILS))
    for f in FAILS:
        print(' -', f)
    return len(FAILS)


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
