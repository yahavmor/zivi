"""Adversarial critic test for metal_vocal.variant.
Renders every kind for six names (meham3, p2, p3a, mushe1, ata1, full), checks stereo shape / finite / peak /
DC / plausible lengths, measures 'demon' and 'slowmo' f0 with an independent autocorrelation tracker (not YIN),
checks 'double' L/R decorrelation, 'reverse_swell' ending at the word onset (and the size of the step at its end),
stutter onsets on the 110 BPM grid, hall tail width, and runs whisper (sherpa-onnx turbo) on 'full' raw/clean/hall.
Run: python3 tests/critic_metal_vocal.py  [--no-whisper]"""
import os, sys, time, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
from scipy import signal
import engine as E
import metal_vocal as MV

SR = E.SR
BEAT = MV.BEAT
NAMES = ('meham3', 'p2', 'p3a', 'mushe1', 'ata1', 'full')
FAILS = []
WARN = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print('  FAIL:', msg)


def warn(cond, msg):
    if not cond:
        WARN.append(msg)
        print('  WARN:', msg)


def rms_db(x):
    return 20 * np.log10(np.sqrt((np.asarray(x) ** 2).mean()) + 1e-12)


def f0_autocorr(x, fmin, fmax, win=2400, hop=480, rel_rms=0.1):
    """independent f0 tracker: framewise normalised autocorrelation, octave-error guard, median over voiced frames."""
    x = E.mono(x)
    tmin, tmax = int(SR / fmax), int(SR / fmin)
    frames = range(0, len(x) - win - tmax, hop)
    rmss = np.array([np.sqrt((x[s:s + win] ** 2).mean()) for s in frames])
    thr = rmss.max() * rel_rms
    f0s = []
    for s, r in zip(frames, rmss):
        if r < thr:
            continue
        fr = x[s:s + win + tmax]
        a = fr[:win] - fr[:win].mean()
        ac = signal.correlate(fr - fr[:win].mean(), a, mode='valid')     # ac[tau] = sum a[j] fr[j+tau]
        ac = ac / (ac[0] + 1e-12)
        seg = ac[tmin:tmax]
        best = tmin + int(np.argmax(seg))
        if ac[best] < 0.5:
            continue
        for k in (3, 2):                                                   # prefer a shorter lag if it is nearly as good
            cand = best // k
            if cand >= tmin and ac[cand] > 0.8 * ac[best]:
                best = cand
                break
        f0s.append(SR / best)
    return float(np.median(f0s)) if len(f0s) >= 3 else float('nan')


def onset_times(x, thr_db=-24):
    e = signal.fftconvolve(np.abs(E.mono(x)), np.ones(48) / 48, 'same')
    e = e / e.max()
    on = (e > E.db(thr_db)).astype(int)
    d = np.diff(np.concatenate([[0], on]))
    return np.where(d > 0)[0] / SR


def main(do_whisper=True):
    t_all = time.time()
    rendered = {}
    print('=== render all kinds x %d names ===' % len(NAMES))
    for nm in NAMES:
        w = MV.word(nm); n = len(w)
        print('\n-- %s (word %.3fs)' % (nm, n / SR))
        for k in MV.KINDS:
            t0 = time.time()
            y = MV.variant(nm, k)
            dt = time.time() - t0
            lead = None
            if isinstance(y, tuple):
                y, lead = y
            rendered[(nm, k)] = (y, lead)
            check(isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 2 and y.dtype == np.float64, '%s/%s shape/dtype %s %s' % (nm, k, getattr(y, 'shape', None), getattr(y, 'dtype', None)))
            check(np.isfinite(y).all(), '%s/%s NaN/inf' % (nm, k))
            pk = np.abs(y).max()
            check(0.3 <= pk <= 0.7, '%s/%s peak %.3f outside 0.3-0.7' % (nm, k, pk))
            check(abs(pk - 0.5) < 1e-6, '%s/%s peak %.6f != 0.5' % (nm, k, pk))
            dc = np.abs(y.mean(0)).max()
            check(dc < 2e-3, '%s/%s DC %.4f' % (nm, k, dc))
            lr = np.corrcoef(y[:, 0], y[:, 1])[0, 1] if np.abs(y).max() > 0 else 1.0
            print('%-14s len %.3fs (%.2fx) peak %.3f rms %6.1f dB L/R corr %+.3f  %.2fs' % (k, len(y) / SR, len(y) / n, pk, rms_db(E.mono(y)), lr, dt))
            # ---- lengths
            if k in ('raw', 'clean', 'demon', 'shout_stack', 'megaphone', 'radio', 'tele'):
                check(len(y) == n, '%s/%s length %d != word %d' % (nm, k, len(y), n))
            elif k == 'hall':
                check(n + 3.0 * SR <= len(y) <= n + 4.5 * SR, '%s/hall length %.2fs not word + 3-4.5 s' % (nm, len(y) / SR))
            elif k == 'double':
                check(n <= len(y) <= n + int(0.05 * SR), '%s/double length %d' % (nm, len(y)))
            elif k == 'stutter':
                want = int(round(3 * 0.25 * BEAT * SR)) + n
                check(len(y) == want, '%s/stutter length %d != %d' % (nm, len(y), want))
            elif k == 'reverse_swell':
                check(lead is not None and abs(len(y) / SR - lead) < 1e-6, '%s/reverse_swell lead_in != len' % nm)
                check(1.0 < lead < 3.5, '%s/reverse_swell lead_in %.2fs implausible' % (nm, lead))
            elif k == 'glitch':
                check(0.5 * n <= len(y) <= 2.5 * n, '%s/glitch length %d' % (nm, len(y)))
            elif k == 'slowmo':
                check(abs(len(y) - 2 * n) <= 2, '%s/slowmo length %d != %d' % (nm, len(y), 2 * n))
    t_render = time.time() - t_all

    # ------------------------------------------------------------------ per-name structural checks
    print('\n=== pitch: clean vs demon vs slowmo (independent autocorrelation tracker) ===')
    for nm in NAMES:
        fc = f0_autocorr(rendered[(nm, 'clean')][0], 120, 450)
        fd = f0_autocorr(rendered[(nm, 'demon')][0], 55, 240)
        fs = f0_autocorr(rendered[(nm, 'slowmo')][0], 55, 240)
        fss = f0_autocorr(rendered[(nm, 'shout_stack')][0], 120, 450)
        rd = np.log2(fd / fc) if fd == fd and fc == fc else float('nan')
        rs = np.log2(fs / fc) if fs == fs and fc == fc else float('nan')
        print('%-7s clean %5.0f Hz  demon %5.0f Hz (%+.2f oct)  slowmo %5.0f Hz (%+.2f oct)  shout_stack %5.0f Hz' % (nm, fc, fd, rd, fs, rs, fss))
        check(abs(rd + 1.0) < 0.17, '%s: demon is not an octave down (%.2f oct)' % (nm, rd))
        check(abs(rs + 1.0) < 0.17, '%s: slowmo is not an octave down (%.2f oct)' % (nm, rs))
        check(abs(np.log2(fss / fc)) < 0.17, '%s: shout_stack f0 moved (%.0f vs %.0f)' % (nm, fss, fc))

    print('\n=== double: L/R must differ, envelopes aligned ===')
    for nm in NAMES:
        d, _ = rendered[(nm, 'double')]
        cc = np.corrcoef(d[:, 0], d[:, 1])[0, 1]
        diff = rms_db(d[:, 0] - d[:, 1]) - rms_db(d[:, 0])
        eL = signal.fftconvolve(np.abs(d[:, 0]), np.ones(480) / 480, 'same'); eR = signal.fftconvolve(np.abs(d[:, 1]), np.ones(480) / 480, 'same')
        lag = (np.argmax(signal.correlate(eL - eL.mean(), eR - eR.mean(), 'full')) - (len(eL) - 1)) / SR * 1000
        # onset of the double vs the clean word (the song places both at the same t)
        c, _ = rendered[(nm, 'clean')]
        ec = signal.fftconvolve(np.abs(E.mono(c)), np.ones(480) / 480, 'same'); ed = signal.fftconvolve(np.abs(E.mono(d)), np.ones(480) / 480, 'same')
        m = min(len(ec), len(ed))
        lag_c = (np.argmax(signal.correlate(ed[:m] - ed[:m].mean(), ec[:m] - ec[:m].mean(), 'full')) - (m - 1)) / SR * 1000
        print('%-7s L/R corr %+.3f  (L-R)/L %+.1f dB  env lag L/R %+.1f ms  double-vs-clean lag %+.1f ms  L %.1f R %.1f dBFS' % (nm, cc, diff, lag, lag_c, rms_db(d[:, 0]), rms_db(d[:, 1])))
        check(cc < 0.9 and diff > -6, '%s/double: L and R too similar (corr %.3f)' % (nm, cc))
        check(abs(lag) < 15, '%s/double: envelope lag %.1f ms' % (nm, lag))
        check(abs(rms_db(d[:, 0]) - rms_db(d[:, 1])) < 2.5, '%s/double: L/R level mismatch' % nm)
        check(0 <= lag_c <= 20, '%s/double: offset vs clean %.1f ms (want 0-20 ms, causal)' % (nm, lag_c))

    print('\n=== reverse_swell: ends at the word onset ===')
    for nm in NAMES:
        sw, lead = rendered[(nm, 'reverse_swell')]
        sw2, lead2 = MV.variant(nm, 'reverse_swell', with_word=True)
        i0 = int(round(lead2 * SR))
        n = len(MV.word(nm))
        check(abs(lead - lead2) < 1e-9, '%s: lead_in differs with_word' % nm)
        check(len(sw2) == i0 + n, '%s: with_word length %d != %d + %d' % (nm, len(sw2), i0, n))
        # the appended word must be the clean word (correlation ~1)
        cw = MV.clean_mono(nm, peak=1.0)
        corr = np.corrcoef(E.mono(sw2[i0:i0 + n]), cw)[0, 1]
        check(corr > 0.999, '%s: with_word tail is not the clean word (corr %.4f)' % (nm, corr))
        # crescendo: 50 ms RMS envelope of the swell must peak inside the last 250 ms before the onset
        env = np.sqrt(signal.fftconvolve(E.mono(sw) ** 2, np.ones(2400) / 2400, 'same'))
        tpk = np.argmax(env) / SR
        first = rms_db(sw[:int(0.1 * SR)]); last = rms_db(sw[-int(0.1 * SR):])
        # the step at the very end (standalone swell with no word following): size of the last sample / peak
        end_step = np.abs(sw[-1]).max() / np.abs(sw).max()
        print('%-7s lead_in %.3fs  first 100 ms %6.1f dB  last 100 ms %6.1f dB  env peak at %.3fs (lead-%.3fs)  end step %.3f of peak' % (nm, lead, first, last, tpk, lead - tpk, end_step))
        check(last - first > 25 and last > -24, '%s: swell is not a crescendo into the onset' % nm)
        check(lead - tpk < 0.35, '%s: swell peaks %.2fs before the onset' % (nm, lead - tpk))
        check(end_step < 0.02, '%s: swell ends with a step of %.1f%% of peak (click when used standalone)' % (nm, end_step * 100))

    print('\n=== stutter on the 110 BPM grid (sample-exact copies at round(k*grid*BEAT*SR), landing = clean word) ===')
    for nm, reps, grid in (('meham3', 4, 0.25), ('p2', 4, 0.25), ('mu1', 8, 0.125), ('mu1', 4, 0.25), ('ata1', 6, 0.125), ('full', 3, 0.5)):
        y = MV.variant(nm, 'stutter', reps=reps, grid=grid)
        n = len(MV.word(nm))
        pos = [int(round(k * grid * BEAT * SR)) for k in range(reps)]
        step = int(round(grid * BEAT * SR))
        s0 = y[:step]
        exact = all(np.allclose(y[pos[k]:pos[k] + step], s0 * 0.92 ** k, atol=1e-9) for k in range(1, reps - 1))
        nulls = all(np.abs(y[pos[k] - 1:pos[k] + 1]).max() < 1e-6 for k in range(1, reps))
        cw = MV.clean_mono(nm, peak=1.0)
        land = np.corrcoef(E.mono(y[pos[-1]:pos[-1] + n]), cw)[0, 1]
        # where does the audible energy of each repeat start, relative to its own cell (local -20 dB)?
        starts = []
        for k in range(reps):
            cell = E.mono(y[pos[k]:pos[k] + step]); e = signal.fftconvolve(np.abs(cell), np.ones(48) / 48, 'same')
            starts.append(np.argmax(e > e.max() * E.db(-20)) / SR * 1000)
        print('%-7s reps %d grid %.3f: len %.3fs = %d + %d  exact copies %s  boundary nulls %s  landing corr %.4f  energy starts %s ms into each cell' % (
            nm, reps, grid, len(y) / SR, pos[-1], n, exact, nulls, land, np.round(starts, 1)))
        check(exact and nulls, '%s stutter: repeats are not sample-exact grid copies' % nm)
        check(land > 0.999, '%s stutter: landing is not the clean word (corr %.4f)' % (nm, land))
        check(len(y) == pos[-1] + n, '%s stutter length %d != %d' % (nm, len(y), pos[-1] + n))
        check(max(starts[:-1]) - min(starts[:-1]) < 1.0, '%s stutter: energy start differs between repeats' % nm)

    print('\n=== hall: tail width and decay ===')
    for nm in NAMES:
        h, _ = rendered[(nm, 'hall')]
        n = len(MV.word(nm))
        tail = h[n + int(0.2 * SR): n + int(1.2 * SR)]
        cc = np.corrcoef(tail[:, 0], tail[:, 1])[0, 1]
        a = rms_db(h[n + int(0.3 * SR): n + int(0.5 * SR)]); b = rms_db(h[n + int(2.3 * SR): n + int(2.5 * SR)])
        end = rms_db(h[-int(0.05 * SR):])
        print('%-7s tail L/R corr %+.3f  +0.3s %.1f dB  +2.3s %.1f dB  slope %.1f dB/s  last 50 ms %.1f dB' % (nm, cc, a, b, (b - a) / 2.0, end))
        check(cc < 0.6, '%s/hall tail is nearly mono (corr %.2f)' % (nm, cc))
        check(-24 < (b - a) / 2.0 < -11, '%s/hall decay slope %.1f dB/s' % (nm, (b - a) / 2.0))
        check(end < -70, '%s/hall does not end quietly (%.1f dB)' % (nm, end))

    print('\n=== determinism (same process, memo cleared) ===')
    MV._mem.clear()
    for nm, k, kw in (('p2', 'double', {}), ('p2', 'glitch', dict(seed=3)), ('p2', 'megaphone', {}), ('p2', 'radio', {})):
        a = MV.variant(nm, k, **kw); MV._mem.clear(); b = MV.variant(nm, k, **kw)
        check(np.array_equal(a, b), '%s/%s not deterministic' % (nm, k))

    # ------------------------------------------------------------------ whisper on the full sentence
    if do_whisper:
        print('\n=== whisper (sherpa-onnx turbo, he) on full sentence ===')
        try:
            import sherpa_onnx
            from scipy.signal import resample_poly
            M = os.path.join(HERE, '..', '..', '..', 'models', 'sherpa-onnx-whisper-turbo')
            rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=os.path.join(M, 'turbo-encoder.int8.onnx'), decoder=os.path.join(M, 'turbo-decoder.int8.onnx'),
                                                             tokens=os.path.join(M, 'turbo-tokens.txt'), language='he', task='transcribe', num_threads=4)
            expect = {'אתה': 3, 'מושעה': 3, 'מהמשחק': 2}

            def transcribe(y):
                m16 = resample_poly(E.mono(y) / (np.abs(y).max() + 1e-9) * 0.7, 1, 3).astype('float32')
                seg = np.concatenate([np.zeros(6400, 'float32'), m16, np.zeros(9600, 'float32')])
                st = rec.create_stream(); st.accept_waveform(16000, seg); rec.decode_stream(st)
                return st.result.text

            def score(txt):
                norm = lambda t: t.replace('\u05e2', '')                     # whisper spells מושעה as מושה; ignore the ayin
                toks = [norm(t) for t in re.sub(r'[^\w\s]', ' ', txt).split()]
                return sum(min(toks.count(norm(w)), c) for w, c in expect.items()), sum(expect.values())

            for k, kw in (('raw', {}), ('clean', {}), ('hall', {}), ('hall', dict(duck_db=0)), ('double', {}), ('shout_stack', {})):
                y = MV.variant('full', k, **kw)
                txt = transcribe(y)
                s, tot = score(txt)
                print('%-18s %d/%d expected words  %r' % (k + (' (un-ducked)' if kw else ''), s, tot, txt))
                if k in ('clean', 'hall') and not kw:
                    check(s >= tot - 1, 'whisper: full/%s only %d/%d expected words: %r' % (k, s, tot, txt))
                elif not kw:
                    warn(s >= tot - 2, 'whisper: full/%s only %d/%d expected words: %r' % (k, s, tot, txt))
        except Exception as ex:
            check(False, 'whisper check failed to run: %r' % (ex,))

    print('\nrender time for %d variants: %.1fs; total %.1fs' % (len(rendered), t_render, time.time() - t_all))
    print('\n%d failures, %d warnings' % (len(FAILS), len(WARN)))
    for f in FAILS:
        print(' - FAIL', f)
    for f in WARN:
        print(' - WARN', f)
    return len(FAILS)


if __name__ == '__main__':
    sys.exit(1 if main('--no-whisper' not in sys.argv) else 0)
