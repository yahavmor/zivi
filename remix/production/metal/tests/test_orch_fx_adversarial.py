"""Adversarial test for metal_orch_fx.py: every public function with typical AND edge arguments
(dur 0.1 / 8 s, vel 0.1 / 1.0, programs 48/61/52/47/55), shape/peak/NaN/DC checks, tail-to-silence and
end-click checks, plus behavioural checks (impact = sub + transient + ring, riser centroid rises,
sub_drop glides down, downlifter falls, drone level does not pulse, metal_hit partials, orch family EQ).
Run: python3 tests/test_orch_fx_adversarial.py   (prints a table, writes tests/orchfx_adv_spec.png / orchfx_drone_env.png)"""
import os, sys, time, traceback
import numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, '..'))
import engine as E
import metal_orch_fx as O
SR = E.SR
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
FAILS = []


def fail(msg):
    FAILS.append(msg); print("  !! FAIL:", msg)


def band_db(x):
    m = E.mono(x); f, P = signal.welch(m, SR, nperseg=min(8192, len(m)))
    e = [P[(f >= lo) & (f < hi)].sum() + 1e-20 for lo, hi in BANDS]
    return [10 * np.log10(v / e[3]) for v in e]


def centroid(x):
    m = E.mono(x); f, P = signal.welch(m, SR, nperseg=min(4096, len(m)))
    return float((f * P).sum() / (P.sum() + 1e-20))


def rms_env(m, win_s=0.1):
    w = max(1, int(win_s * SR)); nb = len(m) // w
    return np.sqrt((m[:nb * w].reshape(nb, w) ** 2).mean(1)) if nb > 0 else np.array([np.sqrt((m ** 2).mean())])


def zc_freq(m):
    s = np.sign(m); z = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
    return SR / np.diff(z).mean() if len(z) > 2 else 0.0


def check(name, fn, stereo=None, pk=(0.25, 0.62), hard_end=False, min_len=None):
    """render, verify basics; returns the array (or None on exception)."""
    try:
        t0 = time.time(); x = fn(); dt = time.time() - t0
    except Exception as e:
        fail("%s raised %s: %s" % (name, type(e).__name__, e)); traceback.print_exc(); return None
    x = np.asarray(x)
    if x.dtype != np.float64: fail("%s dtype %s" % (name, x.dtype))
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] != 2): fail("%s bad shape %s" % (name, x.shape)); return None
    if stereo is True and x.ndim != 2: fail("%s expected stereo, got %s" % (name, x.shape))
    if stereo is False and x.ndim != 1: fail("%s expected mono, got %s" % (name, x.shape))
    if not np.isfinite(x).all(): fail("%s has NaN/inf" % name); return None
    if min_len is not None and len(x) < min_len: fail("%s too short %d < %d" % (name, len(x), min_len))
    p = float(np.abs(x).max()); rms = float(np.sqrt((x ** 2).mean())); dc = float(E.mono(x).mean())
    if not (pk[0] <= p <= pk[1]): fail("%s peak %.3f not in %s" % (name, p, pk))
    if abs(dc) > (0.004 if len(x) >= 0.5 * SR else 0.01): fail("%s DC %.4f" % (name, dc))   # <0.5 s of a 25 Hz sine has no meaningful mean
    m = E.mono(x); w = int(0.1 * SR); nb = len(m) // w
    if nb > 0:
        bm = np.abs(m[:nb * w].reshape(nb, w).mean(1)).max()
        if bm > 0.08 * p: fail("%s block DC %.4f vs peak %.3f" % (name, bm, p))
    # end: last sample ~0 and no step at the end (click); non-hard-end renders must have decayed
    last = float(np.abs(x[-1]).max()); last1ms = float(np.abs(x[-48:]).max())
    if last > 0.02 * p: fail("%s last sample %.4f (peak %.3f) -> end click" % (name, last, p))
    if not hard_end:
        tail = float(np.sqrt((x[-int(0.01 * SR):] ** 2).mean())); lim = 0.03 if len(x) > 0.5 * SR else 0.06
        if last1ms > 0.05 * p: fail("%s last 1 ms absmax %.4f = %.0f%% of peak (no fade-out)" % (name, last1ms, 100 * last1ms / p))
        if tail > lim * p: fail("%s last 10 ms rms %.4f = %.0f%% of peak (tail not silent)" % (name, tail, 100 * tail / p))
    # start: no step at sample 0
    first = float(np.abs(x[0]).max())
    if first > 0.05 * p: fail("%s first sample %.4f (peak %.3f) -> start click" % (name, first, p))
    b = band_db(x)
    print("  %-26s %-12s %6.2fs %5.2fs peak %.3f rms %.4f dc %+.5f | %s" % (name, str(x.shape), len(x) / SR, dt, p, rms, dc, ' '.join('%+5.0f' % v for v in b)))
    return x


def spectrogram_png(x, path, title, marks=(), fmax=16000):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    m = E.mono(x); f, t, S = signal.spectrogram(m, SR, nperseg=2048, noverlap=1536, window='hann')
    S = 10 * np.log10(S + 1e-12); S = np.clip(S, S.max() - 90, S.max())
    fig, ax = plt.subplots(figsize=(18, 6)); sel = f <= fmax
    im = ax.pcolormesh(t, f[sel], S[sel], shading='auto', cmap='magma')
    ax.set_yscale('symlog', linthresh=200); ax.set_ylim(20, fmax); ax.set_xlabel('s'); ax.set_ylabel('Hz')
    for name, tt in marks: ax.axvline(tt, color='w', lw=0.5, alpha=0.6); ax.text(tt + 0.05, fmax * 0.7, name, color='w', fontsize=7, rotation=90)
    ax.set_title(title); fig.colorbar(im, ax=ax, label='dB'); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return path


def montage(items, gap=0.3):
    out = []; marks = []
    for name, x in items:
        if x is None: continue
        marks.append((name, sum(len(o) for o in out) / SR)); out.append(E.stereo(x)); out.append(np.zeros((int(gap * SR), 2)))
    return np.vstack(out), marks


def main():
    T0 = time.time(); shots = []
    print("bands dB rel 250-500:", ' '.join('%d-%d' % b for b in BANDS))
    # ------------------------------------------------------------------ impact: sub + transient + ring
    print("# impact")
    for vel, seed in [(1.0, 0), (0.1, 1), (0.8, 2), (0.6, 3)]:
        x = check('impact v%.1f s%d' % (vel, seed), lambda: O.impact(vel, seed=seed), True, pk=(0.5 * vel, 0.62 * vel))
        if x is None: continue
        if vel == 1.0: shots.append(('impact', x))
        m = E.mono(x); p = np.abs(m).max()
        sub = E.lp(m, 80, 0.707, 4); mid = E.hp(m, 300, 0.707, 4)
        esub = (sub ** 2).sum() / (m ** 2).sum()
        if esub < 0.5: fail("impact v%.1f sub(<80Hz) energy fraction %.2f < 0.5" % (vel, esub))
        tr_peak = np.abs(mid[:int(0.03 * SR)]).max()                          # transient = >300 Hz content in first 30 ms
        lim = -9 if vel >= 0.5 else -20                                          # soft hits are allowed to be duller
        if 20 * np.log10(tr_peak / p + 1e-12) < lim: fail("impact v%.1f transient (>300 Hz, first 30 ms) only %.1f dB below peak -> no 'hit'" % (vel, 20 * np.log10(tr_peak / p)))
        late = np.abs(mid[int(0.5 * SR):]).max()
        if late > tr_peak: fail("impact v%.1f highs louder after 0.5 s than at onset" % vel)
        # ring: partials of the 140 Hz anvil (140, 386, 756) should stand above the local spectrum between 0.25 and 1.0 s
        seg = m[int(0.25 * SR):int(1.0 * SR)]; f, P = signal.welch(seg, SR, nperseg=8192)
        ok = 0
        for fr in (140.0, 140 * 2.76, 140 * 5.4):
            sel = (f > fr * 0.97) & (f < fr * 1.03); nb = (f > fr * 0.8) & (f < fr * 1.25)
            if P[sel].max() > 4 * np.median(P[nb]): ok += 1
        if ok < 2: fail("impact v%.1f metallic ring not detectable (only %d/3 partials)" % (vel, ok))
        env = rms_env(sub, 0.05); tpk = np.argmax(env) * 0.05
        if tpk > 0.4: fail("impact sub peaks late (%.2f s)" % tpk)
        print("     sub frac %.2f  transient %.1f dB vs peak  ring partials %d/3  sub env peak %.2fs  sub -40dB at %.2fs" % (
            esub, 20 * np.log10(tr_peak / p), ok, tpk, (np.argmax(20 * np.log10(env / env.max() + 1e-12) < -40) * 0.05)))
    x = check('impact dur0.5', lambda: O.impact(1.0, dur=0.5), True, pk=(0.5, 0.62))
    x = check('impact dur8', lambda: O.impact(1.0, dur=8.0), True, pk=(0.5, 0.62))
    # ------------------------------------------------------------------ sub_drop glides down
    print("# sub_drop")
    for dur, f0, f1 in [(1.5, 70, 25), (0.1, 80, 25), (8.0, 80, 25), (2.0, 70, 20), (1.6, 80, 22)]:
        x = check('sub_drop %.1f %d->%d' % (dur, f0, f1), lambda: O.sub_drop(dur, f0, f1), False, pk=(0.45, 0.55))
        if x is None: continue
        if dur == 1.5: shots.append(('sub_drop', x))
        n = len(x); fr = []
        for a, b in [(0, 0.1), (0.3, 0.45), (0.6, 0.8), (0.85, 1.0)]:
            seg = x[int(a * n):max(int(b * n), int(a * n) + 8)]; fr.append(zc_freq(seg))
        print("     glide Hz:", ' -> '.join('%.1f' % v for v in fr))
        if dur >= 1.0:
            if not all(fr[i] > fr[i + 1] for i in range(len(fr) - 1)): fail("sub_drop %.1f does not glide monotonically: %s" % (dur, fr))
            if not (0.75 * f0 < fr[0] < 1.05 * f0): fail("sub_drop start freq %.1f vs f0 %d" % (fr[0], f0))
            if not (0.85 * f1 < fr[-1] < 1.4 * f1): fail("sub_drop end freq %.1f vs f1 %d" % (fr[-1], f1))
        hp = E.hp(x, 300, 0.707, 4)
        if (hp ** 2).sum() / (x ** 2).sum() > 0.01: fail("sub_drop has >1%% energy above 300 Hz")
    # ------------------------------------------------------------------ riser rises
    print("# riser")
    for dur, f0, f1 in [(2 * O.BAR, 200, 6000), (0.1, 200, 6000), (8.0, 120, 10000), (3 * O.BAR, 150, 9000)]:
        x = check('riser %.2f %d->%d' % (dur, f0, f1), lambda: O.riser(dur, f0, f1), True, hard_end=True, pk=(0.45, 0.55))
        if x is None: continue
        if abs(dur - 2 * O.BAR) < 1e-6: shots.append(('riser', x))
        if len(x) != O._n(dur): fail("riser length %d != %d" % (len(x), O._n(dur)))
        if dur >= 1.0:
            q = [x[int(i * len(x) / 4):int((i + 1) * len(x) / 4)] for i in range(4)]
            cs = [centroid(s) for s in q]; rs = [float(np.sqrt((s ** 2).mean())) for s in q]
            print("     centroid Hz by quarter:", ' '.join('%.0f' % c for c in cs), " rms:", ' '.join('%.4f' % r for r in rs))
            if not all(cs[i] < cs[i + 1] for i in range(3)): fail("riser %.2f centroid not rising: %s" % (dur, cs))
            if not all(rs[i] < rs[i + 1] for i in range(3)): fail("riser %.2f level not rising: %s" % (dur, rs))
            if rs[-1] / (rs[0] + 1e-9) < 8: fail("riser %.2f dynamic range too small: %.1f" % (dur, rs[-1] / rs[0]))
            # ends at full level: last 100 ms (before the 15 ms stop) should be the loudest 100 ms
            e = rms_env(E.mono(x), 0.1)
            if np.argmax(e) < len(e) - 2: fail("riser %.2f loudest window is not at the end (idx %d of %d)" % (dur, np.argmax(e), len(e)))
    # ------------------------------------------------------------------ downlifter falls
    print("# downlifter")
    for dur in (2.0, 0.1, 8.0):
        x = check('downlifter %.1f' % dur, lambda: O.downlifter(dur), True, pk=(0.45, 0.55))
        if x is None: continue
        if dur == 2.0: shots.append(('downlifter', x))
        if len(x) != O._n(dur) + O._n(1.0): fail("downlifter length %d != dur+1s" % len(x))
        if dur >= 1.0:
            n = O._n(dur); q = [x[int(i * n / 4):int((i + 1) * n / 4)] for i in range(4)]
            cs = [centroid(s) for s in q]; rs = [float(np.sqrt((s ** 2).mean())) for s in q]
            print("     centroid Hz by quarter:", ' '.join('%.0f' % c for c in cs), " rms:", ' '.join('%.4f' % r for r in rs))
            if not all(cs[i] > cs[i + 1] for i in range(3)): fail("downlifter centroid not falling: %s" % cs)
            if not all(rs[i] > rs[i + 1] for i in range(3)): fail("downlifter level not falling: %s" % rs)
    # ------------------------------------------------------------------ metal_hit partials
    print("# metal_hit")
    for vel, pitch, seed in [(0.7, 420, 0), (1.0, 600, 2), (0.1, 420, 0), (1.0, 480, 1), (0.7, 1200, 3)]:
        x = check('metal_hit v%.1f %dHz' % (vel, pitch), lambda: O.metal_hit(vel, pitch, seed=seed), False, pk=(0.45 * vel, 0.55 * vel))
        if x is None: continue
        if vel == 0.7 and pitch == 420: shots.append(('metal_hit', x))
        f, P = signal.welch(x[int(0.05 * SR):], SR, nperseg=16384); found = []
        for ratio in (1.0, 2.76, 5.40, 8.93):
            fr = pitch * ratio
            if fr > 20000: continue
            sel = (f > fr * 0.96) & (f < fr * 1.04); nb = (f > fr * 0.7) & (f < fr * 1.4)
            found.append(P[sel].max() > 6 * np.median(P[nb]))
        print("     partials found:", found)
        if sum(found) < 3: fail("metal_hit %dHz partials missing: %s" % (pitch, found))
        if (E.lp(x, 100, 0.707, 4) ** 2).sum() / (x ** 2).sum() > 0.02: fail("metal_hit has >2%% energy below 100 Hz")
    for dur in (0.1, 8.0):
        check('metal_hit dur%.1f' % dur, lambda: O.metal_hit(1.0, 420, dur=dur), False, pk=(0.45, 0.55))
    # ------------------------------------------------------------------ industrial_clank (incl. short dur)
    print("# industrial_clank")
    for seed in range(6):
        x = check('clank s%d' % seed, lambda: O.industrial_clank(seed), False, pk=(0.45, 0.55))
        if seed == 1 and x is not None: shots.append(('clank', x))
        if x is not None:
            # strikes = onsets in the >1.2 kHz band (4 ms frames jumping >5 dB over the previous frame, above -15 dBFS
            # of the band max); all must lie in the first 0.4 s, the metallic ring may last longer
            h = E.hp(x, 1200, 0.707, 2); e = np.r_[1e-9, 1e-9, 1e-9, rms_env(h, 0.004)]; db = 20 * np.log10(e / e.max() + 1e-12)
            ons = [i for i in range(3, len(e)) if db[i] > -15 and db[i] - db[i - 1] > 5]
            ons = [o - 3 for i, o in enumerate(ons) if i == 0 or o - ons[i - 1] > 5]; ons = np.array(ons) * 0.004
            print("     strike onsets at s:", ' '.join('%.3f' % v for v in ons))
            if len(ons) < 2: fail("clank s%d has fewer than 2 distinct strikes" % seed)
            if len(ons) and ons.max() > 0.4: fail("clank s%d strike at %.2f s (expected within ~0.25 s)" % (seed, ons.max()))
            if np.argmax(rms_env(x, 0.005)) * 0.005 > 0.4: fail("clank s%d loudest window at %.2f s" % (seed, np.argmax(rms_env(x, 0.005)) * 0.005))
    for dur in (0.1, 0.2, 0.3, 8.0):
        for seed in range(4):
            check('clank s%d dur%.1f' % (seed, dur), lambda: O.industrial_clank(seed, dur=dur), False, pk=(0.45, 0.55))
    # ------------------------------------------------------------------ reverse_cymbal
    print("# reverse_cymbal")
    for dur, seed in [(1.5, 0), (1.2, 1), (0.1, 0), (8.0, 1), (2.0, 2)]:
        x = check('rev_cymbal %.1f s%d' % (dur, seed), lambda: O.reverse_cymbal(dur, seed), True, hard_end=True, pk=(0.45, 0.55))
        if x is None: continue
        if dur == 1.5: shots.append(('rev_cym', x))
        if len(x) != O._n(dur): fail("reverse_cymbal length %d != %d" % (len(x), O._n(dur)))
        e = rms_env(E.mono(x), 0.05)
        if dur >= 1.0 and np.argmax(e) < len(e) - 3: fail("reverse_cymbal %.1f loudest window not at the end" % dur)
        if centroid(x) < 2500: fail("reverse_cymbal centroid %.0f Hz too dark for a cymbal" % centroid(x))
    # ------------------------------------------------------------------ drone: level stability, pitch, darkness
    print("# drone")
    for root, dur, seed in [(36, 6.0, 0), (36, 12.0, 0), (36, 0.1, 0), (36, 8.0, 1), (39, 8.0, 2), (48, 6.0, 0)]:
        x = check('drone m%d %.1fs s%d' % (root, dur, seed), lambda: O.drone(root, dur, seed), True, pk=(0.4, 0.5))
        if x is None: continue
        if dur == 12.0: shots.append(('drone', x)); drone12 = x
        if len(x) != O._n(dur): fail("drone length")
        if dur >= 6.0:
            m = E.mono(x); a0 = int(min(1.5, dur * 0.3) * SR) + SR // 10; a1 = len(m) - int(min(2.0, dur * 0.35) * SR) - SR // 10
            env = rms_env(m[a0:a1], 0.1); rng_db = 20 * np.log10(env.max() / env.min())
            envs = [rms_env(x[a0:a1, c], 0.1) for c in range(2)]; chdb = max(20 * np.log10(e.max() / e.min()) for e in envs)
            e = env - env.mean(); F = np.abs(np.fft.rfft(e * np.hanning(len(e)))); fq = np.fft.rfftfreq(len(e), 0.1)
            top = fq[np.argmax(F)]
            print("     steady-state level swing mono %.1f dB, per-channel max %.1f dB, dominant env-mod %.2f Hz" % (rng_db, chdb, top))
            if rng_db > 3.0: fail("drone m%d %.0fs level pulses %.1f dB (beating artifact)" % (root, dur, rng_db))
            if chdb > 4.0: fail("drone m%d %.0fs per-channel level pulses %.1f dB" % (root, dur, chdb))
            f, P = signal.welch(m, SR, nperseg=32768); fpk = f[np.argmax(P)]; f0 = E.midi2hz(root)
            if not (abs(fpk - f0) < 0.06 * f0 or abs(fpk - f0 / 2) < 0.06 * f0 or abs(fpk - 2 * f0) < 0.06 * f0): fail("drone dominant %.1f Hz not root %.1f" % (fpk, f0))
            if centroid(x) > 400: fail("drone centroid %.0f Hz not dark" % centroid(x))
    # ------------------------------------------------------------------ noise_bed
    print("# noise_bed")
    for dur, seed in [(6.0, 0), (0.1, 0), (8.0, 1), (9.7, 0)]:
        x = check('noise_bed %.1f s%d' % (dur, seed), lambda: O.noise_bed(dur, seed), True, pk=(0.3, 0.4))
        if x is None: continue
        if dur == 6.0: shots.append(('noise_bed', x))
        if centroid(x) > 500: fail("noise_bed centroid %.0f Hz too bright" % centroid(x))
        if dur >= 6:
            m = E.mono(x); a0 = int(2.1 * SR); a1 = len(m) - int(2.1 * SR); env = rms_env(m[a0:a1], 0.1)
            print("     level swing %.1f dB" % (20 * np.log10(env.max() / env.min())))
            if 20 * np.log10(env.max() / env.min()) > 8: fail("noise_bed level swings %.1f dB" % (20 * np.log10(env.max() / env.min())))
    # ------------------------------------------------------------------ braam
    print("# braam")
    for root, dur, gm in [(36, 2 * O.BAR, True), (36, 0.1, True), (36, 8.0, True), (39, 2.0, False), (48, 1.0, True)]:
        x = check('braam m%d %.1f gm%d' % (root, dur, gm), lambda: O.braam(root, dur, gm_layer=gm), True, pk=(0.45, 0.55))
        if x is None: continue
        if abs(dur - 2 * O.BAR) < 1e-6: shots.append(('braam', x))
        if len(x) != O._n(dur + 3.0): fail("braam length %d != dur+3" % len(x))
        m = E.mono(x); env = rms_env(m, 0.05); ipk = np.argmax(env) * 0.05
        if dur >= 1.0:
            att = min(0.7, dur * 0.35); i90 = np.argmax(env > 0.9 * env.max()) * 0.05
            print("     attack: 90%% at %.2fs (attack param %.2f)  loudest %.2fs  level at dur+1.0s %.1f dB" % (i90, att, ipk, 20 * np.log10(env[min(len(env) - 1, int((dur + 1.0) / 0.05))] / env.max())))
            if i90 > att * 2 + 0.3: fail("braam attack too slow (90%% at %.2f s)" % i90)
            if i90 < att * 0.4: fail("braam attack too fast (90%% at %.2f s, wanted ~%.2f)" % (i90, att))
            sus = env[int((att + 0.3) / 0.05):int(dur / 0.05)]
            if len(sus) > 4 and 20 * np.log10(sus.max() / sus.min()) > 6: fail("braam sustain pulses %.1f dB" % (20 * np.log10(sus.max() / sus.min())))
            if env[min(len(env) - 1, int((dur + 1.0) / 0.05))] > 0.5 * env.max(): fail("braam does not release after note-off")
        f, P = signal.welch(m, SR, nperseg=32768); f0 = E.midi2hz(root)
        sel = (f > f0 * 0.9) & (f < f0 * 1.1)
        if P[sel].max() < 0.1 * P.max(): fail("braam root %.0f Hz weak" % f0)
    # ------------------------------------------------------------------ timpani / orchestra hit / gm_oneshot
    print("# timpani_roll / orchestra_hit / gm_oneshot")
    for midi, dur, vel in [(36, 2.0, 0.9), (36, 0.1, 0.9), (36, 8.0, 1.0), (43, 1.0, 0.1)]:
        x = check('timpani m%d %.1f v%.1f' % (midi, dur, vel), lambda: O.timpani_roll(midi, dur, vel), True, pk=(0.45 * vel, 0.55 * vel))
        if x is None: continue
        if dur == 2.0: shots.append(('timpani', x))
        if len(x) != O._n(dur + 3.0): fail("timpani length")
        if dur >= 1.0:
            env = rms_env(E.mono(x), 0.1); i = np.argmax(env) * 0.1
            if not (dur * 0.5 <= i <= dur + 0.6): fail("timpani loudest at %.1f s, expected crescendo to ~%.1f" % (i, dur))
    for midi, vel in [(48, 1.0), (48, 0.1), (36, 0.8)]:
        x = check('orch_hit m%d v%.1f' % (midi, vel), lambda: O.orchestra_hit(midi, vel), True, pk=(0.45 * vel, 0.55 * vel))
        if midi == 48 and vel == 1.0 and x is not None: shots.append(('orch_hit', x))
    for prog, midi in [(55, 48), (47, 36), (61, 48)]:
        x = check('gm_oneshot p%d' % prog, lambda: O.gm_oneshot(prog, midi, 1.0, 0.9), True, pk=(0.02, 1.0))
    # ------------------------------------------------------------------ orch: programs, families, lengths
    print("# orch")
    chord = [(0.0, 48, 4 * O.BEAT, 0.7), (0.0, 51, 4 * O.BEAT, 0.7), (0.0, 55, 4 * O.BEAT, 0.7), (0.0, 60, 4 * O.BEAT, 0.7),
             (4 * O.BEAT, 46, 4 * O.BEAT, 0.8), (4 * O.BEAT, 53, 4 * O.BEAT, 0.8)]
    for prog in (48, 61, 52, 47, 55):
        x = check('orch p%d' % prog, lambda: O.orch(chord, prog), True, pk=(0.49, 0.51))
        if x is None: continue
        shots.append(('orch%d' % prog, x))
        if len(x) != O._n(8 * O.BEAT + 4.0): fail("orch p%d length %d" % (prog, len(x)))
        fam = O._family(prog); print("     family:", fam, " centroid %.0f Hz" % centroid(x))
        if prog == 47 and fam != 'perc': fail("orch program 47 (timpani) mapped to family %s, not perc" % fam)
        if prog == 55 and fam != 'hit': fail("orch program 55 mapped to %s" % fam)
    # timpani low notes must keep their fundamental (65 Hz) through the family EQ
    x = O.orch([(0.0, 36, 1.0, 0.9), (1.0, 36, 1.0, 0.9)], 47)
    b = band_db(x); print("     orch p47 low timpani bands:", ' '.join('%+5.0f' % v for v in b))
    if b[1] < b[3] - 3: fail("orch p47 timpani 60-120 Hz band %.0f dB under 250-500 -> family EQ killed the fundamental" % (b[1] - b[3]))
    # exact lengths / edge
    for L in (0.1, 4.0, 8.0):
        x = check('orch total_len %.1f' % L, lambda: O.orch(chord, 48, total_len=L), True, pk=(0.49, 0.51), hard_end=True)
        if x is not None and x.shape != (O._n(L), 2): fail("orch total_len %.1f shape %s" % (L, x.shape))
    x = O.orch([], 48, total_len=2.0)
    if x.shape != (O._n(2.0), 2) or np.abs(x).max() != 0: fail("orch empty events")
    x = O.orch([], 48)
    if x.ndim != 2: fail("orch empty events no total_len")
    x = check('orch vel 0.1', lambda: O.orch([(0.0, 48, 1.0, 0.1)], 48), True, pk=(0.49, 0.51))
    x = check('orch dry', lambda: O.orch(chord, 48, reverb_mix=0.0), True, pk=(0.49, 0.51))
    x = check('orch nonorm', lambda: O.orch(chord, 48, normalize=False), True, pk=(0.02, 1.0))
    # note held through total_len: end must not click
    x = check('orch cut note', lambda: O.orch([(0.0, 48, 10.0, 0.9), (0.0, 55, 10.0, 0.9)], 48, total_len=2.0), True, pk=(0.49, 0.51), hard_end=True)
    if x is not None:                                                  # the cut must be a fade, not a step
        a = np.abs(x); l2 = a[-96:].max(); l30 = a[-1440:].max(); pre = a[-2880:-1440].max()
        print("     cut note: last 2 ms %.3f / last 30 ms %.3f / preceding 30 ms %.3f" % (l2, l30, pre))
        if l2 > 0.15 * l30 or l30 > 1.2 * pre: fail("orch cut note: end is a step, not a fade (%.3f/%.3f/%.3f)" % (l2, l30, pre))
    # relative dynamics inside one program preserved
    y = O.orch([(0.0, 60, 1.0, 0.3), (3.0, 60, 1.0, 1.0)], 48, normalize=False)
    e = rms_env(E.mono(y), 0.5); r = 20 * np.log10(e[6] / (e[0] + 1e-12)); print("     vel 1.0 vs 0.3 = %+.1f dB" % r)
    if r < 6: fail("orch velocity dynamics flat (%.1f dB)" % r)
    # hall
    x = check('hall(impulse)', lambda: O.hall(np.r_[np.zeros(100), 1.0, np.zeros(SR * 4)], mix=1.0) * 0.5 / np.abs(O.hall(np.r_[np.zeros(100), 1.0, np.zeros(SR * 4)], mix=1.0)).max(), True, pk=(0.49, 0.51))
    if x is not None:
        m = E.mono(x); env = rms_env(m, 0.1); t60 = np.argmax(20 * np.log10(env / env.max() + 1e-12) < -60) * 0.1
        print("     hall RT(-60dB) ~ %.1f s" % t60)
        if not (2.0 <= t60 <= 4.5): fail("hall RT60 %.1f s not a big hall" % t60)
    # ------------------------------------------------------------------ outputs
    mont, marks = montage(shots)
    E.write(os.path.join(HERE, 'orchfx_adv.wav'), mont)
    print("spectrogram:", spectrogram_png(mont, os.path.join(HERE, 'orchfx_adv_spec.png'), 'metal_orch_fx adversarial montage', marks))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        m = E.mono(drone12); env = rms_env(m, 0.05); t = np.arange(len(env)) * 0.05
        fig, ax = plt.subplots(figsize=(14, 3)); ax.plot(t, 20 * np.log10(env + 1e-9)); ax.set_ylim(-40, -10); ax.set_xlabel('s'); ax.set_ylabel('dB')
        ax.set_title('drone(36, 12 s) RMS envelope (50 ms)'); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, 'orchfx_drone_env.png'), dpi=100); plt.close(fig)
        print("drone envelope plot:", os.path.join(HERE, 'orchfx_drone_env.png'))
    except Exception as e:
        print("drone plot skipped:", e)
    print("total %.0f s" % (time.time() - T0))
    if FAILS:
        print("\n%d FAILURES:" % len(FAILS)); [print("  -", f) for f in FAILS]; sys.exit(1)
    print("ORCH_FX ADVERSARIAL OK")


if __name__ == '__main__':
    main()
