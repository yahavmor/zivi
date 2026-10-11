"""Adversarial stress test for metal_drums.py (critic pass).
Run: python3 tests/test_drums_stress.py   (from the metal folder; writes PNG/WAV into tests/)

Checks: edge cases (unknown name, empty events, total_len 0/short/long, duplicate events, odd velocities),
per-piece spectra at vel 1.0 and 0.2 (kick sub+click, snare crack, cymbal brightness, tom pitch + spread),
a 300-event 4-bar stress pattern (16th double kick, 32nd snare rolls, crash+china+kick stacks, vel 0.2..1.0)
with total_len None / given, NaN/DC/clip checks, L/R difference, limiter engagement, render time, spectrogram.
"""
import os, sys, time
import numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import engine as E
import metal_drums as D

EDGES = [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]
LABELS = ['20-60', '60-120', '120-250', '250-500', '.5-1k', '1-2k', '2-4k', '4-8k', '8-16k']
FAIL = []
BEAT = 60.0 / 110.0


def bands_db(x, nperseg=4096):
    x = E.mono(x); f, P = signal.welch(x, E.SR, nperseg=min(nperseg, len(x)))
    e = [P[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]
    ref = e[3] + 1e-20
    return [10 * np.log10(v / ref + 1e-20) for v in e]


def fmt(b): return ' '.join(f'{v:6.1f}' for v in b)


def basic(name, x, pk_lo=0.05, pk_hi=0.62):
    x = np.asarray(x); m = E.mono(x)
    pk = np.abs(x).max(); rms = np.sqrt((m ** 2).mean()); dc = abs(m.mean()) / (rms + 1e-12)
    ok = True
    if not np.isfinite(x).all(): FAIL.append(f'{name}: NaN/inf'); ok = False
    if dc > 0.02: FAIL.append(f'{name}: DC offset {dc:.3f} of RMS'); ok = False
    if pk > pk_hi: FAIL.append(f'{name}: peak {pk:.3f} > {pk_hi}'); ok = False
    if pk < pk_lo: FAIL.append(f'{name}: peak {pk:.3f} < {pk_lo}'); ok = False
    b = bands_db(x)
    print(f'{name:30s} pk={pk:5.3f} rms={20*np.log10(rms+1e-12):6.1f}dB len={len(x)/E.SR:6.2f}s dc={dc:.4f} '
          f'{"OK " if ok else "BAD"} | {fmt(b)}')
    return pk, rms, b


def spectrogram_png(x, path, title, fmax=16000):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    m = E.mono(x); f, t, S = signal.spectrogram(m, E.SR, nperseg=2048, noverlap=1536, window='hann')
    S = 10 * np.log10(S + 1e-14); top = S.max()
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=90)
    im = ax.pcolormesh(t, f, S, vmin=top - 80, vmax=top, shading='auto', cmap='magma')
    ax.set_yscale('symlog', linthresh=200); ax.set_ylim(20, fmax); ax.set_ylabel('Hz'); ax.set_xlabel('s')
    ax.set_yticks([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_yticklabels(['30', '60', '120', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    ax.set_title(title); fig.colorbar(im, ax=ax, label='dB'); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def fundamental_hz(x, lo=40, hi=400):
    """strongest spectral peak between lo and hi Hz in the first 150 ms (zero-padded FFT)."""
    m = E.mono(x)[:int(0.15 * E.SR)] * np.hanning(int(0.15 * E.SR))
    F = np.abs(np.fft.rfft(m, 1 << 16)); f = np.fft.rfftfreq(1 << 16, 1 / E.SR)
    sel = (f >= lo) & (f <= hi); return float(f[sel][np.argmax(F[sel])])


def lr_db(x):
    return 20 * np.log10((np.sqrt((x[:, 0] ** 2).mean()) + 1e-9) / (np.sqrt((x[:, 1] ** 2).mean()) + 1e-9))


def stress_events(bars=4, start=0.0, seed=5):
    """~300 events: 16th double kick, hats 8ths, snare 2+4, ghost notes, 32nd snare roll on beat 4 of every bar,
    crash+china+kick stacks on every downbeat, tom fill in the last bar, velocities spread 0.2..1.0."""
    r = np.random.default_rng(seed); ev = []
    for b in range(bars):
        t0 = start + b * 4 * BEAT
        ev += [(t0, 'crash1', 1.0), (t0, 'china', 1.0), (t0, 'kick', 1.0)]
        if b % 2 == 1: ev.append((t0, 'crash2', 0.9))
        for k in range(16):
            ev.append((t0 + k * BEAT / 4, 'kick', float(r.uniform(0.2, 1.0)) if k % 4 else 1.0))
        for k in range(8):
            ev.append((t0 + k * BEAT / 2, 'hat_closed' if b < 2 else 'ride', float(r.uniform(0.2, 1.0))))
        ev += [(t0 + BEAT, 'snare', 1.0), (t0 + 3 * BEAT, 'snare', float(r.uniform(0.7, 1.0)))]
        ev += [(t0 + 1.75 * BEAT, 'snare_ghost', 0.3), (t0 + 2.25 * BEAT, 'snare_ghost', 0.2)]
        for k in range(8):   # 32nd snare roll, crescendo
            ev.append((t0 + 3 * BEAT + k * BEAT / 8, 'snare', 0.2 + 0.8 * k / 7))
        ev.append((t0 + 2 * BEAT, 'hat_open', 0.8)); ev.append((t0 + 2.5 * BEAT, 'hat_pedal', 0.5))
        if b == bars - 1:
            fill = ['tom_hi', 'tom_hi', 'tom_mid', 'tom_mid', 'tom_floor', 'tom_floor', 'tom_floor', 'tom_floor']
            for k, nm in enumerate(fill): ev.append((t0 + 2 * BEAT + k * BEAT / 8, nm, 0.6 + 0.4 * k / 7))
            ev += [(t0 + 2.5 * BEAT, 'splash', 0.8), (t0 + 3.5 * BEAT, 'ride_bell', 0.9), (t0 + 3.5 * BEAT, 'rim', 0.5)]
    tend = start + bars * 4 * BEAT
    ev += [(tend, 'crash1', 1.0), (tend, 'crash2', 1.0), (tend, 'china', 1.0), (tend, 'kick', 1.0), (tend, 'snare', 1.0)]
    return ev


if __name__ == '__main__':
    T0 = time.time()
    print('bands (dB rel 250-500):', ' '.join(f'{l:>6s}' for l in LABELS))

    # ------------------------------------------------------------------ edge cases
    for bad in ['cowbell', '', None, 123]:
        try: D.hit(bad); FAIL.append(f'unknown name {bad!r} did not raise')
        except ValueError: pass
        except Exception as e: FAIL.append(f'unknown name {bad!r} raised {type(e).__name__} instead of ValueError')
    try: D.render_kit([(0.0, 'kick', 1.0), (0.5, 'cowbell', 1.0)]); FAIL.append('render_kit unknown name did not raise')
    except ValueError: pass
    print('unknown names raise ValueError: ok')
    x = D.render_kit([]); print(f'empty events -> {x.shape} pk {np.abs(x).max():.3f}')
    if x.shape != (int(3.5 * E.SR), 2) or np.abs(x).max() != 0: FAIL.append('empty events render wrong')
    for tl in [0.0, 0.001, 0.5, 10.0]:
        try:
            x = D.render_kit([(0.1, 'kick', 1.0), (0.3, 'crash1', 1.0)], total_len=tl)
            if x.shape != (int(round(tl * E.SR)), 2): FAIL.append(f'total_len {tl}: shape {x.shape}')
            if not np.isfinite(x).all(): FAIL.append(f'total_len {tl}: NaN')
            print(f'total_len {tl:6.3f} -> shape {x.shape} pk {np.abs(x).max() if len(x) else 0:.3f}')
        except Exception as e: FAIL.append(f'total_len {tl}: {type(e).__name__}: {e}')
    for v in [0.0, -0.5, 1.0, 1.5, 3.0]:
        y = D.hit('snare', v); pk = np.abs(y).max()
        if not np.isfinite(y).all(): FAIL.append(f'hit vel {v} NaN')
        if pk > 0.51: FAIL.append(f'hit vel {v}: peak {pk:.3f} > 0.51 (vel must clip at 1)')
        print(f'hit snare vel {v:4.1f}: pk {pk:.3f}')
    a = D.render_kit([(0.1, 'kick', 1.0)], total_len=1.0); b = D.render_kit([(0.1, 'kick', 1.0), (0.1, 'kick', 1.0)], total_len=1.0)
    c = D.render_kit([(0.1, 'crash1', 1.0)], total_len=2.0); d = D.render_kit([(0.1, 'crash1', 1.0)] * 3, total_len=2.0)
    print(f'duplicate events: kick x1 pk {np.abs(a).max():.3f} x2 pk {np.abs(b).max():.3f}; crash x1 {np.abs(c).max():.3f} x3 {np.abs(d).max():.3f}')
    if np.abs(b).max() > np.abs(a).max() * 1.05 or np.abs(d).max() > np.abs(c).max() * 1.05:
        FAIL.append('duplicate same-time events stack instead of being merged')
    ev = [[np.float64(0.1), 'kick', np.float32(0.9)], (np.int64(1), 'snare', 1), (-0.2, 'kick', 1.0), (5.0, 'china', 1.0)]
    x = D.render_kit(ev, total_len=2.0)
    if x.shape != (96000, 2) or not np.isfinite(x).all(): FAIL.append('mixed-type events')
    print('mixed-type / negative-t / beyond-total_len events: ok')

    # ------------------------------------------------------------------ per-piece spectra at vel 1.0 and 0.2
    print('\n--- one-shots (hit) at vel 1.0 / 0.2 ---')
    one = {}
    for nm in D.NAMES:
        for v in (1.0, 0.2):
            y = D.hit(nm, v, 0); one[(nm, v)] = y
            pk, rms, b = basic(f'hit {nm} v{v}', y, pk_lo=0.03 if v < 1 else 0.3, pk_hi=0.51)
            if nm in ('hat_closed', 'hat_open', 'hat_pedal', 'ride', 'ride_bell', 'crash1', 'crash2', 'china', 'splash'):
                if b[8] < -15: FAIL.append(f'{nm} v{v}: 8-16k band {b[8]:.1f} dB < -15 dB rel 250-500 (too dark)')
                if b[7] < -10: FAIL.append(f'{nm} v{v}: 4-8k band {b[7]:.1f} dB (too dark)')
                if y.ndim != 2: FAIL.append(f'{nm}: expected stereo')
            else:
                if y.ndim != 1: FAIL.append(f'{nm}: expected mono')
    # kick: sub + click
    for v in (1.0, 0.6, 0.2):
        y = one.get(('kick', v)) if ('kick', v) in one else D.hit('kick', v); b = bands_db(y)
        f0 = fundamental_hz(y, 35, 120)
        hi = E.hp(E.hp(y, 1500), 1500); lo = E.lp(E.lp(y, 400), 400); ratio = np.abs(hi).max() / np.abs(lo).max()
        print(f'kick v{v}: sub peak {f0:.0f} Hz, 60-120 {b[1]:+.1f} dB, 2-4k {b[6]:+.1f} 4-8k {b[7]:+.1f} dB rel 250-500; click/body peak ratio {ratio:.2f}')
        if not (45 <= f0 <= 90): FAIL.append(f'kick v{v}: sub at {f0:.0f} Hz')
        if b[1] < 10: FAIL.append(f'kick v{v}: no sub (60-120 {b[1]:.1f} dB)')
        if v >= 0.6 and (b[6] < -11 or b[7] < -11): FAIL.append(f'kick v{v}: no click (2-4k {b[6]:.1f}, 4-8k {b[7]:.1f})')
        if v >= 0.6 and ratio < 1.0: FAIL.append(f'kick v{v}: click/body ratio {ratio:.2f} weak')
        if v < 0.6 and ratio < 0.7: FAIL.append(f'kick v{v}: click/body ratio {ratio:.2f} weak')
    # snare: crack 2-4 kHz
    for v in (1.0, 0.5, 0.2):
        y = D.hit('snare', v); b = bands_db(y); f0 = fundamental_hz(y, 120, 300)
        print(f'snare v{v}: fundamental {f0:.0f} Hz, 2-4k {b[6]:+.1f} 4-8k {b[7]:+.1f} 8-16k {b[8]:+.1f} dB rel 250-500')
        if b[6] < -8: FAIL.append(f'snare v{v}: 2-4k crack {b[6]:.1f} dB too weak')
    # toms: pitched + distinct
    f_toms = {}
    for nm in ('tom_hi', 'tom_mid', 'tom_floor'):
        y = D.hit(nm, 1.0); f_toms[nm] = fundamental_hz(y, 50, 300)
        # tonal-ness: ratio of peak to median in 50-300 Hz
        m = E.mono(y)[:int(0.3 * E.SR)]; F = np.abs(np.fft.rfft(m * np.hanning(len(m)), 1 << 16)); f = np.fft.rfftfreq(1 << 16, 1 / E.SR)
        sel = (f >= 50) & (f <= 300); tonal = 20 * np.log10(F[sel].max() / (np.median(F[sel]) + 1e-12))
        print(f'{nm}: fundamental {f_toms[nm]:.0f} Hz, peak/median in 50-300 Hz {tonal:.1f} dB (tonal if > 15)')
        if tonal < 15: FAIL.append(f'{nm}: not clearly pitched ({tonal:.1f} dB)')
    if not (f_toms['tom_hi'] > f_toms['tom_mid'] * 1.15 > f_toms['tom_floor'] * 1.15 ** 2 / 1.15):
        FAIL.append(f'toms not ordered in pitch: {f_toms}')
    # tom spread in the kit
    th = D.render_kit([(0.1, 'tom_hi', 1.0)], total_len=1.5); tf = D.render_kit([(0.1, 'tom_floor', 1.0)], total_len=1.5)
    print(f'tom spread: tom_hi L/R {lr_db(th):+.1f} dB, tom_floor L/R {lr_db(tf):+.1f} dB')
    if lr_db(th) < 1.5 or lr_db(tf) > -1.5: FAIL.append('toms not spread L/R')
    # cymbals in the kit at low velocity (after bus processing)
    for nm, v in (('crash1', 0.2), ('hat_closed', 0.2), ('ride', 0.3), ('china', 0.5)):
        x = D.render_kit([(0.1, nm, v)], total_len=2.0); b = bands_db(x)
        print(f'kit {nm} v{v}: pk {np.abs(x).max():.3f} 4-8k {b[7]:+.1f} 8-16k {b[8]:+.1f} dB rel 250-500')
        if b[8] < -15: FAIL.append(f'kit {nm} v{v}: 8-16k {b[8]:.1f} dB (too dark)')
    # velocity monotonic: level grows with vel for kick, snare, crash
    for nm in ('kick', 'snare', 'crash1', 'hat_closed'):
        pks = [np.abs(D.hit(nm, v)).max() for v in np.linspace(0.2, 1.0, 9)]
        if np.any(np.diff(pks) <= 0): FAIL.append(f'{nm}: peak not monotonic in vel: {np.round(pks, 3)}')
    print('velocity monotonic: ok' if not any('monotonic' in f for f in FAIL) else 'velocity monotonic: FAIL')

    # ------------------------------------------------------------------ 300-event stress render
    print('\n--- stress render ---')
    ev = stress_events(4)
    print(f'{len(ev)} events; names: {sorted(set(e[1] for e in ev))}')
    t0 = time.time(); kit = D.render_kit(ev); dt = time.time() - t0
    pk, rms, b = basic(f'stress kit ({len(ev)} ev, total_len=None)', kit, pk_lo=0.3, pk_hi=0.6)
    exp_len = int(round((max(e[0] for e in ev) + 3.5) * E.SR))
    if len(kit) != exp_len: FAIL.append(f'length {len(kit)} != last+3.5 s ({exp_len})')
    print(f'render time {dt:.2f} s for {len(kit)/E.SR:.2f} s audio; L/R rms diff {lr_db(kit):+.2f} dB; '
          f'max |L-R| {np.abs(kit[:,0]-kit[:,1]).max():.3f}; corr(L,R) {np.corrcoef(kit[:,0], kit[:,1])[0,1]:.3f}')
    if np.abs(kit[:, 0] - kit[:, 1]).max() < 0.02: FAIL.append('L/R identical')
    if dt > 20: FAIL.append(f'stress render too slow {dt:.1f}s')
    at_ceiling = (np.abs(kit).max(1) >= 0.562 * 0.995).mean() * 100
    print(f'limiter engagement: {at_ceiling:.4f}% of samples at ceiling')
    # per-downbeat stack peaks: crash+china+kick(+crash2) must not be squashed vs a lone kick+snare
    for bn in range(5):
        i = int(round(bn * 4 * BEAT * E.SR)); seg = kit[i:i + int(0.05 * E.SR)]
        print(f'  downbeat {bn+1}: stack peak {np.abs(seg).max():.3f}')
    # 32nd snare roll separation (beat 4 of bar 1): envelope dips between hits
    env = 20 * np.log10(np.convolve(np.abs(E.mono(kit)), np.ones(96) / 96, 'same') + 1e-9)
    hits = [int(round((3 * BEAT + k * BEAT / 8) * E.SR)) for k in range(8)]
    dips = []
    for h0, h1 in zip(hits[:-1], hits[1:]):
        seg = env[h0:h1]; dips.append(seg.max() - seg[int(len(seg) * 0.5):-int(0.004 * E.SR)].min())
    print(f'32nd snare roll: envelope dip between hits min {min(dips):.1f} dB mean {np.mean(dips):.1f} dB (want >= 6)')
    if min(dips) < 6: FAIL.append(f'32nd snare roll smears: min dip {min(dips):.1f} dB')
    # 16th double kick separation inside the FULL mix (bar 2, beats 1-2: kick 16ths under hats/snare)
    hits = [int(round((4 * BEAT + k * BEAT / 4) * E.SR)) for k in range(0, 8)]
    lowenv = 20 * np.log10(np.convolve(np.abs(E.mono(E.lp(E.lp(kit, 150), 150))), np.ones(96) / 96, 'same') + 1e-9)
    dips = []
    for h0, h1 in zip(hits[:-1], hits[1:]):
        seg = lowenv[h0:h1]; dips.append(seg.max() - seg[int(len(seg) * 0.5):-int(0.004 * E.SR)].min())
    print(f'16th double kick in full mix (<150 Hz env): dip min {min(dips):.1f} dB mean {np.mean(dips):.1f} dB (want >= 10)')
    if min(dips) < 10: FAIL.append(f'double kick smears in mix: {min(dips):.1f} dB')
    E.write(os.path.join(HERE, 'drums_stress.wav'), kit)
    p = spectrogram_png(kit, os.path.join(HERE, 'drums_stress_spec.png'),
                        f'render_kit stress: {len(ev)} events, 4 bars @110 (dbl-kick 16ths, 32nd snare rolls, crash+china stacks, vel 0.2-1)')
    print('spectrogram:', p)
    # total_len given: shorter and longer
    k2 = D.render_kit(ev, total_len=5.0); k3 = D.render_kit(ev, total_len=14.0)
    if len(k2) != 5 * E.SR or len(k3) != 14 * E.SR: FAIL.append('total_len not respected on stress events')
    if not np.isfinite(k2).all() or not np.isfinite(k3).all(): FAIL.append('NaN with total_len')
    print(f'total_len=5 -> {k2.shape} pk {np.abs(k2).max():.3f}; total_len=14 -> {k3.shape} pk {np.abs(k3).max():.3f}')
    # determinism
    k4 = D.render_kit(ev)
    print(f'deterministic across calls: max diff {np.abs(k4 - kit).max():.2e}')
    if np.abs(k4 - kit).max() > 1e-9: FAIL.append('render_kit not deterministic')
    # 300 events spread over the whole song length (sparse) -> timing + length
    ev_long = stress_events(4) + stress_events(4, start=60.0) + stress_events(4, start=100.0)
    t0 = time.time(); song = D.render_kit(ev_long, total_len=113.5); dt = time.time() - t0
    basic(f'sparse song ({len(ev_long)} ev, 113.5 s)', song, pk_lo=0.3, pk_hi=0.6)
    print(f'113.5 s render: {dt:.1f} s')
    if dt > 60: FAIL.append(f'113.5 s render too slow: {dt:.1f}s')

    print(f'\ntotal test time {time.time()-T0:.1f} s')
    print('RESULT:', 'PASS' if not FAIL else 'FAIL'); [print('  -', f) for f in FAIL]
    sys.exit(1 if FAIL else 0)
