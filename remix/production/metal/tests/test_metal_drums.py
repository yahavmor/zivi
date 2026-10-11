"""Smoke test for metal_drums.py: NaN/DC/clipping checks, peak/RMS, band energies, spectrogram PNGs,
double-bass tightness measurement, and a full-song timing benchmark.
Run: python3 tests/test_metal_drums.py   (from the metal folder; writes PNG/WAV into tests/)"""
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


def bands_db(x):
    x = E.mono(x); f, P = signal.welch(x, E.SR, nperseg=4096)
    e = [P[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]
    ref = e[3] + 1e-20
    return [10 * np.log10(v / ref + 1e-20) for v in e]


def check(name, x, pk_lo=0.2, pk_hi=0.62):
    x = np.asarray(x); m = E.mono(x)
    pk = np.abs(x).max(); rms = np.sqrt((m ** 2).mean()); dc = abs(m.mean()) / (rms + 1e-12)
    ok = True
    if not np.isfinite(x).all(): FAIL.append(f'{name}: NaN/inf'); ok = False
    if dc > 0.02: FAIL.append(f'{name}: DC offset {dc:.3f} of RMS'); ok = False
    if pk > pk_hi: FAIL.append(f'{name}: peak {pk:.3f} > {pk_hi}'); ok = False
    if pk < pk_lo: FAIL.append(f'{name}: peak {pk:.3f} < {pk_lo}'); ok = False
    if pk >= 0.999: FAIL.append(f'{name}: clipping'); ok = False
    b = bands_db(x)
    print(f'{name:22s} pk={pk:5.3f} rms={20*np.log10(rms+1e-12):6.1f}dB len={len(x)/E.SR:5.2f}s dc={dc:.4f} '
          f'{"OK " if ok else "BAD"} | ' + ' '.join(f'{v:6.1f}' for v in b))
    return pk, rms, b


def spectrogram_png(x, path, title, fmax=16000):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    m = E.mono(x); f, t, S = signal.spectrogram(m, E.SR, nperseg=2048, noverlap=1536, window='hann')
    S = 10 * np.log10(S + 1e-14); top = S.max()
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=90)
    im = ax.pcolormesh(t, f, S, vmin=top - 80, vmax=top, shading='auto', cmap='magma')
    ax.set_yscale('symlog', linthresh=200); ax.set_ylim(20, fmax); ax.set_ylabel('Hz'); ax.set_xlabel('s')
    ax.set_yticks([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_yticklabels(['30', '60', '120', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    ax.set_title(title); fig.colorbar(im, ax=ax, label='dB'); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def envelope_db(x, win=0.002):
    m = np.abs(E.mono(x)); k = max(1, int(win * E.SR))
    e = np.convolve(m, np.ones(k) / k, 'same'); return 20 * np.log10(e + 1e-9)


if __name__ == '__main__':
    print('bands (dB rel 250-500):', ' '.join(f'{l:>6s}' for l in LABELS))
    # ---------------- one-shots ----------------
    one = {}
    for nm in D.NAMES:
        y = D.hit(nm, 1.0, 0); one[nm] = y; check(f'hit {nm} v1.0', y)
    for nm in ['kick', 'snare', 'hat_closed', 'crash1']:
        y = D.hit(nm, 0.4, 1); check(f'hit {nm} v0.4', y, pk_lo=0.05)
    # velocity: quieter AND darker
    bk1, bk4 = bands_db(D.hit('snare', 1.0)), bands_db(D.hit('snare', 0.4))
    hi_drop = (bk4[7] - bk4[3]) - (bk1[7] - bk1[3])
    print(f'snare v0.4 vs v1.0: peak ratio {np.abs(D.hit("snare",0.4)).max()/np.abs(D.hit("snare",1.0)).max():.2f}, '
          f'4-8k band change {hi_drop:+.1f} dB (expect negative)')
    if hi_drop > -1.0: FAIL.append('velocity-to-timbre: soft snare not darker')
    # round robin: variants differ
    a, b = D.hit('snare', 1.0, 0), D.hit('snare', 1.0, 1)
    m = min(len(a), len(b)); diff = np.abs(a[:m] - b[:m]).max() / np.abs(a).max()
    print(f'round-robin snare seed0 vs seed1: max diff {diff:.2f} of peak (expect > 0.05)')
    if diff < 0.05: FAIL.append('round robin: no variation')
    # montage spectrogram of one-shots
    mont = []
    for nm in ['kick', 'snare', 'snare_ghost', 'tom_hi', 'tom_mid', 'tom_floor', 'hat_closed', 'hat_open', 'ride',
               'ride_bell', 'crash1', 'china', 'splash']:
        y = E.mono(one[nm]); y = y[:int(0.6 * E.SR)]; mont.append(np.concatenate([y, np.zeros(int(0.1 * E.SR))]))
    mont = np.concatenate(mont)
    p = spectrogram_png(mont, os.path.join(HERE, 'drums_oneshots_spec.png'),
                        'one-shots (0.6 s each): kick snare ghost tomH tomM tomF hatC hatO ride bell crash china splash')
    print('spectrogram:', p)
    E.write(os.path.join(HERE, 'drums_oneshots.wav'), mont)

    # ---------------- kit render ----------------
    ev = D.demo_events(bars=2)
    t0 = time.time(); kit = D.render_kit(ev); dt = time.time() - t0
    pk, rms, b = check('render_kit demo 2 bars', kit, pk_lo=0.3, pk_hi=0.6)
    print(f'render_kit: {len(ev)} events, {len(kit)/E.SR:.2f} s audio in {dt:.2f} s; stereo L/R rms diff '
          f'{20*np.log10(np.sqrt((kit[:,0]**2).mean())/np.sqrt((kit[:,1]**2).mean())):+.2f} dB')
    E.write(os.path.join(HERE, 'drums_kit_demo.wav'), kit)
    p = spectrogram_png(kit, os.path.join(HERE, 'drums_kit_demo_spec.png'), 'render_kit demo: 2 bars @110, double-bass 16ths, 32nd fill')
    print('spectrogram:', p)
    # stereo: hats should lean left, ride right
    hats = D.render_kit([(0.1 + k * 0.25, 'hat_closed', 0.8) for k in range(8)])
    ride = D.render_kit([(0.1 + k * 0.25, 'ride', 0.8) for k in range(8)])
    lr = lambda x: 20 * np.log10((np.sqrt((x[:, 0] ** 2).mean()) + 1e-9) / (np.sqrt((x[:, 1] ** 2).mean()) + 1e-9))
    print(f'panning: hats L/R {lr(hats):+.1f} dB (expect >0), ride L/R {lr(ride):+.1f} dB (expect <0)')
    if lr(hats) < 1 or lr(ride) > -1: FAIL.append('panning wrong')

    # hat-open choke: closed hat at 0.35 s must kill the open hat's tail
    opn = D.render_kit([(0.1, 'hat_open', 0.9)]); chk = D.render_kit([(0.1, 'hat_open', 0.9), (0.35, 'hat_closed', 0.6)])
    seg = slice(int(0.5 * E.SR), int(0.8 * E.SR))
    choke_db = 20 * np.log10(np.sqrt((chk[seg] ** 2).mean()) / np.sqrt((opn[seg] ** 2).mean()))
    print(f'hat-open choke: 0.5-0.8 s energy with choke vs without: {choke_db:+.1f} dB (expect <= -10)')
    if choke_db > -10: FAIL.append('hat open not choked')

    # ---------------- double-bass tightness ----------------
    beat = 60 / 110
    for label, step in [('16ths', beat / 4), ('32nds', beat / 8)]:
        evk = [(0.2 + k * step, 'kick', 0.9) for k in range(16)]
        xk = D.render_kit(evk); env = envelope_db(xk)
        hits = [int((0.2 + k * step) * E.SR) for k in range(16)]
        dips = []
        for h0, h1 in zip(hits[:-1], hits[1:]):
            seg = env[h0:h1]; pkv = seg.max(); mn = seg[int(len(seg) * 0.5):-int(0.004 * E.SR)].min(); dips.append(pkv - mn)
        print(f'double-bass {label}: envelope dip between hits min {min(dips):.1f} dB, mean {np.mean(dips):.1f} dB (want >= 12 dB)')
        if min(dips) < 12: FAIL.append(f'double bass {label} smears: dip {min(dips):.1f} dB')
        if label == '16ths':
            E.write(os.path.join(HERE, 'drums_doublebass16.wav'), xk)
            spectrogram_png(xk, os.path.join(HERE, 'drums_doublebass16_spec.png'), 'kick only, 16ths @110 BPM')

    # ---------------- full-song timing benchmark ----------------
    bars = 52; ev = []
    for b_ in range(bars): ev += D.demo_events(bars=1, start=b_ * 4 * beat)
    t0 = time.time(); song = D.render_kit(ev, total_len=113.5); dt = time.time() - t0
    pk2, rms2, b2 = check(f'render_kit full song ({len(ev)} ev)', song, pk_lo=0.3, pk_hi=0.6)
    print(f'full song: {len(song)/E.SR:.1f} s rendered in {dt:.1f} s (budget 120 s)')
    at_ceiling = (np.abs(song).max(1) > pk2 * 0.98).mean() * 100
    print(f'limiter engagement: {at_ceiling:.4f}% of samples within 0.2 dB of the peak (small = limiter is only a safety)')
    for a, b_, lab in [(0, 8.73, 'bars 1-4'), (50, 58.7, 'bars 24-27'), (100, 108.7, 'bars 47-50')]:
        seg = song[int(a * E.SR):int(b_ * E.SR)]
        print(f'  {lab}: peak {np.abs(seg).max():.3f} rms {20*np.log10(np.sqrt((seg**2).mean())):.1f} dB')
    if dt > 120: FAIL.append(f'too slow: {dt:.1f}s')
    if len(song) != int(113.5 * E.SR): FAIL.append('total_len not respected')

    print('\nRESULT:', 'PASS' if not FAIL else 'FAIL'); [print('  -', f) for f in FAIL]
    sys.exit(1 if FAIL else 0)
