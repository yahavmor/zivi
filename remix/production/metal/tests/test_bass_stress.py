"""Adversarial stress test for metal_bass (the song actually plays midi 24-34, i.e. C1 = 32.7 Hz).

Renders: muted 16ths + 32nds at 110 BPM on C1 with vel 0.3..1.0, legato 16ths (dur == gap, as in the
riff JSON), long open C1/G1 notes, slides 29->24 / 24->31 / 32->24, an overlapping pair, a dyad.
Measures: NaN/DC/clip, band energies, sub-band purity (low band distortion leakage above the crossover),
mid-band grit share 700-2k, inter-note dips (smear), harmonic decay vs frequency, pitch of C1 and of the
slide, per-note peak spread, cache growth, render time, and writes spectrograms.
Run: python3 tests/test_bass_stress.py [--full] [--module metal_bass]
"""
import os, sys, time, tracemalloc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import importlib
import numpy as np
import soundfile as sf
from scipy import signal
import engine as E

MOD = sys.argv[sys.argv.index('--module') + 1] if '--module' in sys.argv else 'metal_bass'
B = importlib.import_module(MOD)
SR = E.SR
BEAT = 60 / 110; S16 = BEAT / 4; S32 = BEAT / 8
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def band_energies(x, nperseg=16384):
    f, p = signal.welch(x, SR, nperseg=nperseg)
    ref = p[(f >= 250) & (f < 500)].sum()
    return {(lo, hi): 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / ref + 1e-20) for lo, hi in BANDS}


def fmt_bands(be):
    return '  '.join(f'{lo}-{hi}:{v:+5.1f}' for (lo, hi), v in be.items())


def stats(name, x):
    assert not np.isnan(x).any(), name + ' NaN'
    assert not np.isinf(x).any(), name + ' inf'
    peak = np.abs(x).max(); rms = np.sqrt(np.mean(x ** 2)); dc = x.mean()
    fw, pw = signal.welch(x, SR, nperseg=65536); lf_db = 10 * np.log10(pw[fw < 20].sum() * (fw[1] - fw[0]) + 1e-20)  # true <20 Hz power
    print(f'[{name}] len={len(x)/SR:.2f}s peak={peak:.3f} ({20*np.log10(peak+1e-12):.1f} dBFS) rms={rms:.3f} '
          f'({20*np.log10(rms+1e-12):.1f} dBFS) crest={20*np.log10(peak/(rms+1e-12)):.1f} dB DC={dc:+.5f} '
          f'<20Hz={lf_db:.1f} dBFS clipped={(np.abs(x) > 0.99).sum()}')
    print('   bands rel 250-500: ' + fmt_bands(band_energies(x)))
    return peak, rms, dc


def env_db(x, win=0.005):
    """RMS envelope in dB, hop = win."""
    n = int(win * SR); m = len(x) // n
    e = np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(1))
    return 20 * np.log10(e + 1e-9)


def stress_events():
    ev = []; t = 0.0; marks = {}
    # A: 2 bars of muted 16ths on C1, vel pattern 0.3..1.0, 75% gate
    marks['mute16'] = t
    vels = [1.0, 0.5, 0.7, 0.4, 0.9, 0.3, 0.8, 0.6] * 4
    for i in range(32):
        ev.append((t + i * S16, 24, S16 * 0.75, vels[i], None, True))
    t += 32 * S16
    # B: 1 bar of legato muted 16ths (dur == gap, as in riffs json), vel 0.8
    marks['legato16'] = t
    for i in range(16):
        ev.append((t + i * S16, 24, S16, 0.8, None, True))
    t += 16 * S16
    # C: 1 bar of muted 32nds on C1 (dur 0.068 as in json)
    marks['mute32'] = t
    for i in range(32):
        ev.append((t + i * S32, 24, S32, 0.9 if i % 4 == 0 else 0.7, None, True))
    t += 32 * S32
    # D: long open C1 (4 s), then open G1 2 s at vel 0.5
    marks['openC1'] = t
    ev.append((t, 24, 4.0, 1.0, None, False)); t += 4.3
    marks['openG1'] = t
    ev.append((t, 31, 2.0, 0.5, None, False)); t += 2.3
    # E: slides
    marks['slide29_24'] = t
    ev.append((t, 24, 1.0, 0.9, 29, False)); t += 1.2
    marks['slide24_31'] = t
    ev.append((t, 31, 0.6, 0.9, 24, False)); t += 0.8
    marks['slide32_24'] = t
    ev.append((t, 24, 0.6, 0.9, 32, False)); t += 0.8
    # F: overlapping pair (C1 1.0 s, G1 enters at +0.5 for 1.0 s) and a dyad (24 + 31 same onset)
    marks['overlap'] = t
    ev.append((t, 24, 1.0, 0.9, None, False)); ev.append((t + 0.5, 31, 1.0, 0.9, None, False)); t += 1.8
    marks['dyad'] = t
    ev.append((t, 24, 0.5, 0.9, None, True)); ev.append((t, 31, 0.5, 0.9, None, True)); t += 0.8
    # G: velocity sweep, open C1 8th notes
    marks['velsweep'] = t
    for i, v in enumerate(np.linspace(0.3, 1.0, 8)):
        ev.append((t + i * 2 * S16, 24, 2 * S16 * 0.9, float(v), None, False))
    t += 16 * S16
    # H: open 16ths on G1 (ringing, 100% gate) and open C1
    marks['open16'] = t
    for i in range(8):
        ev.append((t + i * S16, 31 if i < 4 else 24, S16, 0.85, None, False))
    t += 8 * S16 + 0.5
    marks['end'] = t
    return ev, marks


def full_song_events():
    ev = []; roots = [24, 24, 27, 29, 31, 32, 34, 24]; t = 0.0; bar = 0
    while t < 112.0:
        root = roots[bar % len(roots)]
        if bar % 4 == 3:
            ev.append((t, root, BEAT * 3.5, 0.9, root - 2, False))
        else:
            for i in range(16):
                ev.append((t + i * S16, root, S16 * (0.75 if i % 8 != 7 else 1.0), 0.8 + 0.2 * (i % 4 == 0), None, i % 8 != 7))
        t += 4 * BEAT; bar += 1
    return ev


def spectrogram_png(x, path, title, fmax=8000):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    f, tt, S = signal.spectrogram(x, SR, nperseg=4096, noverlap=3072, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].plot(np.arange(len(x)) / SR, x, lw=0.3, color='#444'); ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(-0.7, 0.7)
    ax[0].set_title(title)
    sel = f <= fmax
    im = ax[1].pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 90, vmax=Sdb.max())
    ax[1].set_yscale('symlog', linthresh=100); ax[1].set_ylim(20, fmax)
    ax[1].set_yticks([30, 50, 65, 100, 150, 250, 500, 1000, 2000, 4000, 8000])
    ax[1].set_yticklabels(['30', '50', '65', '100', '150', '250', '500', '1k', '2k', '4k', '8k'])
    ax[1].set_xlabel('s'); fig.colorbar(im, ax=ax[1], label='dB'); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def harmonic_table(x, f0, kmax, t0, t1):
    seg = x[int(t0 * SR):int(t1 * SR)]; seg = seg * np.hanning(len(seg))
    nfft = 1 << 19; F = np.abs(np.fft.rfft(seg, nfft)); f = np.fft.rfftfreq(nfft, 1 / SR)
    out = []
    for h in range(1, kmax + 1):
        sel = (f > f0 * h - f0 * 0.25) & (f < f0 * h + f0 * 0.25)
        out.append(20 * np.log10(F[sel].max() + 1e-12))
    return np.array(out)


def f0_autocorr(x, t, win=0.02, fmin=25, fmax=130):
    seg = x[int((t - win / 2) * SR):int((t + win / 2) * SR)]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    lo, hi = int(SR / fmax), int(SR / fmin)
    lag = lo + np.argmax(ac[lo:hi]); return SR / lag


def decay_rates(x, t0, t1, bands):
    """dB/s slope of the band envelope between t0 and t1 (linear fit on 50 ms RMS frames)."""
    out = {}
    for lo, hi in bands:
        sos = signal.butter(4, [lo, hi], btype='band', fs=SR, output='sos')
        y = signal.sosfiltfilt(sos, x[int(t0 * SR):int(t1 * SR)])
        e = env_db(y, 0.05); tt = np.arange(len(e)) * 0.05
        k = e > e.max() - 45                     # fit only the part above the noise floor
        k[:2] = False                            # skip the attack frames
        if k.sum() >= 4:
            slope = np.polyfit(tt[k], e[k], 1)[0]
        else:
            slope = np.nan
        out[(lo, hi)] = slope
    return out


if __name__ == '__main__':
    ev, marks = stress_events()
    tracemalloc.start()
    t0 = time.time(); di = B.render_di(ev); y = B.render_bass(ev); dt = time.time() - t0
    cur, pk_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f'== module {MOD}: stress render {len(ev)} notes, {len(y)/SR:.1f}s audio in {dt:.2f}s, '
          f'cache entries {len(getattr(B, "_cache", {}))}, cache MB {sum(v[0].nbytes for v in getattr(B, "_cache", {}).values())/1e6:.1f}, '
          f'peak traced mem {pk_mem/1e6:.0f} MB')
    din = di / (np.abs(di).max() + 1e-9) * 0.5
    stats('DI(norm)', din)
    peak, rms, dc = stats('FINAL', y)
    assert 0.3 <= peak <= 0.6, 'peak out of range'
    assert abs(dc) < 1e-3, 'DC'
    assert (np.abs(y) > 0.99).sum() == 0, 'clipping'
    low = B.low_band(di); mid = B.mid_band(di)
    print(f'-- low band: rms {20*np.log10(np.sqrt(np.mean(low**2))):.1f} dBFS  bands ' + fmt_bands(band_energies(low)))
    print(f'-- mid band: rms {20*np.log10(np.sqrt(np.mean(mid**2))):.1f} dBFS  bands ' + fmt_bands(band_energies(mid)))

    # ---- sub-band purity: low band distortion leakage above the crossover (sustained C1 and G1)
    for name, midi, vel in (('C1', 24, 1.0), ('G1', 31, 0.5), ('C2', 36, 0.9)):
        d1 = B.render_di([(0.0, midi, 1.5, vel, None, False)], total_len=1.8)
        lin = E.lp(d1, B.CROSS_HZ, order=2) * B.LOW_PRE * B.LOW_GAIN
        lb = B.low_band(d1)
        f, pl = signal.welch(lin[int(0.1 * SR):int(1.4 * SR)], SR, nperseg=32768)
        f, pb = signal.welch(lb[int(0.1 * SR):int(1.4 * SR)], SR, nperseg=32768)
        def leak(p): return 10 * np.log10(p[(f >= 200) & (f < 2000)].sum() / p[(f >= 20) & (f < 150)].sum())
        f0 = E.midi2hz(midi)
        hl = harmonic_table(lin, f0, 12, 0.1, 1.0); hb = harmonic_table(lb, f0, 12, 0.1, 1.0)
        print(f'-- sub purity {name} vel {vel}: energy 200-2k rel 20-150: linear ref {leak(pl):.1f} dB | low_band {leak(pb):.1f} dB '
              f'(added {leak(pb)-leak(pl):+.1f} dB); harmonics h1..h12 rel h1: lin {np.round(hl-hl[0],1)} | lowband {np.round(hb-hb[0],1)}')
        # mid band grit at this pitch: 700-2k share
        mb = B.mid_band(d1); f, pm = signal.welch(mb[int(0.03 * SR):int(0.6 * SR)], SR, nperseg=16384)
        tot = pm[(f >= 100)].sum()
        print(f'   mid band {name}: share 150-700 {10*np.log10(pm[(f>=150)&(f<700)].sum()/tot):.1f} dB, 700-2k '
              f'{10*np.log10(pm[(f>=700)&(f<2000)].sum()/tot):.1f} dB, 2k-4k {10*np.log10(pm[(f>=2000)&(f<4000)].sum()/tot):.1f} dB, '
              f'>4k {10*np.log10(pm[(f>=4000)].sum()/tot):.1f} dB; mid/low rms {20*np.log10(np.sqrt(np.mean(mb**2))/np.sqrt(np.mean(lb**2))):+.1f} dB')
        yy = B.render_bass([(0.0, midi, 1.5, vel, None, False)], total_len=1.8)
        f, pf = signal.welch(yy[int(0.03 * SR):int(1.0 * SR)], SR, nperseg=16384); tot = pf.sum()
        print(f'   final {name}: 700-2k share {10*np.log10(pf[(f>=700)&(f<2000)].sum()/tot):.1f} dB, <150 share {10*np.log10(pf[f<150].sum()/tot):.1f} dB, '
              f'150-700 {10*np.log10(pf[(f>=150)&(f<700)].sum()/tot):.1f} dB, peak {np.abs(yy).max():.3f} rms {20*np.log10(np.sqrt(np.mean(yy[:int(1.0*SR)]**2))):.1f} dBFS')

    # ---- smear: inter-note dips (final and DI). "15 ms" = floor over the 15 ms before the next onset,
    # "last 5 ms" = the 5 ms frame right before it (the choke acts 5 ms before the onset, so a 100 % gate
    # note can only dip in that last frame).
    e = env_db(y, 0.005); ed = env_db(di, 0.005)
    def dips(t_start, n, step, label):
        d15, d5, dd5 = [], [], []
        for i in range(1, n):
            on = t_start + i * step
            i_on = int(np.floor(on / 0.005 + 1e-9))   # frame i_on contains the onset; i_on-1 ends before it
            pk = e[int((on - step) / 0.005):i_on].max(); pkd = ed[int((on - step) / 0.005):i_on].max()
            d15.append(pk - e[max(0, i_on - 3):i_on].min()); d5.append(pk - e[i_on - 1]); dd5.append(pkd - ed[i_on - 1])
        d15, d5, dd5 = map(np.array, (d15, d5, dd5))
        print(f'-- dips {label}: final 15ms min {d15.min():.1f} med {np.median(d15):.1f} dB | final last-5ms min {d5.min():.1f} '
              f'med {np.median(d5):.1f} dB | DI last-5ms min {dd5.min():.1f} med {np.median(dd5):.1f} dB')
        return d15
    d16 = dips(marks['mute16'], 32, S16, 'muted 16ths C1 (75% gate)')
    dl = dips(marks['legato16'], 16, S16, 'legato muted 16ths C1 (dur == gap)')
    d32 = dips(marks['mute32'], 32, S32, 'muted 32nds C1')
    do = dips(marks['open16'], 8, S16, 'open 16ths G1/C1 (dur == gap)')
    # definitive onset cleanliness: level in the 1 ms right before each onset vs the note's peak (1 ms frames);
    # the final chain blurs ~3 ms (limiter look-ahead + LR4 ringing), so the final is read 2 ms wide.
    e1 = env_db(di, 0.001); e2 = env_db(y, 0.002)
    for label, t_start, n, step in (('muted 16ths', marks['mute16'], 32, S16), ('legato mutes', marks['legato16'], 16, S16),
                                    ('muted 32nds', marks['mute32'], 32, S32), ('open 16ths 100% gate', marks['open16'], 8, S16)):
        cd, cf = [], []
        for i in range(1, n):
            on = t_start + i * step
            j1 = int(np.floor(on / 0.001 + 1e-9)); j2 = int(np.floor(on / 0.002 + 1e-9))
            cd.append(e1[int((on - step) / 0.001):j1].max() - e1[j1 - 1]); cf.append(e2[int((on - step) / 0.002):j2].max() - e2[j2 - 1])
        print(f'-- onset cleanliness {label}: DI last-1ms min {min(cd):.1f} med {np.median(cd):.1f} dB | final last-2ms min {min(cf):.1f} med {np.median(cf):.1f} dB')
    # muted chug tightness: time from peak to -20/-40 dB for a single C1 mute
    ym = B.render_bass([(0.0, 24, S16 * 0.75, 0.9, None, True)], total_len=1.0)
    em = env_db(ym, 0.002); ip = em.argmax()
    t20 = np.argmax(em[ip:] < em[ip] - 20) * 2; t40 = np.argmax(em[ip:] < em[ip] - 40) * 2
    print(f'-- C1 mute chug (dur {S16*0.75*1000:.0f} ms): -20 dB at {t20} ms, -40 dB at {t40} ms after peak')

    # ---- pitch: C1 fundamental and the slides (DI)
    f0 = f0_autocorr(di, marks['openC1'] + 0.3, win=0.1); print(f'-- open C1 pitch {f0:.2f} Hz (expect 32.70, {1200*np.log2(f0/32.703):+.1f} c)')
    for name, fa, fb in (('slide29_24', 61.74, 32.70), ('slide24_31', 32.70, 49.00), ('slide32_24', 51.91, 32.70)):
        ts = marks[name]; row = []
        for ms in (130, 200, 400):            # after the 60 ms glide the pitch must sit on the target
            f_ = f0_autocorr(di, ts + ms / 1000, win=0.12)
            row.append(f'{ms}ms:{f_:.2f}Hz({1200*np.log2(f_/fb):+.0f}c)')
        print(f'-- {name} ({fa:.1f}->{fb:.1f} Hz), post-glide pitch: ' + ' '.join(row))
    # zoomed spectrogram of the three slides (20-400 Hz, 40 ms window / 2 ms hop) to see the glide itself
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axs, ('slide29_24', 'slide24_31', 'slide32_24')):
        a = int((marks[name] - 0.02) * SR); b = int((marks[name] + 0.25) * SR)
        f, tt, S = signal.spectrogram(di[a:b], SR, nperseg=1920, noverlap=1920 - 96, nfft=8192, window='hann')
        Sdb = 10 * np.log10(S + 1e-14); sel = f <= 400
        ax.pcolormesh(tt * 1000 - 20, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 60, vmax=Sdb.max())
        ax.set_title(name + ' (DI, ms from onset)'); ax.set_ylim(20, 400); ax.set_xlabel('ms')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'bass_stress_slides.png'), dpi=100); plt.close(fig)

    # ---- harmonic decay vs frequency on the open C1 (DI and final): dB/s slope per band
    bands = [(28, 80), (80, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    t0_ = marks['openC1']
    rd = decay_rates(di, t0_ + 0.05, t0_ + 3.9, bands); rf = decay_rates(y, t0_ + 0.05, t0_ + 3.9, bands)
    print('-- open C1 decay dB/s per band (DI):    ' + '  '.join(f'{lo}-{hi}:{v:5.1f}' for (lo, hi), v in rd.items()))
    print('-- open C1 decay dB/s per band (FINAL): ' + '  '.join(f'{lo}-{hi}:{v:5.1f}' for (lo, hi), v in rf.items()))
    # harmonic levels of the open C1 at 0.1 s and 2 s (DI)
    h_early = harmonic_table(di, 32.703, 40, t0_ + 0.05, t0_ + 0.45); h_late = harmonic_table(di, 32.703, 40, t0_ + 2.0, t0_ + 2.4)
    print('   DI C1 harmonics rel h1 @0.05-0.45s: h1-12 ' + ' '.join(f'{v:+.0f}' for v in (h_early - h_early[0])[:12]) +
          ' | h16,20,24,30,40: ' + ' '.join(f'{(h_early-h_early[0])[k-1]:+.0f}' for k in (16, 20, 24, 30, 40)))
    print('   DI C1 harmonics rel h1 @2.0-2.4s:   h1-12 ' + ' '.join(f'{v:+.0f}' for v in (h_late - h_late[0])[:12]) +
          ' | h16,20,24,30,40: ' + ' '.join(f'{(h_late-h_late[0])[k-1]:+.0f}' for k in (16, 20, 24, 30, 40)))

    # ---- velocity sweep: per-note peak and rms in DI and final
    pk_di, pk_f, rms_f = [], [], []
    for i in range(8):
        on = marks['velsweep'] + i * 2 * S16; a, b = int(on * SR), int((on + 0.2) * SR)
        pk_di.append(np.abs(di[a:b]).max()); pk_f.append(np.abs(y[a:b]).max()); rms_f.append(np.sqrt(np.mean(y[a:b] ** 2)))
    pk_di, pk_f, rms_f = map(np.array, (pk_di, pk_f, rms_f))
    print(f'-- vel 0.3..1.0 open C1: DI peak spread {20*np.log10(pk_di.max()/pk_di.min()):.1f} dB; final peak spread '
          f'{20*np.log10(pk_f.max()/pk_f.min()):.1f} dB; final rms (dBFS) ' + ' '.join(f'{20*np.log10(v):.1f}' for v in rms_f))
    # per-note peaks over all notes (compressor evenness)
    pk = np.array([np.abs(y[int(t * SR):int((t + min(d, 0.25)) * SR)]).max() for (t, m, d, v, s_, mu) in ev])
    print(f'-- per-note peak all notes: min {pk.min():.3f} max {pk.max():.3f} spread {20*np.log10(pk.max()/pk.min()):.1f} dB')
    # overlap / dyad region sanity
    for name in ('overlap', 'dyad'):
        a, b = int(marks[name] * SR), int((marks[name] + 1.6) * SR)
        print(f'-- {name}: peak {np.abs(y[a:b]).max():.3f} rms {20*np.log10(np.sqrt(np.mean(y[a:b]**2))):.1f} dBFS')
    # click check: biggest sample-to-sample jump anywhere (final), relative to peak
    dj = np.abs(np.diff(y)); print(f'-- max sample jump {dj.max()/peak:.3f} of peak at t={dj.argmax()/SR:.3f}s')

    out = os.path.join(HERE, 'bass_stress.wav'); sf.write(out, y.astype(np.float32), SR, subtype='FLOAT')
    sf.write(os.path.join(HERE, 'bass_stress_di.wav'), din.astype(np.float32), SR, subtype='FLOAT')
    png = os.path.join(HERE, 'bass_stress_spec.png'); spectrogram_png(y, png, f'{MOD} stress (final)')
    png2 = os.path.join(HERE, 'bass_stress_di_spec.png'); spectrogram_png(din, png2, f'{MOD} stress (DI)')
    print('wrote', out, png, png2)
    if '--full' in sys.argv:
        evf = full_song_events()
        B._cache.clear() if hasattr(B, '_cache') else None
        tracemalloc.start(); t0 = time.time(); yf = B.render_bass(evf, total_len=113.0); dt = time.time() - t0
        cur, pk_mem = tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f'== FULL: {len(evf)} notes, {len(yf)/SR:.1f}s audio in {dt:.1f}s wall, peak traced mem {pk_mem/1e6:.0f} MB, '
              f'cache entries {len(getattr(B, "_cache", {}))}')
        stats('FULL', yf)
        pkf = np.array([np.abs(yf[int(t * SR):int((t + min(d, 0.25)) * SR)]).max() for (t, m, d, v, s_, mu) in evf])
        print(f'   per-note peak spread {20*np.log10(pkf.max()/pkf.min()):.1f} dB')
        assert dt < 120, 'too slow'
    print('STRESS DONE')
