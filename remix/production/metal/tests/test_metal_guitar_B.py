"""Smoke / measurement test for metal_guitar_B (oscillator-based metal guitar).
Renders short phrases for rhythm (mono + double-tracked pair), lead, clean and the extras; checks NaN/DC/
clipping; prints peak/RMS/crest, band energies (sum of Welch PSD per band, relative to 250-500 Hz),
chug decay time, open-chord sustain, lead pitch trajectory (bend / vibrato / slide), L/R take correlation;
saves spectrogram PNGs next to this file. Run: python3 tests/test_metal_guitar_B.py [--full]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
import soundfile as sf
from scipy import signal
import engine as E
import metal_guitar_B as G

SR = E.SR
BEAT = 60 / 110
S16 = BEAT / 4
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def band_energies(x):
    f, p = signal.welch(E.mono(x), SR, nperseg=8192)
    ref = p[(f >= 250) & (f < 500)].sum()
    return {(lo, hi): 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / ref + 1e-20) for lo, hi in BANDS}


def band_str(be):
    return '  '.join(f'{lo}-{hi}:{v:+5.1f}' for (lo, hi), v in be.items())


def rms_env(x, win=0.005):
    k = int(win * SR)
    e = np.sqrt(np.convolve(E.mono(x) ** 2, np.ones(k) / k, mode='same'))
    return e


def decay_time(x, db_down=20.0):
    """seconds from the envelope peak until it has fallen db_down (first crossing)."""
    e = rms_env(x)
    i0 = int(np.argmax(e)); thr = e[i0] * 10 ** (-db_down / 20)
    idx = np.where(e[i0:] < thr)[0]
    return (idx[0] / SR) if len(idx) else float('nan'), i0 / SR


def level_at(x, t):
    e = rms_env(x)
    pk = e.max()
    return 20 * np.log10(e[int(t * SR)] / (pk + 1e-12) + 1e-12)


def track_pitch(x, fmin=80, fmax=2500, hop=0.01, win=0.06, nh=5):
    """harmonic-product-spectrum f0 tracker (robust on distorted tones). returns (times, f0 Hz)."""
    x = E.mono(x); n = len(x); k = int(win * SR); h = int(hop * SR); nfft = 1 << 17
    fr = np.fft.rfftfreq(nfft, 1 / SR); df = fr[1]
    i_lo, i_hi = int(fmin / df), int(fmax / df)
    ts, fs = [], []
    w = np.hanning(k)
    for i in range(0, n - k, h):
        seg = x[i:i + k]
        if np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        P = np.log(np.abs(np.fft.rfft(seg * w, nfft)) + 1e-9)
        hps = np.zeros(i_hi - i_lo)
        for m in range(1, nh + 1):
            hps += P[i_lo * m:i_hi * m:m][:i_hi - i_lo]
        j = int(np.argmax(hps)) + i_lo
        ts.append(i / SR + win / 2); fs.append(fr[j])
    return np.array(ts), np.array(fs)


def alias_floor(y, f0, lo=1000, hi=5000):
    """line-to-floor ratio (dB): mean PSD at harmonic multiples of f0 vs mean PSD half-way between them."""
    f, p = signal.welch(E.mono(y), SR, nperseg=65536)
    pdb = 10 * np.log10(p + 1e-20)
    ks = np.arange(int(np.ceil(lo / f0)), int(hi / f0))
    at = np.array([pdb[int(round(k * f0 / f[1]))] for k in ks])
    mid = np.array([pdb[int(round((k + 0.5) * f0 / f[1]))] for k in ks])
    return at.mean() - mid.mean()


def hf_contrast(x, t0, lo=2000, hi=6000):
    """energy in [lo,hi] in the first 12 ms after t0 vs 60-110 ms after t0 (pick-attack brightness contrast, dB)."""
    sos = signal.butter(4, [lo, hi], 'band', fs=SR, output='sos'); b = signal.sosfilt(sos, E.mono(x))
    a = b[int(t0 * SR):int((t0 + 0.012) * SR)]; c = b[int((t0 + 0.06) * SR):int((t0 + 0.11) * SR)]
    return 10 * np.log10(np.mean(a ** 2) / (np.mean(c ** 2) + 1e-20))


def spectrogram_png(x, path, title, fmax=12000, ylim=(40, 12000)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = E.mono(x)
    f, tt, S = signal.spectrogram(x, SR, nperseg=2048, noverlap=1536, window='hann')
    Sdb = 10 * np.log10(S + 1e-14)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={'height_ratios': [1, 3]})
    ax[0].plot(np.arange(len(x)) / SR, x, lw=0.4, color='#444')
    ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(-0.7, 0.7); ax[0].set_ylabel('amp'); ax[0].set_title(title)
    sel = f <= fmax
    im = ax[1].pcolormesh(tt, f[sel], Sdb[sel], shading='auto', cmap='magma', vmin=Sdb.max() - 90, vmax=Sdb.max())
    ax[1].set_yscale('symlog', linthresh=250); ax[1].set_ylim(*ylim)
    ax[1].set_yticks([50, 65, 100, 150, 250, 500, 1000, 2000, 4000, 8000, 12000])
    ax[1].set_yticklabels(['50', '65', '100', '150', '250', '500', '1k', '2k', '4k', '8k', '12k'])
    ax[1].set_ylabel('Hz'); ax[1].set_xlabel('s')
    fig.colorbar(im, ax=ax[1], label='dB')
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def report(name, x):
    x = np.asarray(x)
    assert not np.isnan(x).any(), name + ': NaN'
    assert not np.isinf(x).any(), name + ': inf'
    m = E.mono(x)
    peak = np.abs(x).max(); rms = np.sqrt(np.mean(m ** 2)); dc = m.mean()
    sub20 = E.lp(m, 20, order=2); dc_lf = np.sqrt(np.mean(sub20 ** 2))
    print(f'[{name}] len={len(x)/SR:.2f}s peak={peak:.3f} ({20*np.log10(peak+1e-12):.1f} dBFS) rms={rms:.3f} '
          f'({20*np.log10(rms+1e-12):.1f} dBFS) crest={20*np.log10(peak/(rms+1e-12)):.1f} dB '
          f'DC={dc:+.5f} (<20Hz rms {20*np.log10(dc_lf+1e-12):.1f} dBFS) clipped={(np.abs(x)>0.99).sum()}')
    be = band_energies(m)
    print('  bands rel 250-500 (energy): ' + band_str(be))
    assert abs(dc) < 2e-3, name + ': DC offset'
    assert (np.abs(x) > 0.99).sum() == 0, name + ': clipping'
    return be


def wav(name, x):
    p = os.path.join(HERE, name); sf.write(p, np.asarray(x, dtype=np.float32), SR, subtype='FLOAT'); return p


# ------------------------------------------------------------------ event sets
def rhythm_riff():
    ev = []; t = 0.0
    for i in range(8):                                   # 2 beats of 16th chugs on C2
        ev.append((t + i * S16, 36, S16, 0.95 if i % 4 == 0 else 0.82, True))
    t += 8 * S16
    ev.append((t, 36, 2 * BEAT, 1.0, False))              # open C5 power chord, 2 beats
    t += 2 * BEAT
    for i in range(4): ev.append((t + i * S16, 39, S16, 0.85, True))     # Eb2 chugs
    t += 4 * S16
    for i in range(4): ev.append((t + i * S16, 41, S16, 0.85, True))     # F2 chugs
    t += 4 * S16
    ev.append((t, 43, BEAT, 0.9, False))                  # G5 chord, 1 beat
    t += BEAT
    ev.append((t, 46, 0.5 * BEAT, 0.9, False))            # Bb5 short chord
    t += 0.5 * BEAT
    ev.append((t, 48, 0.5 * BEAT, 0.9, False))            # C3 octave chord
    t += 0.5 * BEAT
    ev.append((t, 36, 3 * BEAT, 1.0, False))              # long ring-out
    return ev


def lead_phrase():
    # (t, midi, dur, vel, bend_semis, vibrato, slide_from)
    return [
        (0.0, 67, 0.6, 0.9, 0.0, 0.0, None),            # G4 plain
        (0.6, 70, 0.6, 0.9, 0.0, 0.0, 67),              # Bb4 slid from G4
        (1.2, 72, 1.6, 1.0, 2.0, 0.35, None),           # C5 whole-step bend + vibrato
        (2.9, 75, 0.3, 0.9, 0.0, 0.0, None),            # fast run (legato cuts)
        (3.1, 74, 0.3, 0.9, 0.0, 0.0, None),
        (3.3, 72, 0.3, 0.9, 0.0, 0.0, None),
        (3.5, 70, 2.5, 1.0, 0.0, 0.5, None),            # Bb4 long sustain with vibrato (feedback creeps in)
    ]


def clean_arp():
    notes = [48, 55, 60, 63, 67, 63, 60, 55] * 2 + [51, 58, 63, 67, 70, 67, 63, 58]
    return [(i * S16 * 2, n, S16 * 2, 0.75 + 0.1 * (i % 2)) for i, n in enumerate(notes)]


def full_song_rhythm():
    """~113 s of realistic chug density: mostly 16th chugs, some chords; ~700 events."""
    ev = []; roots = [36, 36, 39, 41, 43, 44, 46, 48]; t = 0.0; bar = 0
    while t < 112.0:
        root = roots[bar % len(roots)]
        if bar < 4 or 56 < bar < 60:
            ev.append((t, root, 4 * BEAT, 0.9, False))
        elif bar % 4 == 3:
            ev.append((t, root, 2 * BEAT, 1.0, False)); ev.append((t + 2 * BEAT, root + 3, 2 * BEAT, 1.0, False))
        else:
            for i in range(16):
                ev.append((t + i * S16, root if i % 8 != 7 else root + 3, S16, 0.8 + 0.15 * (i % 4 == 0), i % 8 != 6))
        t += 4 * BEAT; bar += 1
    return ev


def full_song_lead():
    ev = []; t = 0.0; k = 0
    while t < 110.0:
        m = [67, 70, 72, 75, 74, 72, 70, 67][k % 8]
        ev.append((t, m, 0.4 if k % 3 else 1.5, 0.9, 1.0 if k % 5 == 0 else 0.0, 0.3 if k % 2 else 0.0, m - 2 if k % 4 == 0 else None))
        t += 0.5 if k % 3 else 1.6; k += 1
    return ev


if __name__ == '__main__':
    # ---------------------------------------------------------------- single chug / chord
    print('=== single events ===')
    chug = G.render_rhythm([(0.05, 36, S16, 1.0, True)], seed=1, total_len=0.6)
    be = report('C2 palm-mute chug', chug)
    dt20, tpk = decay_time(chug, 20); dt30, _ = decay_time(chug, 30)
    e = rms_env(chug); i_on = int(0.05 * SR); atk = (np.argmax(e[i_on:] > 0.7 * e.max())) / SR
    i_pk = int(np.argmax(e)); dt20 = (np.argmax(e[i_pk:] < e.max() * 10 ** (-20 / 20)) / SR) + (i_pk - i_on) / SR
    dt30 = (np.argmax(e[i_pk:] < e.max() * 10 ** (-30 / 20)) / SR) + (i_pk - i_on) / SR
    print(f'  onset-to-70% {atk*1000:.1f} ms, env peak at +{(i_pk-i_on)/SR*1000:.0f} ms, -20 dB at +{dt20*1000:.0f} ms, -30 dB at +{dt30*1000:.0f} ms after onset; level at +100/+150/+200ms {level_at(chug, 0.15):.1f}/{level_at(chug, 0.20):.1f}/{level_at(chug, 0.25):.1f} dB')
    print(f'  pick-attack HF contrast (2-6k, first 12 ms vs 60-110 ms): {hf_contrast(chug, 0.05):+.1f} dB')
    thump = 10 * np.log10(sum(10 ** (be[b] / 10) for b in [(60, 120), (120, 250)]))
    print(f'  thump (60-250 vs 250-500): {thump:+.1f} dB   above-5k: 4-8k {be[(4000,8000)]:+.1f}  8-16k {be[(8000,16000)]:+.1f}')
    chug_di = G.render_rhythm_di([(0.05, 36, S16, 1.0, True)], seed=1, total_len=0.6)
    print('  DI chug peak %.3f  bands: %s' % (np.abs(chug_di).max(), band_str(band_energies(chug_di))))
    open_ = G.render_rhythm([(0.05, 36, 3.0, 1.0, False)], seed=1, total_len=3.6)
    report('C5 open power chord', open_)
    print(f'  level rel peak: 0.5s {level_at(open_, 0.55):+.1f}  1.0s {level_at(open_, 1.05):+.1f}  1.5s {level_at(open_, 1.55):+.1f}  2.0s {level_at(open_, 2.05):+.1f}  2.9s {level_at(open_, 2.95):+.1f} dB (then released)')
    open_di = G.render_rhythm_di([(0.05, 36, 3.0, 1.0, False)], seed=1, total_len=3.6)
    print('  DI chord peak %.3f  bands: %s' % (np.abs(open_di).max(), band_str(band_energies(open_di))))
    # harmonic density: count spectral peaks > -40 dB rel max below 5 kHz in the chord
    f, p = signal.welch(open_[int(0.3 * SR):int(1.3 * SR)], SR, nperseg=16384)
    pdb = 10 * np.log10(p + 1e-20); sel = (f > 50) & (f < 5000)
    pk_idx = signal.find_peaks(pdb[sel], height=pdb[sel].max() - 40, distance=8)[0]
    print(f'  chord 0.3-1.3s: {len(pk_idx)} spectral peaks within 40 dB of max below 5 kHz; mean level 4-5k vs 1-2k {pdb[(f>4000)&(f<5000)].mean()-pdb[(f>1000)&(f<2000)].mean():+.1f} dB; 6-8k vs 1-2k {pdb[(f>6000)&(f<8000)].mean()-pdb[(f>1000)&(f<2000)].mean():+.1f} dB')

    # single string C2, no detune/sub: aliasing floor through the amp, and DI harmonic structure vs an ideal saw
    rr = np.random.default_rng(3); nn = int(1.5 * SR)
    s1 = G._string(np.full(nn, E.midi2hz(36)), 1.0, rr, mute=False, pick_pos=0.13, detune_cents=0.0, sub=0.0, saw2=0.0)
    s1 *= np.exp(-G._t(nn) / 2.2); di1 = G._pickup(s1, 'bridge') * 0.3
    amp1 = G.amp_sim(di1, 'rhythm') * G._RHY_GAIN
    print(f'  single-string C2 through amp: harmonic line-to-floor ratio 1-5 kHz {alias_floor(amp1, E.midi2hz(36)):.1f} dB (DI: {alias_floor(di1, E.midi2hz(36), 200, 3000):.1f} dB)')
    f, p = signal.welch(di1[int(0.1 * SR):], SR, nperseg=65536); pdb = 10 * np.log10(p + 1e-20)
    hk = np.array([pdb[int(round(k * E.midi2hz(36) / f[1]))] for k in range(1, 33)]); hk -= hk[0]
    saw = 20 * np.log10(1 / np.arange(1, 33))
    print('  DI harmonics 1..32 rel h1 (dB): ' + ' '.join(f'{v:.0f}' for v in hk))
    print(f'  deviation from an ideal 1/k saw: rms {np.sqrt(np.mean((hk - saw) ** 2)):.1f} dB, max |dev| {np.abs(hk - saw).max():.1f} dB (plain saw would be ~0)')

    # ---------------------------------------------------------------- rhythm riff (mono + pair)
    print('=== rhythm riff ===')
    ev = rhythm_riff()
    t0 = time.time(); y = G.render_rhythm(ev, seed=1); print(f'render_rhythm: {len(ev)} events, {len(y)/SR:.1f}s in {time.time()-t0:.2f}s')
    report('rhythm mono', y)
    assert 0.3 <= np.abs(y).max() <= 0.6
    t0 = time.time(); st = G.render_rhythm_pair(ev); print(f'render_rhythm_pair: {len(st)/SR:.1f}s in {time.time()-t0:.2f}s')
    report('rhythm pair', st)
    c = np.corrcoef(st[:, 0], st[:, 1])[0, 1]
    print(f'  L/R correlation {c:.2f} (want well below 1: independent takes)   L peak {np.abs(st[:,0]).max():.3f}  R peak {np.abs(st[:,1]).max():.3f}')
    # chug tails between 16ths: level in the last 25 ms before each of the first 8 chugs' successors
    tails = []
    for i in range(1, 8):
        a = int((i * S16 - 0.025) * SR); b = int(i * S16 * SR)
        tails.append(20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-9))
    bodies = [20 * np.log10(np.sqrt(np.mean(y[int((i * S16 + 0.01) * SR):int((i * S16 + 0.06) * SR)] ** 2)) + 1e-9) for i in range(1, 8)]
    print(f'  16th chugs: body (10-60 ms) RMS dBFS {np.round(bodies,1).tolist()}; gap (last 25 ms before next chug) {np.round(tails,1).tolist()}; mean dip {np.mean(np.array(bodies)-np.array(tails)):.1f} dB')
    wav('gtrB_rhythm_pair.wav', st); wav('gtrB_rhythm_mono.wav', y)
    di = G.render_rhythm_di(ev, seed=1); wav('gtrB_rhythm_di.wav', di / (np.abs(di).max() + 1e-9) * 0.5)
    spectrogram_png(y, os.path.join(HERE, 'gtrB_rhythm_spec.png'), 'metal_guitar_B rhythm (mono take, amp+cab)')
    spectrogram_png(di / (np.abs(di).max() + 1e-9) * 0.5, os.path.join(HERE, 'gtrB_rhythm_di_spec.png'), 'metal_guitar_B rhythm DI (pre-amp)')
    spectrogram_png(chug, os.path.join(HERE, 'gtrB_chug_spec.png'), 'single C2 palm-mute chug (amp+cab)', fmax=12000)

    # ---------------------------------------------------------------- lead
    print('=== lead ===')
    lev = lead_phrase()
    t0 = time.time(); ld = G.render_lead(lev); print(f'render_lead: {len(lev)} notes, {len(ld)/SR:.1f}s in {time.time()-t0:.2f}s')
    report('lead', ld)
    assert 0.3 <= np.abs(ld).max() <= 0.6
    ts, f0 = track_pitch(ld, 300, 800)
    def f_at(a, b):
        s = (ts >= a) & (ts < b); return f0[s]
    print(f'  G4 (392 Hz) at 0.2-0.5s: median {np.median(f_at(0.2,0.5)):.1f} Hz')
    sl = f_at(0.6, 0.75); print(f'  slide G4->Bb4: f0 over 0.60-0.75s: {np.round(sl,0).tolist()}')
    bd = f_at(1.2, 2.8); print(f'  bend C5 (523 -> 587 Hz): start {np.median(f_at(1.22,1.3)):.0f} Hz, end {np.median(f_at(1.9,2.1)):.0f} Hz; vibrato window 2.0-2.8s min/max {f_at(2.0,2.8).min():.0f}/{f_at(2.0,2.8).max():.0f} Hz')
    vb = f_at(4.5, 5.9); print(f'  Bb4 (466 Hz) sustain vibrato 4.5-5.9s: min {vb.min():.0f} max {vb.max():.0f} Hz = {1200*np.log2(vb.max()/vb.min()):.0f} cents p-p; level at 1.0s/2.0s into note: {level_at(ld[int(3.5*SR):int(6.1*SR)], 1.0):+.1f}/{level_at(ld[int(3.5*SR):int(6.1*SR)], 2.0):+.1f} dB')
    # vibrato rate from the pitch track
    v = vb - vb.mean(); ac = np.correlate(v, v, 'full')[len(v) - 1:]; lag = np.argmax(ac[5:40]) + 5
    print(f'  vibrato rate estimate {1/(lag*0.01):.1f} Hz')
    wav('gtrB_lead.wav', ld); spectrogram_png(ld, os.path.join(HERE, 'gtrB_lead_spec.png'), 'metal_guitar_B lead (slide, bend, vibrato, legato run, sustain)')

    # ---------------------------------------------------------------- clean
    print('=== clean ===')
    cev = clean_arp()
    t0 = time.time(); cl = G.render_clean(cev); print(f'render_clean: {len(cev)} notes, {len(cl)/SR:.1f}s in {time.time()-t0:.2f}s')
    report('clean', cl)
    print(f'  L/R correlation {np.corrcoef(cl[:,0], cl[:,1])[0,1]:.2f}')
    wav('gtrB_clean.wav', cl); spectrogram_png(cl, os.path.join(HERE, 'gtrB_clean_spec.png'), 'metal_guitar_B clean arpeggio (chorus+reverb)')

    # ---------------------------------------------------------------- extras
    print('=== extras ===')
    ex = {}
    for name, fn in [('feedback_swell', lambda: G.feedback_swell(55, 2.5)), ('pick_scrape', lambda: G.pick_scrape(1.0)),
                     ('pinch_harmonic', lambda: G.pinch_harmonic(48, 1.2)), ('dive_bomb', lambda: G.dive_bomb(48, 2.0))]:
        t0 = time.time(); x = fn(); ex[name] = x
        print(f'{name}: {len(x)/SR:.2f}s in {time.time()-t0:.2f}s'); report(name, x)
    ts, f0 = track_pitch(ex['dive_bomb'], 25, 300, win=0.12, nh=4)
    tt = np.array([0.1, 0.3, 0.8, 1.0, 1.3, 1.5]); exp_f = E.midi2hz(48) * 2 ** (-26.0 * (tt / 2.0) ** 1.6 / 12)
    got = np.array([np.median(f0[(ts > a - 0.05) & (ts < a + 0.05)]) for a in tt])
    print('  dive_bomb f0 (C3 start, 26 semis over 2 s): ' + '  '.join(f't={a:.1f}s {g:.0f}Hz(exp {e:.0f})' for a, g, e in zip(tt, got, exp_f)) + f'  -> {int((np.abs(got/exp_f-1)<0.08).sum())}/{len(tt)} within 8% (tracker octave errors possible on the HP-filtered sweep)')
    ts, f0 = track_pitch(ex['pinch_harmonic'], 100, 2500)
    print(f'  pinch_harmonic (C3=131 Hz) tracked f0 at 0.3-0.6s: {np.median(f0[(ts>0.3)&(ts<0.6)]):.0f} Hz')
    demo = np.concatenate([ex['feedback_swell'], ex['pick_scrape'], ex['pinch_harmonic'], ex['dive_bomb']])
    wav('gtrB_extras.wav', demo); spectrogram_png(demo, os.path.join(HERE, 'gtrB_extras_spec.png'), 'extras: feedback_swell | pick_scrape | pinch_harmonic | dive_bomb')

    # ---------------------------------------------------------------- full-length timing
    if '--full' in sys.argv:
        print('=== full-song timing ===')
        evf = full_song_rhythm(); t0 = time.time(); yf = G.render_rhythm_pair(evf, total_len=113.0); dt = time.time() - t0
        print(f'rhythm pair: {len(evf)} events -> {len(yf)/SR:.1f}s stereo in {dt:.1f}s'); report('FULL rhythm pair', yf)
        lf = full_song_lead(); t0 = time.time(); ylf = G.render_lead(lf, total_len=113.0); dt2 = time.time() - t0
        print(f'lead: {len(lf)} notes in {dt2:.1f}s'); report('FULL lead', ylf)
        cf = [(i * S16 * 2, [48, 55, 60, 63][i % 4], S16 * 2, 0.8) for i in range(int(113 / (S16 * 2)))]
        t0 = time.time(); ycf = G.render_clean(cf, total_len=113.0); dt3 = time.time() - t0
        print(f'clean: {len(cf)} notes in {dt3:.1f}s'); report('FULL clean', ycf)
        print(f'TOTAL module time for a full song: {dt+dt2+dt3:.1f}s')
    print('done')
