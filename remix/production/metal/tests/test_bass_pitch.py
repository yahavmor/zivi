"""Pitch, slide, decay and contract edge-case checks for metal_bass."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
import metal_bass as B, engine as E
SR = E.SR


def f0_of(x, f_expect, t0, t1):
    seg = x[int(t0 * SR):int(t1 * SR)]
    seg = seg * np.hanning(len(seg))
    nfft = 1 << 19
    F = np.abs(np.fft.rfft(seg, nfft)); f = np.fft.rfftfreq(nfft, 1 / SR)
    sel = (f > f_expect * 0.94) & (f < f_expect * 1.06)
    i = np.argmax(np.where(sel, F, 0))
    # parabolic interpolation
    a, b, c = np.log(F[i - 1] + 1e-12), np.log(F[i] + 1e-12), np.log(F[i + 1] + 1e-12)
    d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return (i + d) * SR / nfft


def t60(x):
    env = np.abs(x); hop = 240
    e = np.array([env[i:i + hop].max() for i in range(0, len(env) - hop, hop)])
    edb = 20 * np.log10(e + 1e-9); pk = edb.max(); ipk = edb.argmax()
    below = np.where(edb[ipk:] < pk - 40)[0]
    t40 = below[0] * hop / SR if len(below) else np.nan
    return t40 * 1.5  # -40 dB time scaled to -60 dB


ok = True
print('tuning (DI, cents error):')
for midi in (24, 27, 29, 31, 32, 34, 36, 39, 41, 43, 44, 46, 48, 55, 60):
    fe = E.midi2hz(midi)
    for mute in (False, True):
        x = B.render_di([(0.0, midi, 0.8, 0.9, None, mute)], total_len=1.0)
        f = f0_of(x, fe, 0.03, 0.22 if mute else 0.6)
        cents = 1200 * np.log2(f / fe)
        ok &= abs(cents) < 8
        print(f'  midi {midi} {"mute" if mute else "open"}: expect {fe:.2f} Hz got {f:.2f} Hz ({cents:+.1f} c)')
print('slide C2->F2 instantaneous pitch (DI, 20 ms windows):')
x = B.render_di([(0.0, 41, 0.6, 0.9, 36, False)], total_len=1.0)
for tc in (0.01, 0.03, 0.05, 0.07, 0.10, 0.2):
    seg = x[int((tc - 0.01) * SR):int((tc + 0.01) * SR)]
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    lo, hi = int(SR / 120), int(SR / 50)
    lag = lo + np.argmax(ac[lo:hi]); print(f'  t={tc*1000:4.0f} ms  f~{SR/lag:6.1f} Hz')
print('decay:')
for midi in (24, 31, 36, 43):
    for mute in (False, True):
        y = B.render_di([(0.0, midi, 3.0, 0.9, None, mute)], total_len=3.5)
        print(f'  midi {midi} {"mute" if mute else "open"} DI T60 ~ {t60(y):.2f} s')
yb = B.render_bass([(0.0, 36, 0.12, 0.9, None, True)], total_len=1.0)
print(f'  muted C2 chug (dur 0.12) FINAL: -40 dB point {t60(yb)/1.5*1000:.0f} ms after peak')
print('contract edge cases:')
assert len(B.render_bass([], total_len=2.0)) == int(2.0 * SR)
assert len(B.render_bass([(0.0, 36, 0.5, 0.9, None, False)], total_len=3.0)) == int(3.0 * SR)
y = B.render_bass([(10.0, 36, 0.5, 0.9, None, False)], total_len=3.0); assert len(y) == int(3.0 * SR) and np.abs(y).max() == 0
y = B.render_bass([(0.0, 36, 0.02, 0.9, 41, 1), (0.05, 36, 0.02, 0.2, None, 0), (0.1, 72, 0.3, 1.0, 60, False)])
assert not np.isnan(y).any() and np.abs(y).max() <= 0.52
y = B.render_bass([(0.0, 36, 2.0, 0.9, None, False), (0.5, 39, 0.3, 0.9, None, True)]);
assert len(y) >= int(2.0 * SR)
# click check at the choke point (0.5 s): sample-to-sample jump relative to peak
d = np.abs(np.diff(y)); i0 = int(0.49 * SR); i1 = int(0.5 * SR)
print(f'  max sample jump before next onset (choke fade) {d[i0:i1].max()/np.abs(y).max():.3f} of peak (attack jump {d[i1:i1+500].max()/np.abs(y).max():.3f})')
print('  overlap/choke, short notes, high notes, empty, out-of-range: OK')
print('ALL OK' if ok else 'TUNING OUT OF TOLERANCE')
