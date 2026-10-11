"""Diagnostics: DI harmonic levels of a C2 note, low/mid band balance, pre-limiter peak (how hard the
limiter works), per-note peak before the bus for compressor sanity."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
from scipy import signal
import metal_bass as B, engine as E
sys.path.insert(0, HERE)
from test_bass_smoke import riff, band_energies, report, BANDS
SR = E.SR

def harmonics(x, f0, k=10, t0=0.05, t1=0.4):
    seg = x[int(t0*SR):int(t1*SR)] * np.hanning(int(t1*SR)-int(t0*SR))
    F = np.abs(np.fft.rfft(seg, 1<<18)); f = np.fft.rfftfreq(1<<18, 1/SR)
    out = []
    for h in range(1, k+1):
        sel = (f > f0*h*0.97) & (f < f0*h*1.03); out.append(20*np.log10(F[sel].max()+1e-12))
    out = np.array(out); return out - out[0]

for mute in (False, True):
    di = B.render_di([(0.0, 36, 1.0, 0.9, None, mute)], total_len=1.4)
    print('C2', 'mute' if mute else 'open', 'DI harmonics rel fundamental (dB):', np.round(harmonics(di, 65.41, 10, 0.03, 0.25 if mute else 0.5), 1))
ev = riff()
di = B.render_di(ev)
low = B.low_band(di); mid = B.mid_band(di); s = low + mid
print(f'low rms {20*np.log10(np.sqrt(np.mean(low**2))):.1f} dBFS peak {20*np.log10(np.abs(low).max()):.1f} | mid rms {20*np.log10(np.sqrt(np.mean(mid**2))):.1f} dBFS peak {20*np.log10(np.abs(mid).max()):.1f}')
print(f'sum  rms {20*np.log10(np.sqrt(np.mean(s**2))):.1f} dBFS peak {20*np.log10(np.abs(s).max()):.1f} dBFS (pre-bus)')
pre = E.compress(s, **B.BUS)
pre = E.eq(pre, ('lp', B.FINAL_LP_HZ, 0.707), ('lp', B.FINAL_LP_HZ, 0.707), ('hp', 28, 0.7))
print(f'pre-limiter peak {20*np.log10(np.abs(pre).max()):.1f} dBFS (limiter ceiling {B.PEAK_CEIL_DB}) comp GR est {20*np.log10(np.abs(s).max()/np.abs(pre).max()):.1f} dB on loudest peak')
pk=[]
for (t,m,d,v,sf_,mu) in ev:
    i0=int(t*SR); i1=int((t+min(d,0.25))*SR); pk.append(np.abs(pre[i0:i1]).max())
pk=np.array(pk); print('pre-limiter per-note peaks dBFS:', np.round(20*np.log10(pk),1))
report('low band', low); report('mid band', mid); report('final', B.bus(s), ev)
