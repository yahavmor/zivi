"""Transient-vs-body analysis for kick/snare/toms (helper used while tuning)."""
import os, sys, numpy as np
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import engine as E, metal_drums as D
EDGES = [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]
def bands(x, nperseg=4096):
    f, P = signal.welch(E.mono(x), E.SR, nperseg=nperseg)
    e = [P[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])]; ref = e[3] + 1e-20
    return ' '.join(f'{10*np.log10(v/ref+1e-20):6.1f}' for v in e)
def report(name, y):
    y = E.mono(y); hi = E.hp(E.hp(y, 1500), 1500); lo = E.lp(E.lp(y, 400), 400)
    first = y[:int(0.008 * E.SR)]
    print(f'{name:12s} peak_hi1.5k/peak_lo400 = {np.abs(hi).max()/np.abs(lo).max():.2f}  | whole: {bands(y)}')
    print(f'{"":12s} first 8 ms (384-pt):                {bands(first, 384)}')
    k = int(0.002 * E.SR); env = 10*np.log10(np.convolve(y ** 2, np.ones(k) / k, 'same') + 1e-12); pk = env.max(); ipk = np.argmax(env)
    def tdb(d):
        idx = np.where(env[ipk:] < pk + d)[0]; return idx[0]/E.SR*1000 if len(idx) else -1
    print(f'{"":12s} env: -10dB@{tdb(-10):.0f}ms -20dB@{tdb(-20):.0f}ms -40dB@{tdb(-40):.0f}ms len={len(y)/E.SR*1000:.0f}ms')
if __name__ == '__main__':
    for nm in (sys.argv[1:] or ['kick', 'snare', 'tom_floor', 'tom_hi']):
        report(nm, D.hit(nm, 1.0, 0))
