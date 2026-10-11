"""Diagnostics for metal_guitar_A: cab IR magnitude curves, pre-clip amp peaks per preset, DI levels."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np
from scipy import signal
import metal_guitar_A as G, engine as E
SR = G.SR
pts = [50, 80, 100, 120, 160, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 16000]
for kind in ['4x12', 'lead', 'clean']:
    ir = G.cab_ir(1, kind)
    f, p = signal.freqz(ir, 1, worN=4096, fs=SR)
    mdb = 20 * np.log10(np.abs(p) + 1e-12)
    print('%-5s IR %d ms peak %.2f' % (kind, len(ir) // 48, np.abs(ir).max()))
    print('     ', ' '.join('%d:%+.1f' % (q, mdb[np.argmin(np.abs(f - q))]) for q in pts))
ev = [(0.1, 36, 0.11, 1.0, True), (0.5, 36, 1.5, 1.0, False)]
di = G.render_rhythm_di(ev, seed=1, total_len=2.5)
P = dict(G.PRESETS['rhythm']); P['out'] = 0.01
y = G.amp_sim(di, P, seed=1) / 0.01
print('rhythm pre-clip peak (out=1): %.3f  -> out for 0.5 peak = %.3f   (rms %.3f)' % (np.abs(y).max(), 0.5 / np.abs(y).max(), np.sqrt(np.mean(y ** 2))))
print('DI peak %.3f  chord DI @+0.3s %.3f  @+0.8s %.3f' % (np.abs(di).max(), np.abs(di[int(0.8 * SR):int(0.9 * SR)]).max(), np.abs(di[int(1.3 * SR):int(1.4 * SR)]).max()))
lev = [(0.1, 67, 1.0, 0.9, 0.0, 0.3, None), (1.2, 74, 1.5, 1.0, 2.0, 0.4, None)]
ldi = G.render_lead_di(lev, total_len=3.0)
P = dict(G.PRESETS['lead']); P['out'] = 0.01
y = G.amp_sim(ldi, P, seed=3) / 0.01
print('lead pre-clip peak (out=1): %.3f -> out for 0.5 peak = %.3f' % (np.abs(y).max(), 0.5 / np.abs(y).max()))
print('lead DI peak %.3f' % np.abs(ldi).max())
