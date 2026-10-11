"""Gain-staging harness: prints stage levels for reference notes under candidate constants."""
import sys, os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE+'/..'); sys.path.insert(0, HERE+'/../..')
import numpy as np, engine as E, metal_bass as B
from scipy import signal
from scipy.ndimage import maximum_filter1d
SR=E.SR
def rms_db(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-12)
def pk_db(x): return 20*np.log10(np.abs(x).max()+1e-12)
def share(x, lo, hi, a=0.03, b=1.0):
    f,p=signal.welch(x[int(a*SR):int(b*SR)], SR, nperseg=16384); return 10*np.log10(p[(f>=lo)&(f<hi)].sum()/p.sum())
def low_band_new(di, pre, gain, sat, comp, win=0.03):
    low = E.lp(di, B.CROSS_HZ, order=2) * pre
    low = E.saturate(low, sat['drive'], sat['mix'])
    sc = maximum_filter1d(np.abs(low), size=int(win*SR), mode='nearest')
    low = E.compress(low, sidechain=sc, **comp)
    return low * gain
def run(cfg):
    pre, gain, sat, comp, midg, bus = cfg['LOW_PRE'], cfg['LOW_GAIN'], cfg['LOW_SAT'], cfg['LOW_COMP'], cfg['MID_GAIN'], cfg['BUS']
    rows=[]
    for name, midi, vel, mute in (('C1 open v1.0',24,1.0,False),('C1 open v0.3',24,0.3,False),('G1 open v0.5',31,0.5,False),
                                  ('C1 mute v1.0',24,1.0,True),('C1 mute v0.5',24,0.5,True),('C2 open v0.9',36,0.9,False)):
        dur = 0.102 if mute else 1.2
        di=B.render_di([(0.0,midi,dur,vel,None,mute)], total_len=1.5)
        low=low_band_new(di, pre, gain, sat, comp); mid=B.mid_band(di)*(midg/B.MID_GAIN); s=low+mid
        pre_l=E.compress(s, **bus); pre_l=E.eq(pre_l, ('lp',B.FINAL_LP_HZ,0.707),('lp',B.FINAL_LP_HZ,0.707),('hp',28,0.7))
        fin=E.limiter(pre_l, ceiling_db=B.PEAK_CEIL_DB, lookahead=0.003, release=0.06)
        b_=0.25 if mute else 1.0
        rows.append((name, pk_db(low), rms_db(low[:int(b_*SR)]), pk_db(mid), rms_db(mid[:int(b_*SR)]), pk_db(pre_l), pk_db(fin), rms_db(fin[:int(b_*SR)]),
                     share(fin,20,150,0.03,b_), share(fin,150,700,0.03,b_), share(fin,700,2000,0.03,b_), share(fin,2000,4000,0.03,b_)))
    print('name             | low pk  rms | mid pk  rms | preLim pk | fin pk  rms | shares <150 150-700 700-2k 2-4k')
    for r in rows:
        print(f'{r[0]:16s} | {r[1]:5.1f} {r[2]:5.1f} | {r[3]:5.1f} {r[4]:5.1f} | {r[5]:7.1f}   | {r[6]:5.1f} {r[7]:5.1f} | {r[8]:5.1f} {r[9]:5.1f} {r[10]:5.1f} {r[11]:5.1f}')
if __name__=='__main__':
    import json
    cfgs = {
     'A': dict(LOW_PRE=4.0, LOW_GAIN=1.4, LOW_SAT=dict(drive=1.0,mix=0.15), LOW_COMP=dict(thresh_db=-12,ratio=3.5,attack=0.001,release=0.15,knee=6.0,makeup_db=0.0), MID_GAIN=1.0, BUS=dict(thresh_db=-16,ratio=3.0,attack=0.003,release=0.09,knee=6.0,makeup_db=0.0)),
     'B': dict(LOW_PRE=4.0, LOW_GAIN=1.2, LOW_SAT=dict(drive=1.0,mix=0.15), LOW_COMP=dict(thresh_db=-12,ratio=3.5,attack=0.001,release=0.15,knee=6.0,makeup_db=0.0), MID_GAIN=1.1, BUS=dict(thresh_db=-16,ratio=2.5,attack=0.003,release=0.09,knee=6.0,makeup_db=0.0)),
    }
    for k in (sys.argv[1:] or cfgs.keys()):
        print('== config', k, json.dumps(cfgs[k])); run(cfgs[k])
