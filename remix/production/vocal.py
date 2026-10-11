"""Vocal preparation: cleaning, slicing into words, pitch variants, auto-tune via rubberband pitchmap."""
import numpy as np, subprocess, os, json
from scipy import signal
import engine as E
SR = E.SR
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "original_48k.wav")
if not os.path.exists(RAW):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(HERE, "original_recording.ogg"), "-ar", "48000", "-ac", "1", RAW], check=True)
VD = os.path.join(HERE, "vox")
MODE = os.environ.get("REMIX_MODE", "main")
os.makedirs(VD, exist_ok=True)

# word boundaries in the ORIGINAL file (seconds) — from energy/transcript analysis
WORDS = {
    'ata1':   (0.745, 1.245), 'mushe1': (1.245, 1.875), 'meham1': (1.875, 2.495),
    'ata2':   (2.495, 2.795), 'mushe2': (2.795, 3.235),
    'ata3':   (3.275, 3.655), 'mushe3': (3.655, 3.885), 'meham3': (3.885, 5.33),
    # sub-syllables
    'mu1': (1.245, 1.40), 'she1': (1.40, 1.875),
    'me3': (3.885, 4.00), 'ha3': (4.00, 4.17), 'mis3': (4.17, 4.57), 'chak3': (4.57, 5.33), 'k_click': (5.17, 5.33),
    'mu3': (3.655, 3.76),
}
PHRASES = {
    'p1': (0.745, 2.495),           # אתה מושעה מהמשחק
    'p2': (2.495, 3.235),           # אתה מושעה
    'p3': (3.275, 5.33),            # אתה מושעה מהמשחק!
    'p3a': (3.275, 3.885),          # אתה מושעה (pickup)
    'full': (0.745, 5.33),
    'p12': (0.745, 3.235),
}

def clean_master():
    """denoise, EQ, excite, compress → vox/clean.wav (mono)"""
    tmp1 = f"{VD}/_dn.wav"
    # gentle denoise + highpass + presence + air via exciter + de-ess
    af = ("highpass=f=75:p=2,afftdn=nf=-42:nr=10:tn=1,"
          "equalizer=f=320:t=q:w=1.2:g=-2.5,equalizer=f=3000:t=q:w=1.0:g=2.5,equalizer=f=1200:t=q:w=1.5:g=1,"
          "aexciter=level_in=1:level_out=1:amount=2.5:drive=7:blend=0:freq=3800:ceil=12000,"
          "deesser=i=0.3:m=0.5:f=0.5,"
          "acompressor=threshold=-22dB:ratio=3:attack=8:release=120:makeup=3,"
          "alimiter=limit=0.9:attack=2:release=50")
    E.ffmpeg_filter(RAW, tmp1, af)
    x = E.mono(E.read(tmp1))
    x = x / np.abs(x).max() * 0.85
    E.write(f"{VD}/clean.wav", x)
    return x

def slice_(x, a, b, fade_in=0.004, fade_out=0.008, src_offset=0.0):
    i, j = int((a - src_offset) * SR), int((b - src_offset) * SR)
    y = x[i:j].copy()
    return E.fade(y, fade_in, fade_out)

def rubberband(in_path, out_path, args, cache=True):
    if cache and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        try: return E.mono(E.read(out_path))
        except Exception: pass
    subprocess.run(['rubberband', '-3', '-q'] + args + [in_path, out_path], check=True, capture_output=True)
    return E.mono(E.read(out_path))

def pitch_shift(x, semis, formant=True, tag=''):
    ip = f"{VD}/_ps_in_{MODE}.wav"; op = f"{VD}/ps_{tag}_{semis:+.1f}{'F' if formant else 'N'}.wav"
    E.write(ip, x)
    args = ['-p', str(semis)] + (['-F'] if formant else [])
    return rubberband(ip, op, args)

def time_stretch(x, ratio, tag=''):
    ip = f"{VD}/_ts_in.wav"; op = f"{VD}/ts_{tag}_{ratio:.3f}.wav"
    E.write(ip, x)
    return rubberband(ip, op, ['-t', str(ratio), '-F' if False else '-q'])

# ---------- YIN pitch tracking ----------
def yin_f0(x, fmin=110, fmax=480, hop=240, win=1800, thr=0.12):
    x = E.mono(x); n = len(x); tmin, tmax = int(SR / fmax), int(SR / fmin)
    times, f0s, conf = [], [], []
    for s in range(0, n - win - tmax, hop):
        fr = x[s:s + win + tmax]
        a = fr[:win]
        # d(tau) = sum (a[j]-fr[j+tau])^2
        e_a = (a ** 2).sum()
        cs = np.cumsum(np.concatenate([[0], fr ** 2]))
        e_b = cs[tmax + win] - cs[tmax] if False else None
        taus = np.arange(0, tmax + 1)
        e_b = cs[taus + win] - cs[taus]
        xc = signal.correlate(fr, a, mode='valid')  # length tmax+1 : sum a[j]*fr[j+tau]
        d = e_a + e_b - 2 * xc
        # cumulative mean normalized difference
        cmnd = np.ones_like(d); cum = np.cumsum(d[1:]); cmnd[1:] = d[1:] * np.arange(1, len(d)) / (cum + 1e-12)
        best = None
        for tau in range(tmin, tmax):
            if cmnd[tau] < thr:
                while tau + 1 < tmax and cmnd[tau + 1] < cmnd[tau]: tau += 1
                best = tau; break
        if best is None:
            tau = tmin + int(np.argmin(cmnd[tmin:tmax])); best = tau
        # parabolic interp
        if 1 <= best < len(cmnd) - 1:
            y0, y1, y2 = cmnd[best - 1], cmnd[best], cmnd[best + 1]
            den = (y0 - 2 * y1 + y2); shift = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0
            tau_f = best + np.clip(shift, -1, 1)
        else: tau_f = best
        rms = np.sqrt((a ** 2).mean())
        times.append((s + win / 2) / SR); f0s.append(SR / tau_f); conf.append((cmnd[best], rms))
    times, f0s = np.array(times), np.array(f0s)
    voiced = np.array([(c < 0.35) and (r > 0.01) for c, r in conf])
    return times, f0s, voiced

SCALE_GMINOR = [55, 57, 58, 60, 62, 63, 65]  # G A Bb C D Eb F (as pitch classes via mod 12)
def nearest_scale_midi(f, scale=SCALE_GMINOR):
    m = 69 + 12 * np.log2(f / 440)
    pcs = sorted(set(s % 12 for s in scale))
    cands = []
    for o in range(-1, 2):
        for pc in pcs:
            cands.append(12 * (int(m) // 12 + o) + pc)
    return min(cands, key=lambda c: abs(c - m))

def autotune(x, tag, scale=SCALE_GMINOR, overrides=None, strength=1.0, smooth=0.0, max_shift=7.0, formant=True, base_shift=0.0):
    """hard-tune to scale notes using rubberband pitchmap. overrides: list of (t0,t1,midi) in seconds of x."""
    times, f0, voiced = yin_f0(x)
    offs = np.zeros(len(times))
    last = 0.0
    for i, (t, f, v) in enumerate(zip(times, f0, voiced)):
        if v:
            target = None
            if overrides:
                for (t0, t1, m) in overrides:
                    if t0 <= t < t1: target = m
            if target is None: target = nearest_scale_midi(f, scale)
            m = 69 + 12 * np.log2(f / 440)
            o = np.clip((target - m) * strength, -max_shift, max_shift); last = o
        else:
            o = last
        offs[i] = o
    if smooth > 0:
        k = max(1, int(smooth * SR / 240)); offs = np.convolve(offs, np.ones(k) / k, 'same')
    ip = f"{VD}/_at_in_{tag}_{MODE}.wav"; op = f"{VD}/at_{tag}_{MODE}.wav"; mp = f"{VD}/_pm_{tag}_{MODE}.txt"
    E.write(ip, x)
    with open(mp, 'w') as fh:
        fh.write("0 %.4f\n" % offs[0])
        for t, o in zip(times, offs): fh.write("%d %.4f\n" % (int(t * SR), o))
    args = ['--pitchmap', mp, '-p', str(base_shift)] + (['-F'] if formant else [])
    y = rubberband(ip, op, args, cache=False)
    return y, (times, f0, voiced, offs)

def stutter(x, slice_len, reps, gap=0.0, decay=0.0, fade_ms=3):
    """repeat the first slice_len seconds of x, reps times"""
    s = E.fade(x[:int(slice_len * SR)], fade_ms / 1000, fade_ms / 1000)
    g = np.zeros(int(gap * SR)); out = []
    for i in range(reps):
        out.append(s * (1 - decay) ** i); out.append(g)
    return np.concatenate(out)

if __name__ == '__main__':
    x = clean_master()
    print("clean len %.2fs peak %.2f" % (len(x) / SR, np.abs(x).max()))
    for k, (a, b) in {**WORDS, **PHRASES}.items():
        E.write(f"{VD}/w_{k}.wav", slice_(x, a, b))
    times, f0, voiced = yin_f0(x)
    print("yin: voiced frames %d/%d, median f0 %.0f" % (voiced.sum(), len(voiced), np.median(f0[voiced])))
    for t, f, v in zip(times[::4], f0[::4], voiced[::4]):
        if v: print("  %.2f %.0f Hz %s" % (t, f, ['G','G#','A','Bb','B','C','C#','D','Eb','E','F','F#'][(round(69+12*np.log2(f/440))-55)%12]), end=' |')
    print()
    y, info = autotune(x, 'full_scale')
    print("autotune full len %.2fs" % (len(y) / SR))
