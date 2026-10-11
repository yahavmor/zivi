"""QA for the metal mix: per-section loudness, spectral balance, per-stem activity (empty moments), L/R width, whisper check."""
import numpy as np, soundfile as sf, sys, os, json, subprocess
from scipy import signal
path = sys.argv[1]; stems = sys.argv[2] if len(sys.argv) > 2 else None
x, sr = sf.read(path); m = x.mean(1); BPM = 110; BAR = 240 / BPM
riffs = json.load(open(os.environ.get('METAL_RIFFS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'riffs.json'))))
ORDER = ['intro', 'first_entry', 'riffA', 'riffB', 'build', 'breakdown', 'solo', 'climax', 'outro']
starts = {}; b = 1
for s in ORDER: starts[s] = b; b += riffs['sections'][s]['bars']
def rms_db(seg): return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
print("== per-section RMS (dBFS) / crest / L-R corr")
for s in ORDER:
    a = int((starts[s] - 1) * BAR * sr); e = int(min(len(m), (starts[s] - 1 + riffs['sections'][s]['bars']) * BAR * sr)); seg = m[a:e]
    if len(seg) < sr: continue
    c = np.corrcoef(x[a:e, 0], x[a:e, 1])[0, 1]
    print("  %-12s bars %2d-%2d  RMS %6.1f  peak %5.2f  crest %4.1f dB  corr %.2f" % (s, starts[s], starts[s] + riffs['sections'][s]['bars'] - 1, rms_db(seg), np.abs(seg).max(), 20 * np.log10(np.abs(seg).max() / (np.sqrt((seg ** 2).mean()) + 1e-9)), c))
print("== per-bar RMS")
nb = int(len(m) / sr / BAR) + 1; line = ""
for bar in range(1, nb + 1):
    a = int((bar - 1) * BAR * sr); e = int(min(len(m), bar * BAR * sr))
    if e - a < sr * 0.5: break
    line += "b%02d:%5.1f " % (bar, rms_db(m[a:e]))
    if bar % 8 == 0: print("  " + line); line = ""
if line: print("  " + line)
f, P = signal.welch(m, sr, nperseg=8192); Pdb = 10 * np.log10(P + 1e-15); ref = np.mean(Pdb[(f >= 250) & (f < 500)])
bands = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
print("== spectral bands rel 250-500:", " ".join("%d-%d:%+.1f" % (a, b_, np.mean(Pdb[(f >= a) & (f < b_)]) - ref) for a, b_ in bands))
print("== overall L/R corr %.2f" % np.corrcoef(x[:, 0], x[:, 1])[0, 1])
# empty moments: 250 ms windows where RMS < -40 dBFS (outside intro/outro)
hop = int(0.25 * sr); quiet = []
for i in range(0, len(m) - hop, hop):
    if rms_db(m[i:i + hop]) < -40: quiet.append(i / sr)
print("== quiet windows (<-40 dBFS, 250 ms):", len(quiet), "at", [round(q, 1) for q in quiet[:40]])
if stems and os.path.isdir(stems):
    print("== stem RMS per section")
    names = [n[:-4] for n in sorted(os.listdir(stems)) if n.endswith('.wav')]
    data = {n: sf.read(os.path.join(stems, n + '.wav'))[0].mean(1) for n in names}
    hdr = "  %-12s" % "section" + "".join("%8s" % n for n in names); print(hdr)
    for s in ORDER:
        a = int((starts[s] - 1) * BAR * sr); e = int(min(len(m), (starts[s] - 1 + riffs['sections'][s]['bars']) * BAR * sr))
        print("  %-12s" % s + "".join("%8.1f" % rms_db(data[n][a:e]) for n in names))
