"""Spectrogram + waveform image of a mix, with bar lines (72 BPM)."""
import numpy as np, soundfile as sf, sys
from scipy import signal
from PIL import Image, ImageDraw, ImageFont
path = sys.argv[1]; out = sys.argv[2]; bpm = 72.0
x, sr = sf.read(path); m = x.mean(1) if x.ndim == 2 else x
BAR = 240 / bpm
f, t, S = signal.spectrogram(m, sr, nperseg=2048, noverlap=1536, scaling='spectrum')
S = 10 * np.log10(S + 1e-12)
# log-frequency axis 30 Hz..16 kHz
fl = np.geomspace(30, 16000, 400)
idx = np.searchsorted(f, fl)
img = S[np.clip(idx, 0, len(f) - 1), :]
lo, hi = np.percentile(img, 5), np.percentile(img, 99.5)
img = np.clip((img - lo) / (hi - lo), 0, 1)
H, W = img.shape
W2 = 1800; img_r = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((W2, H), Image.BILINEAR)) / 255.0
# colormap (dark blue -> yellow)
r = np.clip(img_r * 2.2 - 0.6, 0, 1); g = np.clip(img_r * 1.6 - 0.25, 0, 1); b = np.clip(0.4 + img_r * 0.6 - (img_r > 0.6) * (img_r - 0.6) * 2, 0, 1) * (img_r > 0.05)
rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)[::-1]
im = Image.new('RGB', (W2, H + 140), (15, 15, 20)); im.paste(Image.fromarray(rgb), (0, 100))
d = ImageDraw.Draw(im); font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
dur = len(m) / sr
# waveform envelope
env = np.abs(m); k = int(len(env) / W2); e = env[:k * W2].reshape(W2, k).max(1); rms = np.sqrt((m[:k * W2].reshape(W2, k) ** 2).mean(1))
for i in range(W2):
    d.line([(i, 95 - int(e[i] * 85)), (i, 95)], fill=(90, 110, 160)); d.line([(i, 95 - int(rms[i] * 85)), (i, 95)], fill=(200, 160, 60))
nb = int(dur / BAR) + 1
for bar in range(1, nb + 1):
    xpix = int((bar - 1) * BAR / dur * W2)
    d.line([(xpix, 0), (xpix, H + 100)], fill=(255, 255, 255) if bar % 4 == 1 else (120, 120, 120), width=1)
    d.text((xpix + 3, H + 105), str(bar), fill=(230, 230, 230), font=font)
for fr in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    y = H + 100 - int(np.log(fr / 30) / np.log(16000 / 30) * H)
    d.line([(0, y), (W2, y)], fill=(70, 70, 70)); d.text((4, y - 14), f"{fr} Hz", fill=(200, 200, 200), font=font)
d.text((5, 2), path.split('/')[-1] + "  (top: waveform peak/RMS; bottom: log spectrogram; white lines every 4 bars)", fill=(230, 230, 230), font=font)
im.save(out); print("saved", out, im.size)
