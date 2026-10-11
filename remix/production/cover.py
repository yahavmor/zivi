"""Cover art: night-time R&B vinyl vibe with a referee red card joke."""
import numpy as np, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W = 1400; out = sys.argv[1]
F = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; F2 = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
# background gradient navy -> plum
y = np.linspace(0, 1, W)[:, None]; x = np.linspace(0, 1, W)[None, :]
r = 10 + 40 * y + 10 * x; g = 12 + 10 * y; b = 36 + 40 * y + 15 * (1 - x)
# warm amber glow bottom-right
d = np.sqrt((x - 0.78) ** 2 + (y - 0.72) ** 2); glow = np.exp(-(d / 0.42) ** 2)
r = r + 215 * glow; g = g + 110 * glow; b = b + 25 * glow
# cool glow top-left
d2 = np.sqrt((x - 0.15) ** 2 + (y - 0.12) ** 2); glow2 = np.exp(-(d2 / 0.35) ** 2)
r = r + 20 * glow2; g = g + 60 * glow2; b = b + 140 * glow2
img = np.clip(np.stack([r, g, b], -1), 0, 255)
rng = np.random.default_rng(5); img += rng.normal(0, 6, img.shape)  # grain
im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
# vinyl record
rec = Image.new('RGBA', (W, W), (0, 0, 0, 0)); dr = ImageDraw.Draw(rec)
cx, cy, R = int(W * 0.66), int(W * 0.60), int(W * 0.42)
dr.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(14, 12, 20, 235))
for i in range(60):
    rr = R - 14 - i * (R * 0.62 / 60); a = 28 + int(20 * (i % 3 == 0))
    dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(120, 110, 140, a), width=2)
lab = int(R * 0.34); dr.ellipse([cx - lab, cy - lab, cx + lab, cy + lab], fill=(226, 120, 44, 255))
dr.ellipse([cx - lab + 6, cy - lab + 6, cx + lab - 6, cy + lab - 6], outline=(255, 200, 120, 255), width=3)
dr.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(14, 12, 20, 255))
# light reflection on the record
refl = Image.new('RGBA', (W, W), (0, 0, 0, 0)); dref = ImageDraw.Draw(refl)
dref.pieslice([cx - R, cy - R, cx + R, cy + R], 200, 250, fill=(255, 240, 220, 38)); dref.pieslice([cx - R, cy - R, cx + R, cy + R], 20, 70, fill=(255, 240, 220, 26))
refl = refl.filter(ImageFilter.GaussianBlur(18)); rec = Image.alpha_composite(rec, refl)
im = Image.alpha_composite(im.convert('RGBA'), rec)
# red card (referee) — rotated
card = Image.new('RGBA', (300, 420), (0, 0, 0, 0)); dc = ImageDraw.Draw(card)
dc.rounded_rectangle([0, 0, 299, 419], 22, fill=(214, 36, 48, 255), outline=(255, 235, 235, 255), width=6)
fc = ImageFont.truetype(F, 54); dc.text((150, 150), "SUS-", font=fc, fill=(255, 245, 245), anchor='mm'); dc.text((150, 215), "PENDED", font=fc, fill=(255, 245, 245), anchor='mm')
fc2 = ImageFont.truetype(F2, 26); dc.text((150, 330), "from the game", font=fc2, fill=(255, 220, 220), anchor='mm')
card = card.rotate(14, expand=True, resample=Image.BICUBIC)
sh = Image.new('RGBA', card.size, (0, 0, 0, 0)); sh.paste((0, 0, 0, 140), [0, 0, card.size[0], card.size[1]], card.split()[3]); sh = sh.filter(ImageFilter.GaussianBlur(14))
im.alpha_composite(sh, (int(W * 0.05) + 12, int(W * 0.50) + 18)); im.alpha_composite(card, (int(W * 0.05), int(W * 0.50)))
# titles (Hebrew, RTL via raqm)
d = ImageDraw.Draw(im)
big = ImageFont.truetype(F, 230); mid = ImageFont.truetype(F, 96); small = ImageFont.truetype(F2, 40); tiny = ImageFont.truetype(F2, 28)
d.text((W - 70 + 5, 95 + 6), "מושעה", font=big, fill=(0, 0, 0, 120), anchor='ra', direction='rtl')
d.text((W - 70, 95), "מושעה", font=big, fill=(255, 246, 232), anchor='ra', direction='rtl')
d.text((W - 76, 340), "מהמשחק", font=mid, fill=(255, 190, 110), anchor='ra', direction='rtl')
d.text((W - 76, 455), "L A T E   N I G H T   R & B   E D I T", font=small, fill=(220, 215, 235), anchor='ra')
d.text((70, W - 70), "72 BPM  ·  G MINOR  ·  ORIGINAL VOICE RECORDING", font=tiny, fill=(200, 190, 215), anchor='ls')
d.text((70, W - 110), "Side A", font=small, fill=(255, 200, 130), anchor='ls')
im.convert('RGB').save(out, quality=94); print("cover saved", out)
