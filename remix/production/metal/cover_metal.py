"""Metal cover: black/charcoal, cracked-steel texture, red card on fire, brutal typography."""
import numpy as np, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W = 1400; out = sys.argv[1]
F = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; F2 = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
rng = np.random.default_rng(9)
y = np.linspace(0, 1, W)[:, None]; x = np.linspace(0, 1, W)[None, :]
base = 14 + 6 * (1 - y)
# scratched steel texture
tex = rng.normal(0, 1, (W, W)); tex = np.array(Image.fromarray(((tex - tex.min()) / (tex.max() - tex.min()) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))) / 255.0
streak = np.cumsum(rng.normal(0, 1, (W, 1)), 0); streak = (streak - streak.min()) / (streak.max() - streak.min())
r = base + 25 * tex + 10 * streak; g = base + 22 * tex + 8 * streak; b = base + 24 * tex + 12 * streak
# red/orange ember glow from below
d = np.sqrt(((x - 0.5) * 0.9) ** 2 + ((y - 1.12) * 1.1) ** 2); glow = np.exp(-(d / 0.5) ** 2)
r = r + 170 * glow; g = g + 45 * glow; b = b + 8 * glow
img = np.clip(np.stack([r, g, b], -1), 0, 255)
im = Image.fromarray(img.astype(np.uint8)).convert('RGBA')
# embers
emb = Image.new('RGBA', (W, W), (0, 0, 0, 0)); de = ImageDraw.Draw(emb)
for _ in range(260):
    ex, ey = rng.uniform(0, W), rng.uniform(W * 0.35, W); s = rng.uniform(1, 4); a = int(rng.uniform(80, 230))
    de.ellipse([ex - s, ey - s, ex + s, ey + s], fill=(255, int(rng.uniform(120, 200)), 40, a))
emb = emb.filter(ImageFilter.GaussianBlur(0.8)); im = Image.alpha_composite(im, emb)
# cracks
cr = Image.new('RGBA', (W, W), (0, 0, 0, 0)); dc = ImageDraw.Draw(cr)
for _ in range(5):
    px, py = rng.uniform(0, W), rng.uniform(W * 0.3, W)
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.uniform(15, 40))):
        ang += rng.normal(0, 0.5); step = rng.uniform(10, 30)
        nx, ny = px + np.cos(ang) * step, py + np.sin(ang) * step
        dc.line([(px, py), (nx, ny)], fill=(0, 0, 0, 150), width=1); px, py = nx, ny
cr = cr.filter(ImageFilter.GaussianBlur(0.6))
im = Image.alpha_composite(im, cr)
# burning red card: fire + card drawn on one tall canvas, rotated together
CW, CH, FH = 340, 470, 300
layer = Image.new('RGBA', (CW, CH + FH), (0, 0, 0, 0)); dl = ImageDraw.Draw(layer)
for i in range(2600):
    fx_ = rng.uniform(25, CW - 25); h = abs(rng.normal(0, 1)) * (110 + 90 * np.sin(fx_ / 19.0) ** 2) + rng.uniform(0, 30)
    fy_ = FH + 18 - h; s_ = max(1.5, 13 * (1 - h / 260)) * rng.uniform(0.6, 1.3)
    heat = float(np.clip(1 - h / 230, 0, 1))
    col = (255, int(40 + 200 * heat), int(5 + 90 * heat ** 3), int(25 + 170 * heat))
    dl.ellipse([fx_ - s_, fy_ - s_ * 2.0, fx_ + s_, fy_ + s_ * 0.5], fill=col)
layer = layer.filter(ImageFilter.GaussianBlur(2.5))
dl = ImageDraw.Draw(layer)
dl.rounded_rectangle([0, FH, CW - 1, FH + CH - 1], 20, fill=(170, 18, 24, 255), outline=(255, 200, 160, 255), width=6)
# scorched top edge of the card
for i in range(300):
    ex, ey = rng.uniform(10, CW - 10), FH + abs(rng.normal(0, 14))
    dl.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(40, 10, 5, int(rng.uniform(60, 160))))
fc = ImageFont.truetype(F, 62); dl.text((CW / 2, FH + 165), "SUS-", font=fc, fill=(255, 235, 220), anchor='mm'); dl.text((CW / 2, FH + 240), "PENDED", font=fc, fill=(255, 235, 220), anchor='mm')
dl.text((CW / 2, FH + 370), "FROM THE GAME", font=ImageFont.truetype(F, 26), fill=(255, 190, 170), anchor='mm')
layer = layer.rotate(-11, expand=True, resample=Image.BICUBIC)
cx, cy = int(W * 0.50), int(W * 0.26)
sh = Image.new('RGBA', layer.size, (0, 0, 0, 0)); sh.paste((0, 0, 0, 170), [0, 0, layer.size[0], layer.size[1]], layer.split()[3]); sh = sh.filter(ImageFilter.GaussianBlur(16))
im.alpha_composite(sh, (cx + 16, cy + 22)); im.alpha_composite(layer, (cx, cy))
# typography
dr = ImageDraw.Draw(im)
big = ImageFont.truetype(F, 250); mid = ImageFont.truetype(F, 100); small = ImageFont.truetype(F, 44); tiny = ImageFont.truetype(F2, 28)
for dx, dy in [(6, 6), (-3, -3)]:
    dr.text((W - 70 + dx, 80 + dy), "מושעה", font=big, fill=(0, 0, 0, 200), anchor='ra', direction='rtl')
dr.text((W - 70, 80), "מושעה", font=big, fill=(236, 230, 222), anchor='ra', direction='rtl')
dr.line([(W - 70, 335), (W - 560, 335)], fill=(200, 30, 30), width=5)
dr.text((W - 74, 345), "מהמשחק", font=mid, fill=(220, 40, 40), anchor='ra', direction='rtl')
dr.text((70, W - 150), "METAL VERSION", font=small, fill=(236, 230, 222), anchor='ls')
dr.text((70, W - 100), "110 BPM  ·  DROP C  ·  C MINOR  ·  ORIGINAL VOICE RECORDING", font=tiny, fill=(190, 180, 175), anchor='ls')
dr.text((70, W - 60), "DOUBLE-TRACKED GUITARS · DOUBLE KICK · ORCHESTRA · CHOIR · BREAKDOWN", font=tiny, fill=(150, 140, 135), anchor='ls')
# vignette
vig = Image.new('L', (W, W), 0); dv = ImageDraw.Draw(vig); dv.ellipse([-W * 0.25, -W * 0.25, W * 1.25, W * 1.25], fill=255); vig = vig.filter(ImageFilter.GaussianBlur(220))
dark = Image.new('RGBA', (W, W), (0, 0, 0, 255)); dark.putalpha(Image.eval(vig, lambda p: 255 - p))
im = Image.alpha_composite(im, dark)
im.convert('RGB').save(out, quality=94); print("saved", out)
