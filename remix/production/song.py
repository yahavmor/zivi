"""Arrangement: 'מושעה מהמשחק' — late-night R&B remix. 72 BPM, G minor, 26 bars."""
import numpy as np, os, sys, json
import engine as E, vocal as V, midi_render as M
SR = E.SR
BPM = 72.0; BEAT = 60 / BPM; BAR = 4 * BEAT; S16 = BEAT / 4
NBARS = 26; LEN = NBARS * BAR + 4.0
MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'   # main | instrumental | extra
os.environ['REMIX_MODE'] = MODE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
STEMS = f"{OUT}/stems_{MODE}"; os.makedirs(STEMS, exist_ok=True)
rng = np.random.default_rng(42)

def T(bar, beat=0.0): return (bar - 1) * BAR + beat * BEAT
SWING = 0.10  # fraction of a 16th that off-16ths are delayed
def swing_t(bar, beat):
    pos16 = round(beat * 4) ; off = (pos16 % 2 == 1)
    return T(bar, beat) + (SWING * S16 if off else 0.0)
def hum(ms=5): return rng.normal(0, ms / 1000)

CHORDS = {
    'Gm': dict(bass=43, keys=[58, 62, 65, 69], pad=[55, 58, 62, 65], top=[62, 65, 69, 74]),
    'Eb': dict(bass=39, keys=[55, 58, 62, 65], pad=[51, 55, 58, 62], top=[58, 62, 65, 70]),
    'Cm': dict(bass=36, keys=[55, 58, 62, 63], pad=[48, 51, 55, 58], top=[58, 62, 63, 67]),
    'F':  dict(bass=41, keys=[57, 60, 63, 67], pad=[45, 48, 53, 57], top=[60, 63, 67, 69]),
    'D7': dict(bass=38, keys=[54, 57, 60, 65], pad=[50, 54, 57, 60], top=[57, 60, 65, 66]),
}
BAR_CHORDS = ['Gm', 'Eb', 'Cm', 'D7',  'Gm', 'Eb', 'Cm', 'F',  'Gm', 'Eb', 'Cm', 'D7',
              'Gm', 'Eb', 'Cm', 'F',  'Gm', 'Eb', 'Cm', 'D7',  'Gm', 'Eb', 'Gm', 'Eb', 'Cm', 'Gm']
def chord(bar): return CHORDS[BAR_CHORDS[bar - 1]]
def cname(bar): return BAR_CHORDS[bar - 1]

# ---------------------------------------------------------------- tracks
tr = {k: E.Track(k, LEN) for k in ['drums', 'bass', 'keys', 'pad', 'vox', 'voxfx', 'fx', 'choir', 'oohs']}
kick_times = []

# ---------------------------------------------------------------- drums
def drums_bar(bar, style, g=0.0):
    """style: 'none','snaps','light','full','drop','fill','stop3' (stop after beat 3)"""
    d = tr['drums']
    if style == 'none': return
    if style == 'snaps':
        for b in [1, 3]: d.put(E.snap(0.7, seed=bar), T(bar, b) + hum(6), (-4) + g)
        return
    stop = (style == 'stop3')
    heavy = style in ('drop', 'fill')
    # kick
    kpat = [0, 1.75, 2.5] if bar % 2 == 1 else [0, 1.5, 2.5, 3.75]
    if style == 'drop' and bar % 4 == 0: kpat = [0, 1.75, 2.5, 3.5]
    for b in kpat:
        if stop and b >= 3: continue
        t = T(bar, b); d.put(E.kick(1.0 if b == 0 else 0.85, seed=bar), t, (0 if heavy else -2) + g); kick_times.append(t)
    # snare / clap / rim
    for b in [1, 3]:
        if stop and b >= 3: continue
        t = T(bar, b) + hum(3)
        if style == 'light':
            d.put(E.rim(0.8, seed=bar), t, -6); 
            if b == 3: d.put(E.snap(0.5, seed=bar + 3), t, (-10) + g)
        else:
            d.put(E.snare(1.0, seed=bar), t, (-3) + g); d.put(E.clap(0.9, seed=bar), t + 0.004, (-4) + g)
    # ghost snare
    if heavy: d.put(E.snare(0.35, seed=bar + 9), swing_t(bar, 2.75), (-16) + g)
    # hats
    hat_grid = np.arange(0, 4, 0.5) if style == 'light' else np.arange(0, 4, 0.25)
    for b in hat_grid:
        if stop and b >= 3: continue
        on8 = (round(b * 4) % 2 == 0); onbeat = (round(b * 4) % 4 == 0)
        v = 0.75 if onbeat else (0.55 if on8 else 0.33); v *= rng.uniform(0.9, 1.1)
        d.put(E.hat(v, seed=int(b * 4) + bar), swing_t(bar, b) + hum(3), (-4.5 if style != 'drop' else -3.5) + g, pan_=0.25)
    # open hat at the end of even bars
    if bar % 2 == 0 and not stop:
        d.put(E.hat(0.6, open_=True, seed=bar), swing_t(bar, 3.5), (-8) + g, pan_=0.25)
    # shaker (drop only)
    if style == 'drop':
        for b in np.arange(0, 4, 0.25):
            v = 0.5 if round(b * 4) % 2 == 0 else 0.3
            d.put(E.shaker(v * rng.uniform(0.85, 1.15), seed=int(b * 4)), swing_t(bar, b) + hum(4), (-11) + g, pan_=-0.4)
    # fill at the end
    if style == 'fill':
        for i, b in enumerate(np.arange(3.0, 4.0, 0.125)):
            d.put(E.hat(0.4 + 0.5 * i / 8, seed=i), T(bar, b), (-8) + g, pan_=0.25)
        for i, b in enumerate([3.0, 3.25, 3.5, 3.625, 3.75, 3.875]):
            d.put(E.snare(0.5 + 0.08 * i, seed=i + 50), T(bar, b), (-8) + g)

# ---------------------------------------------------------------- bass
BASS_BARS = {
    ('Gm', 'Eb'): [(0, 43, 1.4), (1.5, 43, .45), (2.0, 43, .85), (3.0, 46, .45), (3.5, 41, .45)],
    ('Gm', 'Gm'): [(0, 43, 1.4), (1.5, 43, .45), (2.0, 43, .85), (3.0, 46, .45), (3.5, 45, .45)],
    ('Eb', 'Cm'): [(0, 39, 1.4), (1.5, 39, .45), (2.0, 39, .85), (3.0, 43, .45), (3.5, 38, .45)],
    ('Eb', 'Gm'): [(0, 39, 1.4), (1.5, 39, .45), (2.0, 39, .85), (3.0, 41, .45), (3.5, 42, .45)],
    ('Cm', 'F'):  [(0, 36, 1.4), (1.5, 36, .45), (2.0, 43, .45), (2.5, 36, .45), (3.0, 39, .45), (3.5, 40, .45)],
    ('Cm', 'D7'): [(0, 36, 1.4), (1.5, 36, .45), (2.0, 43, .45), (2.5, 36, .45), (3.0, 39, .45), (3.5, 37, .45)],
    ('Cm', 'Gm'): [(0, 36, 1.4), (1.5, 36, .45), (2.0, 43, .45), (2.5, 36, .45), (3.0, 39, .45), (3.5, 41, .45)],
    ('F', 'Gm'):  [(0, 41, 1.0), (1.0, 41, .45), (1.5, 48, .45), (2.0, 41, .85), (3.0, 41, .45), (3.5, 42, .45)],
    ('D7', 'Gm'): [(0, 38, 1.4), (1.5, 38, .45), (2.0, 38, .85), (3.0, 42, .45), (3.5, 45, .45)],
}
def bass_bar(bar, style='groove', g=0.0):
    b = tr['bass']; c = cname(bar); nxt = cname(bar + 1) if bar < NBARS else 'Gm'
    if style == 'none': return
    if style == 'long':
        nb = chord(bar)['bass'] - 12; nb = nb if nb >= 29 else nb + 12
        b.put(E.bass(nb, BAR * 0.95, 0.7, growl=0.15, cutoff=400), T(bar), -6 + g); return
    if style == 'stop3':
        pat = [(p, n, d) for (p, n, d) in BASS_BARS.get((c, nxt), BASS_BARS[('Gm', 'Eb')]) if p < 3]
    else:
        pat = BASS_BARS.get((c, nxt), BASS_BARS[('Gm', 'Eb')])
    prev = None
    for (p, n, d) in pat:
        gl = prev if (prev is not None and abs(n - prev) <= 2 and p >= 3.0) else None
        nn = n - 12 if n - 12 >= 29 else n
        gln = (gl - 12 if gl - 12 >= 29 else gl) if gl is not None else None
        b.put(E.bass(nn, d * BEAT * 0.92, 0.95 if p == 0 else 0.8, glide_from=gln), T(bar, p) + hum(2), g)
        prev = n

# ---------------------------------------------------------------- keys / pad
def keys_bar(bar, style='comp', bright=1.0, g=0.0):
    k = tr['keys']; ch = chord(bar)
    if style == 'none': return
    pats = {
        'soft': [(0, 3.9, 0.5)],
        'comp': [(0, 1.4, 0.7), (1.5, 0.9, 0.55), (2.5, 1.0, 0.62), (3.5, 0.45, 0.5)],
        'stab': [(0, 0.7, 0.85), (1.5, 0.45, 0.6), (2.5, 0.9, 0.72), (3.75, 0.25, 0.55)],
        'hold': [(0, 3.95, 0.75)],
        'stop3': [(0, 1.4, 0.7), (1.5, 0.9, 0.55), (2.5, 0.45, 0.62)],
    }
    for (p, d, v) in pats[style]:
        notes = ch['keys'] if p < 2.5 or style != 'comp' else ch['top'][:3]
        k.put(E.rhodes_chord(notes, d * BEAT, v, strum=0.014 if style != 'stab' else 0.006, bright=bright), T(bar, p) + hum(4), -2 + g)
def pad_bar(bar, vel=0.5, attack=0.9, g=0.0):
    p = tr['pad']; ch = chord(bar)
    p.put(E.pad(ch['pad'], BAR * 1.02, vel, attack=attack, release=1.2, cutoff=1800), T(bar) - 0.05, g)
def rhodes_lick(t, notes, dur=0.22, vel=0.75, gap=0.0):
    k = tr['keys']
    for i, n in enumerate(notes):
        k.put(E.rhodes(n, dur, vel * rng.uniform(0.9, 1.05), release=0.5), t + i * (dur + gap), 0)

# ---------------------------------------------------------------- vocal helpers
clean = E.mono(E.read(f"{V.VD}/clean.wav"))
_vcache = {}
def W(name, semis=0.0, formant=True, stretch=1.0):
    """get a processed word/phrase slice (mono) with optional pitch shift / stretch, cached."""
    key = (name, round(semis, 2), formant, round(stretch, 3))
    if key in _vcache: return _vcache[key]
    a, b = {**V.WORDS, **V.PHRASES}[name]
    y = V.slice_(clean, a, b)
    if stretch != 1.0:
        ip = f"{V.VD}/_in_{name}_{MODE}.wav"; op = f"{V.VD}/st_{name}_{stretch:.2f}.wav"; E.write(ip, y)
        y = V.rubberband(ip, op, ['-t', str(stretch)] + (['-F'] if formant else []))
    if semis != 0.0:
        ip = f"{V.VD}/_in_{name}_{MODE}.wav"; op = f"{V.VD}/ps_{name}_{semis:+.1f}{'F' if formant else 'N'}_{stretch:.2f}.wav"; E.write(ip, y)
        y = V.rubberband(ip, op, ['-p', str(semis)] + (['-F'] if formant else []))
    y = E.fade(y, 0.003, 0.01)
    _vcache[key] = y
    return y
def vox(t, name, semis=0.0, formant=True, stretch=1.0, g=0.0, pan_=0.0, fx=None, track='vox'):
    """place a vocal slice. fx: None | 'pp' (ping-pong delay) | 'ghost' (lowpassed, -12) | 'tele' | 'whisper' | 'big' (huge reverb) | 'crush' | 'wide'"""
    y = W(name, semis, formant, stretch)
    if fx == 'ghost':
        y = E.lp(y, 2200, order=2); 
    elif fx == 'tele':
        y = E.telephone(y, 3.0); y = y / (np.abs(y).max() + 1e-9) * 0.8
    elif fx == 'whisper':
        y = E.lp(y, 2500, order=2)
    elif fx == 'crush':
        y = E.bitcrush(y, 7, 6); y = E.eq(y, ('lp', 6000))
    dest = tr['voxfx'] if fx in ('pp', 'big', 'whisper', 'tele', 'crush') else tr[track]
    if fx == 'pp':
        tr['vox'].put(y, t, g, pan_)
        tr['voxfx'].put(E.delay(y, BEAT * 0.75, 0.45, 1.0, pingpong=True, lowpass=3800, highpass=250)[len(y):] if False else E.delay(np.concatenate([y, np.zeros(int(BEAT * 6 * SR))]), BEAT * 0.75, 0.45, 1.0, pingpong=True, lowpass=3800, highpass=250) - E.stereo(np.concatenate([y, np.zeros(int(BEAT * 6 * SR))])), t, g - 6, 0)
        return y
    if fx == 'big':
        tr['voxfx'].put(E.reverb(y, decay=3.5, predelay=0.03, mix=1.0, width=1.2, seed=5) * 1.0, t, g, pan_); return y
    if fx == 'whisper':
        tr['voxfx'].put(E.reverb(y, decay=2.8, predelay=0.02, mix=0.8, width=1.3, seed=6), t, g, pan_); return y
    if fx in ('tele', 'crush'):
        tr['voxfx'].put(E.reverb(y, decay=0.5, predelay=0.005, mix=0.25, seed=7), t, g, pan_); return y
    dest.put(y, t, g, pan_)
    return y
def choir_hit(t, name='mushe1', stretch=1.8, shifts=(-3, 0, 4, 7), gains=(-5, -2, -4, -7), g=0.0, decay=3.5, fade_in=0.08):
    """harmonized, slowed, reverbed 'choir' from one word"""
    base = W(name, 0, True, stretch)
    base, _ = V.autotune(base, f"choir_{name}_{stretch}", max_shift=4)
    pans = [-0.6, 0.0, 0.6, -0.2]
    for s, gg, p in zip(shifts, gains, pans):
        y = base if s == 0 else V.pitch_shift(base, s, True, tag=f'ch{name}{stretch}')
        y = E.fade(y, fade_in, 0.05); y = E.lp(y, 5000)
        tr['choir'].put(E.reverb(y, decay=decay, predelay=0.02, mix=0.55, width=1.3, seed=8), t, g + gg, p)

# ---------------------------------------------------------------- SONG
def build():
    fx = tr['fx']
    # ===== INTRO bars 1-2
    for bar in [1, 2]:
        keys_bar(bar, 'soft', bright=0.8, g=-3); pad_bar(bar, 0.35, 1.2, g=-2); bass_bar(bar, 'long')
    vox(T(1, 2.0), 'p2', fx='whisper', g=-9, pan_=-0.3)
    vox(T(2, 2.0), 'mushe1', semis=-12, fx='whisper', g=-13, pan_=0.35)
    # reverse-reverb swell of the big word into bar 3
    y = W('meham3'); rv = E.reverb(y, decay=2.0, mix=1.0, seed=9); rvr = E.reverse(rv)
    fx.put(E.fade(rvr, 0.3, 0.02), T(3) - len(rvr) / SR + 0.02, -10)
    # ===== STATEMENT bars 3-4: the clean sentence; BOOM lands on bar 4 beat 1
    for bar in [3, 4]:
        keys_bar(bar, 'soft' if bar == 3 else 'hold', bright=0.9, g=-2); pad_bar(bar, 0.4, 0.8, g=-1); drums_bar(bar, 'snaps')
    bass_bar(3, 'long')
    t_full = T(4) - (V.WORDS['me3'][0] - V.PHRASES['full'][0])
    vox(t_full, 'p12', g=0, fx=None)
    vox(t_full + (V.PHRASES['p3'][0] - V.PHRASES['full'][0]), 'p3', g=0.5, fx='pp')
    fx.put(E.reverse_cymbal(1.4, 0.5), T(4) - 1.4, -8)
    tr['bass'].put(E.bass808(38, 2.6, 0.9, bend=2.2), T(4), -1)          # D1 808 under the BOOM
    tr['keys'].put(E.rhodes_chord(chord(4)['keys'], 1.2, 0.9, strum=0.004, bright=1.2), T(4), 0)
    fx.put(E.crash(0.5, 2.5), T(4), -14)
    vox(T(4), 'meham3', semis=-12, fx='ghost', g=-9)
    # ===== GROOVE A bars 5-8
    for bar in [5, 6, 7, 8]:
        drums_bar(bar, 'light' if bar < 7 else ('full' if bar < 8 else 'fill'), g=-2.5); bass_bar(bar); keys_bar(bar, 'comp', g=-1); pad_bar(bar, 0.3)
    # vocal: sparse chant
    def chant_sparse(bar):
        vox(T(bar, 0.0), 'ata1'); vox(T(bar, 0.75), 'mushe1', fx='pp', g=0)
        vox(T(bar, 2.0), 'ata2', g=-1, pan_=-0.2); vox(T(bar, 2.4), 'mushe2', g=-1, pan_=0.2)
        vox(T(bar + 1, 0.0), 'meham1', g=0)
        vox(T(bar + 1, 2.0), 'ata3', semis=3, g=-5, pan_=0.5); vox(T(bar + 1, 2.45), 'mushe3', semis=3, g=-5, pan_=0.5, fx='pp')
    chant_sparse(5); chant_sparse(7)
    rhodes_lick(T(6, 3.0), [74, 77, 79], 0.2, 0.6); rhodes_lick(T(8, 2.5), [79, 77, 74, 70], 0.18, 0.65)
    vox(T(8, 3.25), 'mu1', g=-2); vox(T(8, 3.5), 'mu1', g=-2); vox(T(8, 3.75), 'mushe1', g=0)   # stutter into comedy
    # ===== COMEDY bars 9-12
    for bar in [9, 10, 11, 12]:
        drums_bar(bar, {9: 'full', 10: 'light', 11: 'full', 12: 'stop3'}[bar], g=-2); bass_bar(bar, 'stop3' if bar == 12 else 'groove')
        keys_bar(bar, {9: 'comp', 10: 'hold', 11: 'comp', 12: 'stop3'}[bar]); pad_bar(bar, 0.35 if bar != 12 else 0.5)
    # bar 9: referee whistle + PA announcement + rising cartoon stutter
    fx.put(E.whistle(0.45, vel=0.7), T(9, 0.0), -9, pan_=0.3)
    vox(T(9, 0.6), 'p1', fx='tele', g=-1)
    for i, b in enumerate([3.0, 3.25, 3.5, 3.75]):
        y = E.resample_speed(W('mu1'), 1.0 + 0.09 * i)
        tr['vox'].put(E.mono(y), T(9, b), -3 + i, 0.3 * (i - 1.5))
    # bar 10: gospel choir on 'מושעה', bass-singer 'מהמשחק' an octave down
    choir_hit(T(10, 0.0), 'mushe1', 1.8, g=-3)
    tr['keys'].put(E.rhodes_chord(chord(10)['keys'] + [70], 2.4, 0.8, strum=0.05), T(10, 0.0), -1)
    vox(T(10, 2.5), 'meham1', semis=-12, g=-1, fx=None); vox(T(10, 2.5), 'meham1', semis=-12, fx='big', g=-10)
    # bar 11: call and response
    vox(T(11, 0.0), 'ata2', g=0); vox(T(11, 0.4), 'mushe2', g=0)
    rhodes_lick(T(11, 1.0), [70, 74, 77, 79], 0.17, 0.8)
    vox(T(11, 2.0), 'ata2', semis=5, g=-2, pan_=0.4); vox(T(11, 2.4), 'mushe2', semis=5, g=-2, pan_=0.4, fx='pp')
    rhodes_lick(T(11, 3.0), [79, 82, 79, 77], 0.17, 0.8)
    # bar 12: build, riser, stop at beat 3, dry pickup
    fx.put(E.riser(BAR * 0.75, 0.5, 300, 6000), T(12, 0.0), -12)
    for i, b in enumerate(np.arange(2.0, 3.0, 0.125)):
        tr['drums'].put(E.hat(0.4 + 0.6 * i / 8, seed=i), T(12, b), -9, pan_=0.25)
    fx.put(E.reverse_cymbal(1.6, 0.6), T(13) - 1.6, -9)
    vox(T(13) - (V.WORDS['me3'][0] - V.PHRASES['p3a'][0]), 'p3a', g=1.0)          # pickup "אתה מושעה"
    # ===== MAIN DROP bars 13-16
    for bar in [13, 14, 15, 16]:
        drums_bar(bar, 'drop' if bar < 16 else 'fill', g=1); bass_bar(bar, g=1); keys_bar(bar, 'stab', bright=1.1, g=2); pad_bar(bar, 0.45, 0.3, g=2)
    tr['bass'].put(E.bass808(31, 2.2, 1.0, bend=2.0), T(13), -1)                # G1 808
    fx.put(E.crash(0.6, 2.5), T(13), -11)
    vox(T(13), 'meham3', g=1.5, fx='pp'); vox(T(13), 'meham3', semis=-12, fx='ghost', g=-7)
    vox(T(13), 'meham3', semis=7, g=-12, pan_=-0.5); vox(T(13), 'meham3', semis=4, g=-12, pan_=0.5)
    choir_hit(T(13, 0.0), 'chak3', 1.6, shifts=(-5, 0, 4, 7), gains=(-6, -4, -5, -8), g=-6, decay=2.5)
    # GM "Voice Oohs" pad under the drop
    oohs = []
    for bar in [13, 14, 15, 16]:
        for n in chord(bar)['pad'][1:]: oohs.append((T(bar) + 0.02, n + 12, BAR * 0.95, 0.5))
    tr['oohs'].put(E.reverb(E.lp(M.render(oohs, 53, f"{STEMS}/_oohs.wav", total_len=LEN), 4500), decay=2.5, mix=0.5, seed=12), 0, -7)
    def chant_dense(bar, var=0):
        vox(T(bar, 2.0), 'ata2', g=0, pan_=-0.15); vox(T(bar, 2.4), 'mushe2', g=0, pan_=0.15, fx='pp')
        vox(T(bar, 3.25), 'mu1', g=-3); vox(T(bar, 3.5), 'mu1', g=-3); vox(T(bar, 3.75), 'mu1', g=-3)
        vox(T(bar + 1, 0.0), 'mushe1', g=0.5); vox(T(bar + 1, 0.0), 'mushe1', semis=4, g=-9, pan_=0.5); vox(T(bar + 1, 0.0), 'mushe1', semis=-5, g=-9, pan_=-0.5)
        vox(T(bar + 1, 1.0), 'ata3', g=0); vox(T(bar + 1, 1.5), 'mushe3', g=0)
        vox(T(bar + 1, 2.0), 'ata1', g=0); vox(T(bar + 1, 2.75), 'mushe1', semis=(3 if var else 0), g=-1, pan_=0.3 if var else 0, fx='pp')
        vox(T(bar + 1, 3.5), 'ata3', g=0)
        vox(T(bar + 2) - (V.WORDS['me3'][0] - V.WORDS['mushe3'][0]), 'mushe3', g=0)
    chant_dense(13)
    # bar 15: BOOM again with a telephone-y crushed answer
    vox(T(15), 'meham3', g=1.0, fx='pp'); vox(T(15), 'meham3', semis=-12, fx='ghost', g=-8)
    tr['bass'].put(E.bass808(36, 1.6, 0.9, bend=2.0), T(15), -2)
    vox(T(15, 2.0), 'p2', fx='crush', g=-2, pan_=0.3)
    vox(T(15, 3.0), 'ata2', semis=-7, g=-1); vox(T(15, 3.4), 'mushe2', semis=-7, g=-1)
    vox(T(16, 0.0), 'meham1', g=0.5, fx='pp')
    vox(T(16, 2.0), 'ata1', g=0); vox(T(16, 2.5), 'mushe1', g=0)
    vox(T(16, 3.25), 'mu1', g=-2); vox(T(16, 3.5), 'mu1', semis=2, g=-2); vox(T(16, 3.75), 'mu1', semis=4, g=-2)
    # ===== DROP 2: the diva (auto-tuned sung sentence) bars 17-20
    for bar in [17, 18, 19, 20]:
        drums_bar(bar, 'drop' if bar < 20 else 'stop3', g=0.5); bass_bar(bar, 'stop3' if bar == 20 else 'groove', g=0.5); keys_bar(bar, 'comp' if bar < 20 else 'stop3', g=1); pad_bar(bar, 0.45, 0.3, g=1)
    full = W('full'); o = V.PHRASES['full'][0]
    def rel(name): a, b = V.WORDS[name]; return (a - o, b - o)
    mel = [  # (t0,t1,midi) within the full phrase
        (*rel('ata1'), 58), (rel('mu1')[0], rel('mu1')[1], 60), (*rel('she1'), 62), (*rel('meham1'), 60),
        (*rel('ata2'), 55), (*rel('mushe2'), 58),
        (*rel('ata3'), 60), (*rel('mushe3'), 62), (*rel('me3'), 63), (*rel('ha3'), 63), (*rel('mis3'), 62), (rel('chak3')[0], rel('chak3')[0] + 0.3, 58), (rel('chak3')[0] + 0.3, rel('chak3')[1], 55),
    ]
    diva, info = V.autotune(full, 'diva', overrides=mel, max_shift=6)
    diva = E.fade(diva, 0.005, 0.02)
    t_d = T(18) - (V.WORDS['me3'][0] - o) + 0.0
    tr['vox'].put(diva, t_d, 1.0); tr['voxfx'].put(E.reverb(diva, decay=2.2, predelay=0.04, mix=1.0, width=1.2, seed=13), t_d, -9)
    tr['voxfx'].put(E.delay(np.concatenate([diva, np.zeros(int(BEAT * 6 * SR))]), BEAT * 0.75, 0.4, 1.0, pingpong=True, lowpass=3500, highpass=300) - E.stereo(np.concatenate([diva, np.zeros(int(BEAT * 6 * SR))])), t_d, -9)
    tr['bass'].put(E.bass808(31, 2.0, 0.9, bend=2.0), T(18), -2)
    # answer: deep narrator + high "sidekick"
    vox(T(19, 2.0), 'ata2', semis=-7, g=-1, pan_=-0.3); vox(T(19, 2.4), 'mushe2', semis=-7, g=-1, pan_=-0.3)
    vox(T(19, 3.0), 'ata2', semis=5, g=-3, pan_=0.4); vox(T(19, 3.4), 'mushe2', semis=5, g=-3, pan_=0.4, fx='pp')
    # bar 20: D7#9 tension, drums stop, tape-stop on 'מהמשחק'
    vox(T(20, 0.0), 'ata1', g=0); vox(T(20, 0.75), 'mushe1', g=0)
    ts = E.tape_stop(E.stereo(W('meham1')), 0.25, 0.9, curve=1.6)
    tr['vox'].put(ts, T(20, 1.75), 0); tr['voxfx'].put(E.reverb(ts, decay=2.0, mix=1.0, seed=14), T(20, 1.75), -10)
    # ===== BREAKDOWN + FINAL bars 21-24
    for bar in [21, 22]:
        keys_bar(bar, 'soft', bright=0.85, g=-2); pad_bar(bar, 0.45, 0.8, g=-1); bass_bar(bar, 'long'); drums_bar(bar, 'snaps')
    vox(T(21, 0.3), 'p1', g=0, fx='pp')
    choir_hit(T(21, 2.5), 'mushe2', 2.0, shifts=(-3, 0, 4), gains=(-6, -3, -5), g=-5, decay=3.0)
    vox(T(22, 0.0), 'ata2', semis=-12, g=-4, fx='whisper'); vox(T(22, 0.35), 'mushe2', semis=-12, g=-4, fx='whisper')
    rhodes_lick(T(22, 1.0), [70, 74, 77, 79, 82], 0.16, 0.7)
    fx.put(E.riser(BAR * 0.6, 0.45, 400, 7000), T(22, 1.4), -13)
    fx.put(E.reverse_cymbal(1.5, 0.6), T(23) - 1.5, -9)
    vox(T(23) - (V.WORDS['me3'][0] - V.PHRASES['p3a'][0]), 'p3a', g=1.0)
    for bar in [23, 24]:
        drums_bar(bar, 'drop' if bar == 23 else 'fill', g=1); bass_bar(bar, g=1); keys_bar(bar, 'stab', bright=1.15, g=2); pad_bar(bar, 0.5, 0.3, g=2)
    tr['bass'].put(E.bass808(31, 2.4, 1.0, bend=2.2), T(23), -1)
    fx.put(E.crash(0.7, 3.0), T(23), -10)
    vox(T(23), 'meham3', g=2.0, fx='pp'); vox(T(23), 'meham3', semis=-12, fx='ghost', g=-6)
    for s, gg, p in [(-5, -10, -0.6), (4, -10, 0.6), (7, -12, -0.3), (12, -15, 0.3)]:
        vox(T(23), 'meham3', semis=s, g=gg, pan_=p)
    choir_hit(T(23, 0.0), 'chak3', 1.6, shifts=(-5, 0, 4, 7), gains=(-5, -3, -4, -7), g=-5, decay=3.0)
    aahs = [(T(23) + 0.02, n + 12, BAR * 1.9, 0.6) for n in chord(23)['pad'][1:]] + [(T(24) + 0.02, n + 12, BAR * 0.9, 0.55) for n in chord(24)['pad'][1:]]
    tr['oohs'].put(E.reverb(E.lp(M.render(aahs, 52, f"{STEMS}/_aahs.wav", total_len=LEN), 4500), decay=3.0, mix=0.5, seed=15), 0, -7)
    chant_dense(23, var=1)
    # bar 24 end: everything stops on beat 4 → final cadence
    # ===== OUTRO bars 25-26
    keys_bar(25, 'soft', bright=0.8); pad_bar(25, 0.4, 0.6); bass_bar(25, 'long')
    tr['keys'].put(E.rhodes_chord(chord(26)['keys'] + [74], 5.5, 0.6, strum=0.05, release=2.5), T(26), 0)
    tr['pad'].put(E.pad(chord(26)['pad'], 5.0, 0.35, attack=0.6, release=2.5), T(26), 0)
    tr['bass'].put(E.bass(31, 4.5, 0.6, growl=0.1, cutoff=350, release=1.5), T(26), -3)
    vox(T(25, 1.0), 'ata2', g=-1); vox(T(25, 1.4), 'mushe2', g=-1, fx='pp')
    vox(T(25, 3.0), 'mushe1', semis=-12, g=-5, fx='whisper')
    fx.put(E.bell(79, 2.5, 0.5), T(26, 1.0), -14, pan_=0.2); fx.put(E.bell(86, 2.0, 0.3), T(26, 1.25), -18, pan_=-0.2)
    # the final 'ק' click as the very last sound
    tr['vox'].put(W('k_click'), T(26, 3.0), 0); tr['voxfx'].put(E.reverb(W('k_click'), decay=1.5, mix=1.0, seed=16), T(26, 3.0), -8)

    if MODE == 'extra':
        # bar 6: DJ scratch on "מו"
        sc = E.scratch(W('mushe1'), [(0, 0.35, 0.11), (0.35, 0.05, 0.1), (0.05, 0.4, 0.12), (0.4, 0.1, 0.09), (0.1, 0.6, 0.16), (0.6, 0.0, 0.14), (0.0, 1.0, 0.45)])
        tr['vox'].put(sc, T(6, 2.5), -1, 0.2)
        # bar 11: chipmunk sidekick instead of the +5 answer (formant NOT preserved)
        vox(T(11, 2.0), 'ata2', semis=7, formant=False, g=-1, pan_=0.45); vox(T(11, 2.4), 'mushe2', semis=7, formant=False, g=-1, pan_=0.45, fx='pp')
        # bars 14-15: the final 'ק' click as a percussion tick on off-16ths
        for bar in [14, 15]:
            for b in np.arange(0.25, 4, 0.5):
                tr['drums'].put(W('k_click') * 0.8, swing_t(bar, b), -9, pan_=-0.3)
        # bar 15: whistle on the second BOOM
        fx.put(E.whistle(0.35, f=3100, vel=0.7), T(15, 0.0) - 0.05, -10, pan_=-0.3)
        # bar 19: slow-motion giant "מהמשחק" (half speed, formant drops = monster)
        slow = E.resample_speed(E.stereo(W('meham3')), 0.55)
        tr['vox'].put(slow, T(19, 0.0), -2); tr['voxfx'].put(E.reverb(slow, decay=2.5, mix=1.0, seed=31), T(19, 0.0), -9)
        # bar 22: vinyl rewind into the final drop
        rw = E.resample_speed(E.reverse(E.stereo(W('p1'))), 1.9); rw = E.fade(rw, 0.02, 0.05)
        tr['voxfx'].put(rw, T(23) - len(rw) / SR - 0.62, -4)
        # outro: whispered 'מושעה' one more time, pitched down two octaves (the verdict is final)
        vox(T(26, 1.5), 'mushe1', semis=-19, g=-3, fx='whisper', stretch=1.4)
    # vinyl bed throughout, fading out
    vn = E.vinyl_noise(LEN, -40)
    env = np.ones(len(vn)); env[int(T(5) * SR):int(T(21) * SR)] = 0.45; env[int(T(26, 2) * SR):] = np.linspace(1, 0, len(env) - int(T(26, 2) * SR)) ** 0.5
    fx.put((vn.T * env).T, 0, 0)

# ---------------------------------------------------------------- MIX
def mix_and_master(tag):
    d = tr['drums'].get(); b = tr['bass'].get(); k = tr['keys'].get(); p = tr['pad'].get(); v = tr['vox'].get(); vf = tr['voxfx'].get(); f = tr['fx'].get(); c = tr['choir'].get(); oo = tr['oohs'].get()
    # drums bus
    d = E.compress(E.eq(d, ('hp', 35), ('peak', 60, 1.0, 1.5), ('peak', 250, 1.2, -1.5), ('hs', 9000, 0.7, 1.5)), -14, 2.5, 0.004, 0.09, makeup_db=1)
    d = d + E.reverb(d, decay=0.6, predelay=0.008, mix=1.0, seed=20) * 0.12
    d = E.saturate(d, 1.3, 0.4)
    # bass: mono, controlled
    b = E.eq(E.mono(b), ('hp', 34, 0.8), ('hp', 34, 0.8), ('lp', 3500))
    b = E.compress(b, -14, 3.0, 0.008, 0.15, makeup_db=1)
    b = E.duck(b, kick_times, 0.35, 0.004, 0.02, 0.18)
    b = E.stereo(b)
    # keys: tremolo pan + chorus + plate
    k = E.eq(k, ('hp', 110), ('peak', 350, 1.0, -2), ('peak', 2500, 1.0, 1.0), ('ls', 180, 0.7, 1))
    k = E.tremolo_pan(E.chorus(k, 0.5, 0.0025, 0.3), 4.6, 0.35)
    k = E.compress(k, -20, 2.5, 0.01, 0.2, makeup_db=2)
    k = k + E.reverb(k, decay=1.6, predelay=0.02, mix=1.0, seed=21) * 0.25
    k = E.duck(k, kick_times, 0.18, 0.005, 0.03, 0.2)
    # pad
    p = E.eq(p, ('hp', 150), ('lp', 6000), ('peak', 400, 1.0, -2))
    p = p + E.reverb(p, decay=3.0, predelay=0.03, mix=1.0, width=1.3, seed=22) * 0.35
    p = E.widen(p, 0.3, 9)
    p = E.duck(p, kick_times, 0.3, 0.005, 0.03, 0.25)
    # vocals: EQ, compression, short plate, light slap
    v = E.eq(v, ('hp', 90), ('peak', 280, 1.2, -1.5), ('peak', 3200, 1.0, 1.5), ('hs', 7000, 0.7, 1.0))
    v = E.compress(v, -16, 3.5, 0.006, 0.1, makeup_db=2)
    v = v + E.reverb(v, decay=1.3, predelay=0.025, mix=1.0, width=1.1, lowcut=250, seed=23) * 0.2
    v = E.saturate(v, 1.2, 0.3)
    vf = E.eq(vf, ('hp', 150), ('lp', 7000))
    c = E.widen(E.eq(c, ('hp', 180), ('lp', 6000)), 0.3, 11)
    oo = E.widen(E.eq(oo, ('hp', 200), ('lp', 5000)), 0.35, 13)
    # bus levels (dB)
    L = dict(drums=-5.5, bass=-12, keys=-8, pad=-10, vox=-2.5, voxfx=-6, fx=-8, choir=-5, oohs=-3)
    if tag == 'instrumental':
        L['vox'] = -120; L['voxfx'] = -120; L['choir'] = -120
    if tag == 'extra':  # more comedy: louder fx/choir, more ghost
        L['choir'] += 3; L['fx'] += 2; L['voxfx'] += 2
    allb = [('drums', d), ('bass', b), ('keys', k), ('pad', p), ('vox', v), ('voxfx', vf), ('fx', f), ('choir', c), ('oohs', oo)]
    mix = sum(E.gain(x, L[n]) for n, x in allb)
    for n, x in allb:
        if tag == 'main': E.write(f"{STEMS}/{n}.wav", E.gain(x, L[n]))
    # master chain
    m = E.eq(mix, ('hp', 30, 0.8), ('hp', 30, 0.8), ('peak', 300, 1.0, -1.0), ('peak', 2500, 1.0, 1.0), ('peak', 4000, 1.0, 1.0), ('hs', 8000, 0.7, 3.5))
    m = E.compress(m, -12, 1.5, 0.02, 0.25, knee=8, makeup_db=0)
    m = E.saturate(m, 1.15, 0.25)
    # loudness normalize to target via measurement
    tmp = f"{OUT}/_pre_{tag}.wav"; E.write(tmp, m)
    meas = E.measure(tmp); target = -14.0
    g = target - meas['lufs']
    m = E.limiter(E.gain(m, g), -1.0, 0.004, 0.12)
    # trim tail: find last sample above -70 dB, add 0.5 s
    a = np.abs(m).max(1); idx = np.where(a > 1e-4)[0]; end = min(len(m), (idx[-1] if len(idx) else len(m)) + int(0.6 * SR))
    m = m[:end]
    return m, meas

if __name__ == '__main__':
    import time; t0 = time.time()
    build(); print("built in %.0fs" % (time.time() - t0))
    m, meas = mix_and_master(MODE)
    name = {'main': 'mushe_remix', 'instrumental': 'mushe_remix_instrumental', 'extra': 'mushe_remix_extra_comedy'}[MODE]
    E.write(f"{OUT}/{name}_48k.wav", m)
    print("pre-master:", meas, " final len %.1fs peak %.3f" % (len(m) / SR, np.abs(m).max()))
    print("final:", E.measure(f"{OUT}/{name}_48k.wav"))
    print("done in %.0fs" % (time.time() - t0))
