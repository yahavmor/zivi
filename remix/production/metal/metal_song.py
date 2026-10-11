"""METAL version — arrangement assembly. Reads riffs.json (judged arrangement) + module renders + vocal placement."""
import numpy as np, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E, vocal as V
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('METAL_OUT', os.path.join(HERE, '..', '..', 'out', 'metal')); os.makedirs(OUT, exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'   # main | instrumental
SR = E.SR; BPM = 110.0; BEAT = 60 / BPM; BAR = 4 * BEAT
RIFFS = json.load(open(os.environ.get('METAL_RIFFS', os.path.join(HERE, 'riffs.json'))))
ORDER = ['intro', 'first_entry', 'riffA', 'riffB', 'build', 'breakdown', 'solo', 'climax', 'outro']
SEC_START = {}; _b = 1
for s in ORDER:
    SEC_START[s] = _b; _b += RIFFS['sections'][s]['bars']
NBARS = _b - 1; LEN = NBARS * BAR + 6.0
def T(bar, beat=0.0): return (bar - 1) * BAR + beat * BEAT
def ST(sec, beat=0.0): return T(SEC_START[sec], beat)
rng = np.random.default_rng(11)
print("sections:", {s: SEC_START[s] for s in ORDER}, "bars", NBARS, "len %.1fs" % LEN)

# ---------------------------------------------------------------- collect absolute events from riffs.json
gtr, bass, drums, lead, clean, orch = [], [], [], [], [], []
for s in ORDER:
    sec = RIFFS['sections'][s]; t0 = ST(s)
    for e in sec.get('guitar', []):  gtr.append((t0 + e['beat'] * BEAT, int(e['root']), float(e['dur']) * BEAT, float(e['vel']), bool(e.get('mute', False))))
    for e in sec.get('bass', []):    bass.append((t0 + e['beat'] * BEAT, int(e['midi']), float(e['dur']) * BEAT, float(e['vel']), e.get('slide_from'), bool(e.get('mute', False))))
    for e in sec.get('drums', []):   drums.append((t0 + e['beat'] * BEAT + rng.normal(0, 0.001 if e['name'] in ('kick', 'snare') else 0.0025), e['name'], float(e['vel'])))
    for e in sec.get('lead', []):    lead.append((t0 + e['beat'] * BEAT, int(e['midi']), float(e['dur']) * BEAT, float(e['vel']), float(e.get('bend', 0) or 0), float(e.get('vibrato', 0) or 0), e.get('slide_from')))
    for e in sec.get('clean', []):   clean.append((t0 + e['beat'] * BEAT, int(e['midi']), float(e['dur']) * BEAT, float(e['vel'])))
    for key in ('orch', 'strings', 'choir', 'brass'):
        for e in sec.get(key, []):
            prog = int(e.get('program', {'strings': 48, 'choir': 52, 'brass': 61, 'orch': 48}[key]))
            notes = e['midi'] if isinstance(e['midi'], list) else [e['midi']]
            for n in notes: orch.append((t0 + e['beat'] * BEAT, int(n), float(e['dur']) * BEAT, float(e.get('vel', 0.7)), prog))
print("events: guitar %d bass %d drums %d lead %d clean %d orch %d" % (len(gtr), len(bass), len(drums), len(lead), len(clean), len(orch)))

tr = {k: E.Track(k, LEN) for k in ['gtr', 'lead', 'clean', 'bass', 'drums', 'orch', 'fx', 'vox', 'voxfx']}
timing = {}
def timed(name, fn):
    t = time.time(); r = fn(); timing[name] = time.time() - t; print("  %s rendered in %.0fs" % (name, timing[name])); return r

# ---------------------------------------------------------------- instruments
import importlib
G = importlib.import_module(os.environ.get('METAL_GUITAR_MODULE', 'metal_guitar'))
import metal_drums as DR, metal_bass as B, metal_orch_fx as O, metal_vocal as MV
tr['gtr'].put(timed('rhythm guitars', lambda: G.render_rhythm_pair(gtr, total_len=LEN)), 0, 0)
if lead: tr['lead'].put(timed('lead guitar', lambda: G.render_lead(lead, total_len=LEN)), 0, 0)
if clean: tr['clean'].put(timed('clean guitar', lambda: G.render_clean(clean, total_len=LEN)), 0, 0)
tr['bass'].put(timed('bass', lambda: B.render_bass(bass, total_len=LEN)), 0, 0)
tr['drums'].put(timed('drums', lambda: DR.render_kit(drums, total_len=LEN)), 0, 0)
by_prog = {}
for (t, n, d, v, p) in orch: by_prog.setdefault(p, []).append((t, n, d, v))
for p, evs in by_prog.items():
    tr['orch'].put(timed('orch prog %d' % p, lambda: O.orch(evs, p, total_len=LEN)), 0, 0)

# ---------------------------------------------------------------- cinematic FX layer (placed by section)
fx = tr['fx']
def put(track, x, t, g=0.0, p=0.0): tr[track].put(x, t, g, p)
put('fx', O.noise_bed(ST('first_entry') + 1.0), 0, -16)
put('fx', O.drone(36, ST('riffA') - 0.5), 0, -14)
put('fx', O.riser(BAR * 2, 200, 6000), ST('first_entry') - BAR * 2, -12)
put('fx', O.reverse_cymbal(1.5), ST('first_entry') - 1.5, -10)
put('fx', O.impact(0.9), ST('first_entry'), -8)
put('fx', O.riser(BAR * 2, 300, 8000), ST('riffA') - BAR * 2, -10)
put('fx', O.reverse_cymbal(1.2), ST('riffA') - 1.2, -9)
put('fx', O.impact(1.0), ST('riffA'), -4)
put('fx', O.sub_drop(1.2, 70, 25), ST('riffA') - 0.02, -6)
put('fx', O.impact(0.8), ST('riffB'), -7)
put('fx', O.downlifter(2.0), ST('riffB'), -14)
put('fx', O.riser(BAR * 3, 150, 9000), ST('breakdown') - BAR * 3, -9)
put('fx', E.fade(O.drone(36, BAR * 3 + BEAT * 2.5), 0.3, 0.15), ST('build'), -12)
put('fx', O.reverse_cymbal(1.6), ST('breakdown') - 1.6, -8)
put('fx', O.sub_drop(1.6, 80, 22), ST('breakdown') - 0.02, -3)
put('fx', O.impact(1.0), ST('breakdown'), -3)
for k in range(4): put('fx', O.metal_hit(0.7, 420 + 60 * k), ST('breakdown') + BAR * (2 * k + 1) + BEAT * 3.5, -12, 0.5 * (-1) ** k)
put('fx', O.industrial_clank(1), ST('breakdown') + BAR * 3.5, -14, -0.4)
put('fx', O.industrial_clank(2), ST('breakdown') + BAR * 7.5, -14, 0.4)
put('fx', O.riser(BAR * 2, 200, 9000), ST('solo') - BAR * 2, -12)
put('fx', O.impact(0.8), ST('solo'), -8)
put('fx', O.riser(BAR * 4, 120, 10000), ST('climax') - BAR * 4, -8)
put('fx', O.reverse_cymbal(2.0), ST('climax') - 2.0, -8)
put('fx', O.impact(1.0), ST('climax'), -3)
put('fx', O.sub_drop(1.5, 75, 24), ST('climax') - 0.02, -4)
put('fx', O.braam(36, BAR * 2), ST('climax') + BAR * 4, -10)
put('fx', O.impact(1.0), ST('outro'), -3)
put('fx', O.sub_drop(2.0, 70, 20), ST('outro') - 0.02, -4)
put('fx', E.fade(O.drone(36, BAR * 3 + BEAT * 3.2), 0.4, 1.8), ST('outro'), -10)   # fades into the final hit, which is the last sound
put('fx', O.impact(1.0), ST('outro') + BAR * 3 + BEAT * 3, -1); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -3)   # the last hit (both modes)

# ---------------------------------------------------------------- VOCALS
o_full = V.PHRASES['full'][0]
def off(name): return V.WORDS[name][0] - o_full          # offset of a word inside 'full'
SHOUT = off('me3')                                       # 3.14 s into the full sentence
def vox(name, kind, t, g=0.0, p=0.0, track='vox', **kw):
    x = MV.variant(name, kind, **kw)
    if isinstance(x, tuple): x, lead_in = x; t = t - lead_in
    tr[track].put(x, t, g, p)
if MODE != 'instrumental':
    # intro: a whisper from the void + reverse swell into the first entry
    vox('p2', 'hall', ST('intro') + BAR * 2, -10, -0.3)
    vox('mushe1', 'demon', ST('intro') + BAR * 3 + BEAT * 2, -14, 0.3, track='voxfx')
    vox('p2', 'reverse_swell', ST('first_entry') + BEAT * 0.5, -8, 0, track='voxfx')
    # first entry: "אתה מושעה" call at bar 5, then the full sentence so the shout lands on riffA's downbeat
    vox('p2', 'hall', ST('first_entry') + BEAT * 0.5, -1, 0)
    vox('p2', 'double', ST('first_entry') + BAR + BEAT * 0.5, -1, 0)
    vox('full', 'clean', ST('riffA') - SHOUT, 0)
    vox('full', 'hall', ST('riffA') - SHOUT, -7, track='voxfx')
    # riffA downbeat: the shout, stacked
    vox('meham3', 'shout_stack', ST('riffA'), 1.5)
    vox('meham3', 'hall', ST('riffA'), -6, track='voxfx')
    # answers in the light windows (bars 4 and 8 of riffA → beats 2.5)
    vox('p2', 'double', ST('riffA') + BAR * 3 + BEAT * 2.5, 0)
    vox('ata2', 'double', ST('riffA') + BAR * 7 + BEAT * 2.5, 0)                    # 'a-ta' then the mu-mu-mu stutter
    vox('mu1', 'stutter', ST('riffA') + BAR * 7 + BEAT * 3.0, -1, reps=4, grid=0.25)
    # riffB: rhythmic chops + a delayed "מהמשחק"
    for k, (bo, bt, nm) in enumerate([(0, 0, 'ata2'), (0, 0.5, 'mushe2'), (1, 2.5, 'ata2'), (1, 3.0, 'mushe2'), (2, 0, 'ata2'), (2, 0.5, 'mushe2'), (3, 1.5, 'meham1')]):
        vox(nm, 'clean', ST('riffB') + BAR * bo + BEAT * bt, 3.0 if nm.startswith('ata') else 1.5, 0.25 * (-1) ** k)
    vox('meham1', 'hall', ST('riffB') + BAR * 3 + BEAT * 1.5, -6, track='voxfx')
    x = MV.variant('meham1', 'clean'); tr['voxfx'].put(E.delay(np.concatenate([x, np.zeros((int(BEAT * 6 * SR), 2))]), BEAT * 0.75, 0.45, 1.0, pingpong=True, lowpass=4000) - np.concatenate([x, np.zeros((int(BEAT * 6 * SR), 2))]), ST('riffB') + BAR * 3 + BEAT * 1.5, -5)
    vox('p3a', 'clean', ST('build') - 0.61, 1.0)                                # dry pickup into the build downbeat
    vox('p3', 'hall', ST('build') - 0.61, -4, track='voxfx')
    vox('meham3', 'megaphone', ST('build'), -1)
    # build: radio sentence, glitch, then silence and a dry pickup
    vox('p1', 'radio', ST('build') + BAR * 1 + BEAT * 0.5, -1)
    vox('mushe1', 'glitch', ST('build') + BAR * 2 + BEAT * 0.0, -2, seed=3)
    vox('mu1', 'stutter', ST('build') + BAR * 2 + BEAT * 2.0, -2, reps=8, grid=0.125)
    vox('p3a', 'clean', ST('breakdown') - (off('me3') - off('ata3')), 1.0)   # dry, alone, right before the drop
    # BREAKDOWN downbeat: demon + shout
    vox('meham3', 'demon', ST('breakdown'), 1.0)
    vox('meham3', 'shout_stack', ST('breakdown'), -1)
    vox('meham3', 'hall', ST('breakdown'), -5, track='voxfx')
    vox('ata2', 'demon', ST('breakdown') + BAR * 2 + BEAT * 2.0, -1); vox('mushe2', 'demon', ST('breakdown') + BAR * 2 + BEAT * 2.6, -1)
    vox('meham1', 'slowmo', ST('breakdown') + BAR * 4 + BEAT * 2.0, -2)
    vox('p2', 'megaphone', ST('breakdown') + BAR * 6 + BEAT * 0.0, -2, 0.3)
    vox('mushe1', 'reverse_swell', ST('solo') + BAR * 1 + BEAT * 2.0, -9, track='voxfx')
    # solo: one answer
    vox('mushe1', 'hall', ST('solo') + BAR * 1 + BEAT * 2.0, -4, 0.3)
    # CLIMAX: shout on the downbeat, full sentence hall+double, then chops, final shout on the outro downbeat
    vox('meham3', 'shout_stack', ST('climax'), 2.0)
    vox('meham3', 'hall', ST('climax'), -4, track='voxfx')
    vox('full', 'double', ST('climax') + BAR * 4 - SHOUT, 0.5)
    vox('full', 'hall', ST('climax') + BAR * 4 - SHOUT, -6, track='voxfx')
    vox('meham3', 'shout_stack', ST('climax') + BAR * 4, 1.0)
    for k, (bo, bt, nm) in enumerate([(5, 2.5, 'ata2'), (5, 3.0, 'mushe2'), (6, 0, 'ata2'), (6, 0.5, 'mushe2')]):
        vox(nm, 'double', ST('climax') + BAR * bo + BEAT * bt, 3.0 if nm.startswith('ata') else 1.5)
    vox('p3a', 'clean', ST('outro') - (off('me3') - off('ata3')), 1.0)
    vox('meham3', 'shout_stack', ST('outro'), 2.5)
    vox('meham3', 'demon', ST('outro'), -3)
    vox('meham3', 'hall', ST('outro'), -3, track='voxfx')
    vox('mushe1', 'demon', ST('outro') + BAR * 2 + BEAT * 2, -8, track='voxfx')
    tr['vox'].put(E.stereo(E.read(os.path.join(HERE, '..', 'vox', 'w_k_click.wav'))), ST('outro') + BAR * 3 + BEAT * 2, -6)

# ---------------------------------------------------------------- MIX
def bus(name): return tr[name].get()
g = bus('gtr'); ld = bus('lead'); cl = bus('clean'); b = bus('bass'); d = bus('drums'); o = bus('orch'); f = bus('fx'); v = bus('vox'); vf = bus('voxfx')
kick_times = [t for (t, n, vel) in drums if n == 'kick']
# ---------------------------------------------------------------- section dynamics (automation)
SEC_GAIN = {'intro': -2.0, 'first_entry': -3.0, 'riffA': 0.0, 'riffB': -1.0, 'build': -1.0, 'breakdown': 0.5, 'solo': -1.0, 'climax': 1.5, 'outro': 0.5}
def section_env(n):
    env = np.ones(n)
    for s in ORDER:
        a = int(ST(s) * SR); b_ = int(min(n, (ST(s) + RIFFS['sections'][s]['bars'] * BAR) * SR)); env[a:b_] = E.db(SEC_GAIN[s])
    k = int(0.06 * SR); env = np.convolve(env, np.ones(k) / k, 'same')
    return env
senv = section_env(len(g))
def auto_(x): return (np.asarray(x).T * senv[:len(x)]).T

# rhythm guitars: tight low end, controlled fizz, hard L/R already
g = E.eq(g, ('hp', 95, 0.8), ('hp', 95, 0.8), ('peak', 180, 1.2, -1.0), ('peak', 450, 1.0, -1.0), ('peak', 3200, 1.0, 1.0), ('lp', 9500, 0.8), ('lp', 9500, 0.8))
g = E.compress(g, -12, 2.0, 0.01, 0.12)
g = E.duck(g, kick_times, 0.12, 0.003, 0.01, 0.08)
if MODE != 'instrumental':   # vocal-keyed ducking: guitars (and orchestra) step back ~2 dB while the voice speaks
    vkey = E.mono(tr['vox'].get()); vkey = E.eq(vkey, ('hp', 300), ('lp', 5000))
    g = E.compress(g, -14, 1.2, 0.012, 0.18, knee=4, sidechain=vkey)
# lead: centre-ish, delay + plate
ld = E.eq(ld, ('hp', 150), ('peak', 2500, 1.0, 1.5), ('lp', 8000))
if MODE != 'instrumental': ld = E.compress(ld, -12, 2.0, 0.012, 0.18, knee=4, sidechain=vkey)
_solo_env = np.ones(len(ld)); _solo_env[int(ST('solo') * SR):int(ST('climax') * SR)] = E.db(3.0); _k = int(0.06 * SR); _solo_env = np.convolve(_solo_env, np.ones(_k) / _k, 'same'); ld = (ld.T * _solo_env[:len(ld)]).T
ld = ld + E.delay(ld, BEAT * 0.75, 0.35, 1.0, pingpong=True, lowpass=4500) * 0.25 + E.reverb(ld, decay=1.8, predelay=0.02, mix=1.0, seed=41) * 0.25
cl = cl + E.reverb(cl, decay=2.5, mix=1.0, width=1.3, seed=42) * 0.4
# bass: mono, sub tight, grit audible
b = E.eq(E.mono(b), ('hp', 34, 0.8), ('peak', 65, 1.2, 2.0), ('peak', 900, 1.0, 2.0), ('peak', 2200, 1.0, 1.0), ('lp', 6000)); b = E.compress(b, -14, 4.0, 0.006, 0.1, makeup_db=1); b = E.stereo(b)
b = E.duck(b, kick_times, 0.25, 0.003, 0.015, 0.1)
_benv = np.ones(len(b)); _benv[int(ST('breakdown') * SR):int(ST('solo') * SR)] = E.db(1.5); _k = int(0.06 * SR); _benv = np.convolve(_benv, np.ones(_k) / _k, 'same'); b = (b.T * _benv[:len(b)]).T
# drums bus: punch + parallel comp
d = E.eq(d, ('hp', 30), ('peak', 250, 1.2, -1.5), ('peak', 3500, 1.0, 1.5), ('hs', 10000, 0.7, 4.5))
d = d * 0.75 + E.compress(d, -24, 6.0, 0.002, 0.08, makeup_db=4) * 0.35
# orchestra: hall, sits behind
o = E.eq(o, ('hp', 120), ('peak', 500, 1.0, -1.5), ('lp', 9000)); o = E.duck(o, kick_times, 0.15, 0.004, 0.02, 0.12)
if MODE != 'instrumental': o = E.compress(o, -14, 1.2, 0.012, 0.2, knee=4, sidechain=vkey)
# vocals: metal chain — bright, compressed, saturated; sits on top
v = E.eq(v, ('hp', 110), ('peak', 300, 1.2, -2.0), ('peak', 1800, 1.0, 1.0), ('peak', 3500, 1.0, 0.5), ('hs', 8000, 0.7, 2.0))
v = E.compress(v, -12, 2.0, 0.004, 0.08, makeup_db=1.5); v = E.saturate(v, 1.4, 0.35)
dek = E.mono(E.eq(v, ('hp', 4500, 0.7), ('hp', 4500, 0.7))); v = E.compress(v, -12, 3.0, 0.002, 0.06, knee=4, sidechain=dek)   # de-esser
v = E.limiter(v, -6.0, 0.002, 0.05)
VOX_SEC_GAIN = {'first_entry': -1.0, 'build': -2.0, 'climax': 1.5, 'outro': 2.5}
venv = np.ones(len(v))
for s_, g_ in VOX_SEC_GAIN.items(): venv[int(ST(s_) * SR):int((ST(s_) + RIFFS['sections'][s_]['bars'] * BAR) * SR)] = E.db(g_)
k_ = int(0.06 * SR); venv = np.convolve(venv, np.ones(k_) / k_, 'same'); v = (v.T * venv[:len(v)]).T
vf = E.eq(vf, ('hp', 200), ('lp', 7000))
f = E.eq(f, ('hp', 30, 0.8), ('peak', 60, 1.0, -2.0))
L = dict(gtr=-6.0, lead=-12.0, clean=-12.0, bass=-7.0, drums=-5.0, orch=-12.0, fx=-9.0, vox=-2.0, voxfx=-7.0)
if MODE == 'instrumental': L['vox'] = -120; L['voxfx'] = -120
allb = [('gtr', auto_(g)), ('lead', auto_(ld)), ('clean', cl), ('bass', auto_(b)), ('drums', auto_(d)), ('orch', auto_(o)), ('fx', f), ('vox', v), ('voxfx', vf)]
STEMS = os.path.join(OUT, 'stems_' + MODE); os.makedirs(STEMS, exist_ok=True)
mix = 0
for n, x in allb:
    xs = E.gain(x, L[n]); E.write(os.path.join(STEMS, n + '.wav'), xs); mix = mix + xs

# master: glue, tilt, true-peak limit, loudness target for metal
m = E.eq(mix, ('hp', 28, 0.8), ('hp', 28, 0.8), ('peak', 4000, 1.2, -2.0), ('hs', 9000, 0.7, 4.0))
m = E.compress(m, -10, 1.6, 0.03, 0.25, knee=8)
m = E.saturate(m, 1.2, 0.3)
pre = os.path.join(OUT, '_pre_%s.wav' % MODE); E.write(pre, m); meas = E.measure(pre)
target = -9.5; g0 = target - meas['lufs']; tmp = os.path.join(OUT, '_lim_%s.wav' % MODE)
for _ in range(4):
    m2 = E.limiter(E.gain(m, g0), -1.0, 0.004, 0.12); E.write(tmp, m2); err = target - E.measure(tmp)['lufs']
    if abs(err) <= 0.15: break
    g0 += err
m = m2; print("master gain %.2f dB" % g0)
a = np.abs(m).max(1); idx = np.where(a > 1e-3)[0]; m = m[:min(len(m), idx[-1] + SR)]; m[-int(0.2 * SR):] *= np.linspace(1, 0, int(0.2 * SR))[:, None]
name = {'main': 'mushe_metal', 'instrumental': 'mushe_metal_instrumental'}[MODE]
E.write(os.path.join(OUT, name + '_48k.wav'), m)
print("pre-master:", meas, "| final:", E.measure(os.path.join(OUT, name + '_48k.wav')), "| len %.1fs" % (len(m) / SR))
print("timing:", {k: round(v) for k, v in timing.items()})
