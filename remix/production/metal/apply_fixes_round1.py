"""Round-1 fixes (vocal review V1-V6 + producer notes). Applies to metal_song.py and riffs.json. Idempotent-ish: run once."""
import json, re
s = open('metal_song.py').read()
def rep(a, b, must=True):
    global s
    if a not in s:
        if must: raise SystemExit("MISSING: " + a[:80])
        return
    s = s.replace(a, b)
# V1: gentler vocal bus compression + post-compressor section gain so the final shout is the biggest
rep("v = E.compress(v, -16, 4.0, 0.004, 0.08, makeup_db=3); v = E.saturate(v, 1.4, 0.35)",
    """v = E.compress(v, -12, 2.0, 0.004, 0.08, makeup_db=1.5); v = E.saturate(v, 1.4, 0.35)
VOX_SEC_GAIN = {'first_entry': -1.0, 'climax': 0.5, 'outro': 2.5}
venv = np.ones(len(v))
for s_, g_ in VOX_SEC_GAIN.items(): venv[int(ST(s_) * SR):int((ST(s_) + RIFFS['sections'][s_]['bars'] * BAR) * SR)] = E.db(g_)
k_ = int(0.06 * SR); venv = np.convolve(venv, np.ones(k_) / k_, 'same'); v = (v.T * venv[:len(v)]).T""")
# V2a: lead ducks under the voice
rep("ld = E.eq(ld, ('hp', 150), ('peak', 2500, 1.0, 1.5), ('lp', 8000))",
    "ld = E.eq(ld, ('hp', 150), ('peak', 2500, 1.0, 1.5), ('lp', 8000))\nif MODE != 'instrumental': ld = E.compress(ld, -20, 3.0, 0.012, 0.18, knee=4, sidechain=vkey)")
# vkey must exist before the lead line: it is defined in the gtr block which comes first (check order below)
# V2c: chop gains: 'ata' chops +3, 'mushe' chops +1.5 (riffB list and climax list)
rep("""        vox(nm, 'clean', ST('riffB') + BAR * bo + BEAT * bt, -1, 0.25 * (-1) ** k)""",
    """        vox(nm, 'clean', ST('riffB') + BAR * bo + BEAT * bt, 3.0 if nm.startswith('ata') else 1.5, 0.25 * (-1) ** k)""")
rep("""        vox(nm, 'double', ST('climax') + BAR * bo + BEAT * bt, 0)""",
    """        vox(nm, 'double', ST('climax') + BAR * bo + BEAT * bt, 3.0 if nm.startswith('ata') else 1.5)""")
# V3: build drop pickup: dry p3a + hall p3 behind it
rep("""    vox('p3', 'hall', ST('build') - SHOUT + off('ata3') + 0.0, -2)            # "אתה מושעה מֵהמשחק" with the shout on 'build' downbeat""",
    """    vox('p3a', 'clean', ST('build') - 0.61, 1.0)                                # dry pickup into the build downbeat
    vox('p3', 'hall', ST('build') - 0.61, -4, track='voxfx')""")
# V4: overlaps
rep("""    for bar_off in (3, 7):
        vox('p2', 'double', ST('riffA') + BAR * bar_off + BEAT * 2.5, 0)""",
    """    vox('p2', 'double', ST('riffA') + BAR * 3 + BEAT * 2.5, 0)
    vox('ata2', 'double', ST('riffA') + BAR * 7 + BEAT * 2.5, 0)                    # 'a-ta' then the mu-mu-mu stutter""")
rep("vox('p12', 'radio', ST('build') + BAR * 1 + BEAT * 0.5, -1)", "vox('p1', 'radio', ST('build') + BAR * 1 + BEAT * 0.5, -1)")
rep("""    for k, (bo, bt, nm) in enumerate([(0, 0, 'ata1'), (0, 0.5, 'mushe1'), (1, 2.5, 'ata2'), (1, 3.0, 'mushe2'), (2, 0, 'ata1'), (2, 0.5, 'mushe1'), (3, 1.5, 'meham1')]):""",
    """    for k, (bo, bt, nm) in enumerate([(0, 0, 'ata2'), (0, 0.5, 'mushe2'), (1, 2.5, 'ata2'), (1, 3.0, 'mushe2'), (2, 0, 'ata2'), (2, 0.5, 'mushe2'), (3, 1.5, 'meham1')]):""")
# V5: comedy pacing in the climax: drop the last stutter + last chop pair → breath before the final shout
rep("""    for k, (bo, bt, nm) in enumerate([(5, 2.5, 'ata2'), (5, 3.0, 'mushe2'), (6, 0, 'ata1'), (6, 0.5, 'mushe1'), (6, 2.5, 'ata2'), (6, 3.0, 'mushe2')]):""",
    """    for k, (bo, bt, nm) in enumerate([(5, 2.5, 'ata2'), (5, 3.0, 'mushe2'), (6, 0, 'ata2'), (6, 0.5, 'mushe2')]):""")
rep("""    vox('mu1', 'stutter', ST('climax') + BAR * 7 + BEAT * 2.0, -1, reps=6, grid=0.125)\n""", "")
rep("""    vox('mu1', 'stutter', ST('breakdown') + BAR * 7 + BEAT * 2.0, -2, reps=8, grid=0.125)\n""", "")
# V6: reverse swells end on the word they introduce
rep("vox('meham3', 'reverse_swell', ST('first_entry'), -8, 0, track='voxfx')", "vox('p2', 'reverse_swell', ST('first_entry') + BEAT * 0.5, -8, 0, track='voxfx')")
rep("vox('meham3', 'reverse_swell', ST('solo'), -9, track='voxfx')", "vox('mushe1', 'reverse_swell', ST('solo') + BAR * 2 + BEAT * 2.5, -9, track='voxfx')")
# producer note 1: the drone must not fill the stop at the end of the build
rep("put('fx', O.drone(36, BAR * 4), ST('build'), -12)", "put('fx', E.fade(O.drone(36, BAR * 3 + BEAT * 2.5), 0.3, 0.15), ST('build'), -12)")
# producer note 2: deliberate ending: final low impact after the last click, then silence
rep("tr['vox'].put(E.stereo(E.read(os.path.join(HERE, '..', 'vox', 'w_k_click.wav'))), ST('outro') + BAR * 3 + BEAT * 2, 0)",
    "tr['vox'].put(E.stereo(E.read(os.path.join(HERE, '..', 'vox', 'w_k_click.wav'))), ST('outro') + BAR * 3 + BEAT * 2, 0)\n    put('fx', O.impact(0.9), ST('outro') + BAR * 3 + BEAT * 3, -5); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -6)")
open('metal_song.py', 'w').write(s)
# riffs.json edits: V2b (climax lead 22<=beat<30 deleted), V3 (final drop pickups removed, orch chords shortened)
d = json.load(open('riffs.json')); c = d['sections']['climax']
n0 = len(c['lead']); c['lead'] = [e for e in c['lead'] if not (22 <= e['beat'] < 30)]
nb = len(c['bass']); c['bass'] = [e for e in c['bass'] if e['beat'] not in (31.0, 31.25, 31.5, 31.75)]
ng = len(c['guitar']); c['guitar'] = [e for e in c['guitar'] if e['beat'] != 31.75]
for key in ('strings', 'brass', 'choir'):
    for e in c.get(key, []):
        if e['beat'] == 28.0 and e['dur'] == 4.0: e['dur'] = 3.0
json.dump(d, open('riffs.json', 'w'))
print("riffs: climax lead %d->%d bass %d->%d guitar %d->%d" % (n0, len(c['lead']), nb, len(c['bass']), ng, len(c['guitar'])))
print("metal_song.py patched")
