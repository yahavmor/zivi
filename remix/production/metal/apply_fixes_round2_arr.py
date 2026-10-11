"""Round-2 arrangement fixes (ARR-1..ARR-6) on riffs.json + the solo answer move / solo lead boost in metal_song.py."""
import json
d = json.load(open('riffs.json')); S = d['sections']; log = []
def rm(sec, layer, pred, label):
    before = len(S[sec][layer]); S[sec][layer] = [e for e in S[sec][layer] if not pred(e)]; log.append("%s.%s %s: removed %d" % (sec, layer, label, before - len(S[sec][layer])))
def add(sec, layer, ev): S[sec].setdefault(layer, []).append(ev); log.append("%s.%s added %s" % (sec, layer, ev))
def near(a, b, tol=1e-6): return abs(a - b) < tol
# ARR-1 climax lead: resolve the phrase, fill the free window, bar 44 held note
add('climax', 'lead', {"beat": 22.0, "midi": 84, "dur": 0.5, "vel": 0.95, "bend": 0.0, "vibrato": 0.3, "slide_from": None})
for b, m, vb in [(26.0, 77, 0.3), (27.0, 74, 0.4), (28.0, 75, 0.3), (29.0, 74, 0.3)]:
    add('climax', 'lead', {"beat": b, "midi": m, "dur": 1.0, "vel": 0.9, "bend": 0.0, "vibrato": vb, "slide_from": None})
rm('climax', 'lead', lambda e: any(near(e['beat'], b) for b in (12.0, 13.0, 13.5, 14.0)), 'bar44 moving line')
add('climax', 'lead', {"beat": 12.0, "midi": 79, "dur": 1.0, "vel": 0.8, "bend": 0.0, "vibrato": 0.4, "slide_from": None})
# ARR-2 not confirmed by the verifier: no riffs.json change; instead orch sidechain ratio 2.5 + climax vocal +1.5 (metal_song.py, below)
# ARR-3 fill vocabulary
rm('riffB', 'drums', lambda e: (e['name'] == 'snare' and 31.0 - 1e-6 <= e['beat'] <= 31.875 + 1e-6) or (e['name'] == 'kick' and any(near(e['beat'], b) for b in (31.25, 31.5, 31.75))), 'bar24 old fill')
for b, n, v in [(31.0, 'snare', 0.95), (31.25, 'tom_hi', 0.9), (31.5, 'tom_mid', 0.9), (31.75, 'tom_floor', 0.95), (31.75, 'kick', 1.0)]: add('riffB', 'drums', {"beat": b, "name": n, "vel": v})
rm('breakdown', 'drums', lambda e: e['name'] == 'snare' and any(near(e['beat'], b) for b in (15.5, 15.625)), 'bar32 2-note pickup')
# ARR-3: solo bar 40 / outro bar 50 edits skipped (verifier: optional / not duplicates)
# ARR-4 riffB pads up an octave + choir louder
for e in S['riffB'].get('strings', []):
    if e['dur'] == 4.0: e['midi'] = [m + 12 for m in e['midi']]; e['vel'] = 0.85
for e in S['riffB'].get('choir', []): e['vel'] = min(1.0, e['vel'] + 0.2)
log.append('riffB pads +12, choir +0.2')
# ARR-6 final hit stack
rm('outro', 'drums', lambda e: near(e['beat'], 8.0) and e['name'] in ('crash1', 'crash2'), 'final stack')
json.dump(d, open('riffs.json', 'w')); print("\n".join(log))
# metal_song.py: ARR-5 solo answer moved onto held notes + lead +3 dB in the solo
s = open('metal_song.py').read()
a = "vox('mushe1', 'reverse_swell', ST('solo') + BAR * 2 + BEAT * 2.5, -9, track='voxfx')"; assert a in s
s = s.replace(a, "vox('mushe1', 'reverse_swell', ST('solo') + BAR * 1 + BEAT * 2.0, -9, track='voxfx')")
a = "vox('mushe1', 'hall', ST('solo') + BAR * 2 + BEAT * 2.5, -4, 0.3)"; assert a in s
s = s.replace(a, "vox('mushe1', 'hall', ST('solo') + BAR * 1 + BEAT * 2.0, -4, 0.3)")
a = "if MODE != 'instrumental': ld = E.compress(ld, -20, 3.0, 0.012, 0.18, knee=4, sidechain=vkey)"; assert a in s
s = s.replace(a, a + "\n_solo_env = np.ones(len(ld)); _solo_env[int(ST('solo') * SR):int(ST('climax') * SR)] = E.db(3.0); _k = int(0.06 * SR); _solo_env = np.convolve(_solo_env, np.ones(_k) / _k, 'same'); ld = (ld.T * _solo_env[:len(ld)]).T")
a = "if MODE != 'instrumental': o = E.compress(o, -14, 1.5, 0.012, 0.2, knee=4, sidechain=vkey)"; assert a in s
s = s.replace(a, "if MODE != 'instrumental': o = E.compress(o, -14, 2.5, 0.012, 0.2, knee=4, sidechain=vkey)")
s = s.replace("'climax': 0.5, 'outro': 2.5}", "'climax': 1.5, 'outro': 2.5}")
open('metal_song.py', 'w').write(s); print("metal_song.py: solo answer moved, solo lead +3 dB, orch sidechain 2.5, climax vox +1.5")
