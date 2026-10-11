"""Round-2 mix fixes (MIX-1..MIX-6) for metal_song.py (+ band-delivered final hit in riffs.json)."""
import json
s = open('metal_song.py').read()
def rep(a, b):
    global s
    assert a in s, "MISSING: " + a[:70]; s = s.replace(a, b)
# MIX-1 low end: restore the bass 2nd harmonic, drop the master 80 Hz cut, breakdown bass +1.5
rep("('hp', 34, 0.8), ('ls', 110, 0.7, -2.5), ('peak', 900, 1.0, 2.0)", "('hp', 34, 0.8), ('peak', 65, 1.2, 2.0), ('peak', 900, 1.0, 2.0)")
rep("('peak', 80, 1.0, -2.0), ('peak', 350, 1.0, -1.5), ('peak', 2800, 1.0, 1.0)", "('peak', 4000, 1.2, -2.0)")   # also MIX-2 (2800 -> 4000 cut) and MIX-6 (350 cut removed)
rep("b = E.duck(b, kick_times, 0.25, 0.003, 0.015, 0.1)",
    "b = E.duck(b, kick_times, 0.25, 0.003, 0.015, 0.1)\n_benv = np.ones(len(b)); _benv[int(ST('breakdown') * SR):int(ST('solo') * SR)] = E.db(1.5); _k = int(0.06 * SR); _benv = np.convolve(_benv, np.ones(_k) / _k, 'same'); b = (b.T * _benv[:len(b)]).T")
# MIX-2 harshness: vocal presence boost down
rep("('peak', 3500, 1.0, 2.5), ('hs', 8000, 0.7, 2.0))", "('peak', 3500, 1.0, 0.5), ('hs', 8000, 0.7, 2.0))")
# MIX-3 vocal peaks: vocal-only ceiling, vox bus +1, target -9
rep("v = E.compress(v, -12, 2.0, 0.004, 0.08, makeup_db=1.5); v = E.saturate(v, 1.4, 0.35)",
    "v = E.compress(v, -12, 2.0, 0.004, 0.08, makeup_db=1.5); v = E.saturate(v, 1.4, 0.35); v = E.limiter(v, -6.0, 0.002, 0.05)")
rep("L = dict(gtr=-6.0, lead=-12.0, clean=-12.0, bass=-7.0, drums=-5.0, orch=-12.0, fx=-9.0, vox=-3.0, voxfx=-7.0)",
    "L = dict(gtr=-6.0, lead=-12.0, clean=-12.0, bass=-7.0, drums=-5.0, orch=-12.0, fx=-9.0, vox=-2.0, voxfx=-7.0)")
rep("target = -10.0;", "target = -9.0;")
# MIX-4 ending: louder final hit, quieter click, drone ends 1.2 s after the hit
rep("put('fx', O.impact(0.9), ST('outro') + BAR * 3 + BEAT * 3, -5); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -6)",
    "put('fx', O.impact(1.0), ST('outro') + BAR * 3 + BEAT * 3, -1); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -3)")
rep("tr['vox'].put(E.stereo(E.read(os.path.join(HERE, '..', 'vox', 'w_k_click.wav'))), ST('outro') + BAR * 3 + BEAT * 2, 0)",
    "tr['vox'].put(E.stereo(E.read(os.path.join(HERE, '..', 'vox', 'w_k_click.wav'))), ST('outro') + BAR * 3 + BEAT * 2, -6)")
# MIX-4 verifier: keep the current drone line (already fades out by the hit)
# MIX-5 build: band less reduced, vocal a bit lower there
rep("'build': -2.5,", "'build': -1.0,")
rep("VOX_SEC_GAIN = {'first_entry': -1.0, 'climax': 0.5, 'outro': 2.5}", "VOX_SEC_GAIN = {'first_entry': -1.0, 'build': -2.0, 'climax': 0.5, 'outro': 2.5}")
# MIX-6 low-mid scoop: guitar 450 cut halved
rep("('peak', 450, 1.0, -2.0)", "('peak', 450, 1.0, -1.0)")
open('metal_song.py', 'w').write(s); print("metal_song.py mix fixes applied")
# MIX-4 stronger option: the band delivers the last hit (outro beat 15 = bar 52 beat 4)
d = json.load(open('riffs.json')); o = d['sections']['outro']
o.setdefault('guitar', []).append({"beat": 15.0, "root": 36, "dur": 2.0, "vel": 1.0, "mute": False})
o.setdefault('bass', []).append({"beat": 15.0, "midi": 24, "dur": 2.0, "vel": 1.0, "slide_from": None, "mute": False})
o.setdefault('drums', []).extend([{"beat": 15.0, "name": "kick", "vel": 1.0}, {"beat": 15.0, "name": "crash1", "vel": 1.0}, {"beat": 15.0, "name": "china", "vel": 0.9}])
json.dump(d, open('riffs.json', 'w')); print("riffs.json: band final hit added")
