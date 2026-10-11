"""Round-2 code fixes (C1..C7). Run AFTER apply_fixes_round2_mix.py and apply_fixes_round2_arr.py."""
s = open('metal_song.py').read()
def rep(a, b):
    global s
    assert a in s, "MISSING: " + a[:80]; s = s.replace(a, b)
# C1 lead sidechain depth
rep("ld = E.compress(ld, -20, 3.0, 0.012, 0.18, knee=4, sidechain=vkey)", "ld = E.compress(ld, -12, 2.0, 0.012, 0.18, knee=4, sidechain=vkey)")
# C2 gtr/orch sidechain ratio (orch was raised to 2.5 by the ARR-2 verdict; the code reviewer measured the shouts being ducked instead of speech → cap both at 1.2)
rep("g = E.compress(g, -14, 1.5, 0.012, 0.18, knee=4, sidechain=vkey)", "g = E.compress(g, -14, 1.2, 0.012, 0.18, knee=4, sidechain=vkey)")
rep("o = E.compress(o, -14, 2.5, 0.012, 0.2, knee=4, sidechain=vkey)", "o = E.compress(o, -14, 1.2, 0.012, 0.2, knee=4, sidechain=vkey)")
# C3 de-esser (single-band, keyed by HF) between saturate and the vocal limiter
rep("v = E.saturate(v, 1.4, 0.35); v = E.limiter(v, -6.0, 0.002, 0.05)",
    "v = E.saturate(v, 1.4, 0.35)\ndek = E.mono(E.eq(v, ('hp', 4500, 0.7), ('hp', 4500, 0.7))); v = E.compress(v, -12, 3.0, 0.002, 0.06, knee=4, sidechain=dek)   # de-esser\nv = E.limiter(v, -6.0, 0.002, 0.05)")
# C4 loudness: iterate gain -> limiter -> measure until within 0.15 LU (verifier: two passes undershoot)
rep("target = -9.0; m = E.limiter(E.gain(m, target - meas['lufs']), -1.0, 0.004, 0.12)",
    """target = -9.5; g0 = target - meas['lufs']; tmp = os.path.join(OUT, '_lim_%s.wav' % MODE)
for _ in range(4):
    m2 = E.limiter(E.gain(m, g0), -1.0, 0.004, 0.12); E.write(tmp, m2); err = target - E.measure(tmp)['lufs']
    if abs(err) <= 0.15: break
    g0 += err
m = m2; print("master gain %.2f dB" % g0)""")
# C5 final hit belongs to the FX layer (both modes)
hit = "    put('fx', O.impact(1.0), ST('outro') + BAR * 3 + BEAT * 3, -1); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -3)\n"
rep(hit, "")
rep("put('fx', E.fade(O.drone(36, BAR * 3 + BEAT * 3.2), 0.4, 1.8), ST('outro'), -10)   # fades into the final hit, which is the last sound",
    "put('fx', E.fade(O.drone(36, BAR * 3 + BEAT * 3.2), 0.4, 1.8), ST('outro'), -10)   # fades into the final hit, which is the last sound\nput('fx', O.impact(1.0), ST('outro') + BAR * 3 + BEAT * 3, -1); put('fx', O.sub_drop(1.6, 60, 20), ST('outro') + BAR * 3 + BEAT * 3, -3)   # the last hit (both modes)")
# C6 drum jitter: kicks/snares tight, cymbals/hats human
rep("for e in sec.get('drums', []):   drums.append((t0 + e['beat'] * BEAT + rng.normal(0, 0.003), e['name'], float(e['vel'])))",
    "for e in sec.get('drums', []):   drums.append((t0 + e['beat'] * BEAT + rng.normal(0, 0.001 if e['name'] in ('kick', 'snare') else 0.0025), e['name'], float(e['vel'])))")
# C7 not confirmed by the verifier (shift would worsen the climax overlap): no change
open('metal_song.py', 'w').write(s); print("code fixes applied")
