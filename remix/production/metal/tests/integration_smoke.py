"""Contract check: every module renders tiny event lists with the signatures metal_song.py uses."""
import sys, os, numpy as np, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import engine as E
BEAT = 60 / 110
def chk(name, x, stereo=None):
    x = np.asarray(x); assert x.ndim in (1, 2), (name, x.shape)
    if stereo is True: assert x.ndim == 2 and x.shape[1] == 2, (name, x.shape)
    assert np.isfinite(x).all(), (name, 'nan')
    pk = np.abs(x).max(); print("  %-28s shape %-14s peak %.3f dc %.4f" % (name, x.shape, pk, x.mean()))
    assert pk <= 1.05, (name, pk)
t0 = time.time()
try:
    import metal_guitar as G
except ImportError:
    G = None; print('  (metal_guitar not available yet)')
if G: chk('render_rhythm', G.render_rhythm([(0.0, 36, 0.12, 1.0, True), (BEAT, 36, 0.12, 0.9, True), (2 * BEAT, 39, 1.5, 1.0, False)], seed=1, total_len=5.0))
if G: chk('render_rhythm_pair', G.render_rhythm_pair([(0.0, 36, 0.12, 1.0, True), (BEAT, 43, 1.0, 1.0, False)], total_len=4.0), stereo=True)
if G: chk('render_lead', G.render_lead([(0.0, 67, 1.0, 0.9, 0.0, 0.3, None), (1.0, 70, 1.5, 0.9, 2.0, 0.4, 67)], total_len=4.0))
if G: chk('render_clean', G.render_clean([(0.0, 48, 0.5, 0.6), (0.3, 55, 0.5, 0.6), (0.6, 60, 1.0, 0.6)], total_len=3.0), stereo=True)
import metal_bass as B
chk('render_bass', B.render_bass([(0.0, 24, 0.12, 1.0, None, True), (0.5, 27, 1.0, 0.9, 29, False)], total_len=3.0))
import metal_drums as DR
chk('render_kit', DR.render_kit([(0.0, 'kick', 1.0), (0.0, 'crash1', 0.9), (BEAT, 'snare', 1.0), (BEAT * 1.5, 'china', 0.8), (BEAT * 2, 'tom_floor', 0.9), (BEAT * 2.25, 'hat_closed', 0.5), (BEAT * 2.5, 'ride', 0.6), (BEAT * 3, 'snare_ghost', 0.4)], total_len=4.0), stereo=True)
for n in ['kick', 'snare', 'snare_ghost', 'tom_hi', 'tom_mid', 'tom_floor', 'hat_closed', 'hat_open', 'hat_pedal', 'ride', 'ride_bell', 'crash1', 'crash2', 'china', 'splash']:
    chk('hit:' + n, DR.hit(n, 0.9))
import metal_orch_fx as O
chk('orch', O.orch([(0.0, 48, 2.0, 0.7), (0.0, 55, 2.0, 0.7)], 48, total_len=4.0), stereo=True)
for fn, args in [('drone', (36, 3.0)), ('riser', (2.0, 200, 6000)), ('impact', (1.0,)), ('sub_drop', (1.2, 70, 25)), ('downlifter', (2.0,)), ('metal_hit', (0.7, 420.0)), ('reverse_cymbal', (1.5,)), ('noise_bed', (3.0,)), ('braam', (36, 2.0)), ('industrial_clank', (1,))]:
    chk(fn, getattr(O, fn)(*args))
import metal_vocal as MV
for kind in ['clean', 'hall', 'double', 'demon', 'shout_stack', 'megaphone', 'radio', 'glitch', 'slowmo', 'tele']:
    chk('vocal:' + kind, MV.variant('meham3', kind), stereo=True)
chk('vocal:stutter', MV.variant('mu1', 'stutter', reps=4, grid=0.25), stereo=True)
r = MV.variant('meham3', 'reverse_swell'); x, lead_in = (r if isinstance(r, tuple) else (r, None)); chk('vocal:reverse_swell', x, stereo=True); print("  reverse_swell lead_in:", lead_in)
print("ALL CONTRACTS OK in %.0fs" % (time.time() - t0))
