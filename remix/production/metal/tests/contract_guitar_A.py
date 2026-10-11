"""Contract check: the exact calls metal_song.py / tests/integration_smoke.py make on metal_guitar, run against metal_guitar_A."""
import sys, os, numpy as np, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import metal_guitar_A as G
BEAT = 60 / 110
def chk(name, x, stereo=None):
    x = np.asarray(x); assert x.ndim in (1, 2), (name, x.shape)
    if stereo is True: assert x.ndim == 2 and x.shape[1] == 2, (name, x.shape)
    assert np.isfinite(x).all(), (name, 'nan')
    pk = np.abs(x).max(); print("  %-28s shape %-14s peak %.3f dc %.4f" % (name, x.shape, pk, x.mean()))
    assert pk <= 1.05, (name, pk)
t0 = time.time()
chk('render_rhythm', G.render_rhythm([(0.0, 36, 0.12, 1.0, True), (BEAT, 36, 0.12, 0.9, True), (2 * BEAT, 39, 1.5, 1.0, False)], seed=1, total_len=5.0))
chk('render_rhythm_pair', G.render_rhythm_pair([(0.0, 36, 0.12, 1.0, True), (BEAT, 43, 1.0, 1.0, False)], total_len=4.0), stereo=True)
chk('render_lead', G.render_lead([(0.0, 67, 1.0, 0.9, 0.0, 0.3, None), (1.0, 70, 1.5, 0.9, 2.0, 0.4, 67)], total_len=4.0))
chk('render_clean', G.render_clean([(0.0, 48, 0.5, 0.6), (0.3, 55, 0.5, 0.6), (0.6, 60, 1.0, 0.6)], total_len=3.0), stereo=True)
# edge cases: empty event lists, total_len shorter than the events, event beyond total_len
chk('render_rhythm empty', G.render_rhythm([], total_len=2.0)); chk('render_lead empty', G.render_lead([], total_len=2.0)); chk('render_clean empty', G.render_clean([], total_len=2.0), stereo=True)
chk('rhythm short total_len', G.render_rhythm([(1.5, 36, 3.0, 1.0, False), (2.5, 36, 0.1, 1.0, True)], total_len=2.0))
assert len(G.render_rhythm([(0.0, 36, 1.0, 1.0, False)], total_len=7.0)) == 7 * 48000
for n in ['feedback_swell', 'pick_scrape', 'pinch_harmonic', 'dive_bomb']: assert callable(getattr(G, n))
print("CONTRACT OK in %.1fs" % (time.time() - t0))
