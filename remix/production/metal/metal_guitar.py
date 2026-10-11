"""metal_guitar.py -- FINAL metal guitar module (Drop C, 48 kHz, float64), merged from the two candidate modules
after the objective comparison in tests/judge_guitar.py (run it with `metal_guitar` to re-measure this file).

Winner per function (tests/judge_guitar.py, A = metal_guitar_A physical model, B = metal_guitar_B oscillators):
  rhythm  -> A   penalty 0.00 vs 2.33: B's 120-250 Hz sits +9..+11 dB above 250-500 Hz (boomy), its chord spectrum
                 above 500 Hz is a speckled wash (flatness -8.7 dB) and its chugs end in a hard gate slam
                 (-70 dB at 150 ms); A's chugs decay naturally (-20 dB at 138 ms, HF closing first) and its chords
                 keep distinct harmonic lines. A's cab/amp chain (4x oversampled, min-phase 4x12 IR) also measures a
                 cleaner line-to-floor ratio.
  lead    -> A   both pass every pitch target (bend +2.24 / +2.07, vibrato 34 / 36 cents p-p at 5.5 Hz, slide
                 -2.7 / -2.3 -> 0); B's lead is a brick-wall-flat sustain with a dense noise floor below 500 Hz and an
                 abrupt end, A's has a natural attack/decay and clean harmonic lines.
  clean   -> A   A shows pick attacks and ringing lines with a brighter balance (2-4k -2.4 dB), B is a pad-like wash
                 (2-4k -9.9, 4-8k -26 dB, no visible attacks).
  extras  -> A   (all four peaks 0.44-0.55, DC < 1e-4, ~0.1 s each)

Adaptations made here (the only changes to the winning implementation):
  1. rhythm: post-cabinet low-end EQ (+4 dB @ 130 Hz, low shelf +2 dB @ 90 Hz, +1 dB @ 3 kHz) so the chugs have the
     SPEC's "strong 80-200 Hz" (thump 65-200 vs 250-500: +1.8 -> +4.6 dB on chugs, +0.3 -> +2.4 dB on chords) plus
     a fixed trim and a soft ceiling at 0.6 so peaks stay ~0.5 regardless of content.
  2. lead: legato fix. A faded every note over 120 ms after the next note had started, so 16th-note runs
     (136 ms apart) overlapped for most of their length through the amp. Now the fade is 20 ms when the next note
     overlaps and up to 120 ms (natural finger release) when it does not.
  3. clean: notes ring up to 1 s past their written duration, or until the same string is re-struck (an arpeggio
     "let ring"), with a 150 ms release, instead of being damped at `dur` (-37 dB one second after the last note).
  4. lead vibrato swings -0.25..+0.75 x depth around the fretted pitch instead of 0..+depth (A's vib=0.4 notes
     averaged +20 cents sharp against the band; now +11 cents, still upward-biased like finger vibrato).
  5. extras get the SPEC signatures with defaults: feedback_swell(midi, dur=3.0), pick_scrape(dur=1.2),
     pinch_harmonic(midi, dur=1.2), dive_bomb(midi, dur=2.0).
Measured (tests/judge_guitar.py, 110 BPM material): R1 peak 0.480 rms 0.116 crest 11.3 dB; chug thump +4.6 dB, chord
scoop -8.1, presence -1.8, rolloff 6-8k -27 dB, fizz 8-12k -49 dB; chug -20 dB at 138 ms; pair L/R wave corr 0.08,
env corr 0.76; lead bend +0.09 -> +2.13 semis, vibrato 34 c p-p at 5.5 Hz; clean peak 0.40, L/R corr 0.63.
Song-sized render (riffs_3.json, 119 s, 356/107/33 events, 1 CPU): pair 24 s + lead 9 s + clean 18 s = 52 s.

Public API (SPEC.md metal_guitar.py contract):
  render_rhythm(events, seed=0, total_len=None) -> mono (n,)         events: (t, root_midi, dur, vel, mute)
  render_rhythm_pair(events, total_len=None)    -> stereo (n, 2)     two takes (seeds 1 / 2), hard L / R
  render_lead(events, total_len=None)           -> mono              events: (t, midi, dur, vel, bend_semis, vibrato, slide_from)
  render_clean(events, total_len=None)          -> stereo            events: (t, midi, dur, vel)
  feedback_swell(midi, dur=3.0, vel=0.9), pick_scrape(dur=1.2, seed=0), pinch_harmonic(midi, dur=1.2, vel=1.0),
  dive_bomb(midi, dur=2.0, vel=1.0, depth=-26.0) -> mono one-shots
  diagnostics re-exported from metal_guitar_A: render_rhythm_di, render_lead_di, amp_sim, cab_ir, string_note, PRESETS
Length of every render = total_len seconds if given, else last event end + tail. Empty events -> zeros.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, '..'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine as E            # noqa: E402
import metal_guitar_A as A    # noqa: E402  (winning implementation)

SR = A.SR
CACHE = A.CACHE

# re-exported building blocks / diagnostics
string_note = A.string_note
render_rhythm_di = A.render_rhythm_di
amp_sim = A.amp_sim
cab_ir = A.cab_ir
PRESETS = A.PRESETS

# ---------------------------------------------------------------------------------------------- rhythm
RHYTHM_POST_EQ = [('peak', 130, 0.9, 4.0), ('ls', 90, 0.7, 2.0), ('peak', 3000, 1.0, 1.0)]   # SPEC: strong 80-200 Hz
RHYTHM_TRIM = 0.85          # fixed gain after the EQ: an R1 riff (vel 1 chugs + chord) peaks ~0.48
RHYTHM_KNEE, RHYTHM_CEIL = 0.50, 0.62


def _soft_ceiling(y, knee=RHYTHM_KNEE, ceiling=RHYTHM_CEIL):
    """linear below knee, tanh-compressed above; never exceeds ceiling. Only rare peaks are touched."""
    a = np.abs(y)
    over = a > knee
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (ceiling - knee) * np.tanh((a[over] - knee) / (ceiling - knee)))
    return y


def _rhythm_post(y):
    return _soft_ceiling(A._eq(y, RHYTHM_POST_EQ) * RHYTHM_TRIM)


def render_rhythm(events, seed=0, total_len=None):
    """(t, root_midi, dur, vel, mute) -> mono distorted rhythm guitar (one take). seed=0: no timing jitter."""
    di = A.render_rhythm_di(list(events), seed=seed, total_len=total_len, take=0)
    return _rhythm_post(A.amp_sim(di, 'rhythm', seed=seed))


def render_rhythm_pair(events, total_len=None):
    """two independent takes (seeds 1 / 2: L early / R late by 3-6 ms + 1 ms jitter per hit, +-4 cents global and
    +-3 cents per-string detune, random pick position / brightness, different cab IRs) hard-panned L / R."""
    events = list(events)
    L = _rhythm_post(A.amp_sim(A.render_rhythm_di(events, seed=1, total_len=total_len, take=1), 'rhythm', seed=1))
    R = _rhythm_post(A.amp_sim(A.render_rhythm_di(events, seed=2, total_len=total_len, take=2), 'rhythm', seed=2))
    return np.stack([L, R], 1)


# ---------------------------------------------------------------------------------------------- lead
VIB_BIAS = 0.25             # vibrato centre above the fretted pitch, as a fraction of its depth (A used 0.5)


def _lead_pitch_track(n, dur, bend, vib, slide_semis, r, vib_rate=5.5):
    """semitone offset per sample for a lead note: slide-in, bend-up, finger vibrato, as metal_guitar_A but with
    the vibrato swinging from -0.25*depth to +0.75*depth (A: 0..+depth, which left vib=0.4 notes +20 cents sharp)."""
    t = np.arange(n) / SR
    s = np.zeros(n)
    if slide_semis:
        tau = 0.10 if abs(slide_semis) <= 3 else 0.16
        s += slide_semis * (1 - A._smoothstep(t / tau))
    t_b = 0.0
    if bend:
        t_b = float(np.clip(0.4 * dur, 0.15, 0.45))
        s += bend * A._smoothstep(t / t_b) ** 1.3
    if vib:
        start = max(t_b, 0.12 if slide_semis else 0.18)
        depth = vib * A._smoothstep((t - start) / 0.35)
        rate = vib_rate * (0.95 + 0.1 * r.random())
        s += depth * (VIB_BIAS - 0.5 * np.cos(2 * np.pi * rate * (t - start) + r.uniform(0, 0.4)))
    return s


def render_lead_di(events, total_len=None, seed=5):
    """DI stream for lead events (t, midi, dur, vel, bend_semis, vibrato, slide_from). Same string / pitch model as
    metal_guitar_A.render_lead_di; the note release is 20 ms when the next note overlaps this one (legato / runs)
    and up to 120 ms otherwise."""
    r = np.random.default_rng(seed)
    events = list(events)
    if not events:
        return np.zeros(int((total_len or 1.0) * SR))
    end = max(e[0] + e[2] for e in events) + 1.5
    n_tot = int((total_len if total_len else end) * SR)
    out = np.zeros(n_tot)
    evs = sorted(events, key=lambda e: e[0])
    for k, ev in enumerate(evs):
        t, midi, dur, vel = float(ev[0]), int(ev[1]), float(ev[2]), float(np.clip(ev[3], 0.05, 1.0))
        bend = float(ev[4] or 0.0) if len(ev) > 4 else 0.0
        vib = float(ev[5] or 0.0) if len(ev) > 5 else 0.0
        sf = ev[6] if len(ev) > 6 else None
        nxt = float(evs[k + 1][0]) if k + 1 < len(evs) else 1e9
        cut = min(dur, max(0.05, nxt - t + 0.02))                       # legato: stop 20 ms into the next note
        fl = float(np.clip(nxt - (t + cut) + 0.03, 0.02, 0.12))         # release: 20 ms if overlapped, else <= 120 ms
        n_note = int((cut + fl + 0.05) * SR)
        semis = _lead_pitch_track(n_note, dur, bend, vib, (float(sf) - midi) if sf is not None else 0.0, r)
        tt = np.arange(n_note) / SR
        semis += 0.12 * vel ** 2 * np.exp(-tt / 0.035)                  # pick attack sharpening
        f0 = E.midi2hz(midi) * 2 ** (r.uniform(-2, 2) / 1200)
        n_src = int(np.ceil((2.0 ** (semis / 12)).sum())) + 16
        t60 = float(np.clip(6.0 * (130.8 / f0) ** 0.4, 1.5, 7.0))
        src = A.string_note(f0, n_src, vel * (0.6 if sf is not None else 1.0), False, seed=int(r.integers(1 << 30)),
                            pick_pos=0.13 + 0.05 * r.random(), t60=t60, hf_pole=0.22 + 0.1 * (1 - vel),
                            bright=2500 + 4000 * vel, settle=False)
        y = A._var_rate_read(src, semis)
        i0 = int(cut * SR); fk = int(fl * SR)
        if i0 < n_note:
            e = min(n_note, i0 + fk); y[i0:e] *= np.linspace(1, 0, e - i0) ** 0.5; y[e:] = 0
        A._place(out, y, int(round(t * SR)))
    return A._eq(A._antidenormal(out), [('lp', 4200, 1.3)])


def render_lead(events, total_len=None):
    """(t, midi, dur, vel, bend_semis, vibrato, slide_from) -> mono lead guitar (lead amp + compressor)."""
    di = render_lead_di(events, total_len=total_len)
    y = A.amp_sim(di, 'lead', seed=3)
    y = E.compress(y, -16, 2.5, 0.01, 0.15, makeup_db=2.0)
    return np.clip(y, -0.8, 0.8)


# ---------------------------------------------------------------------------------------------- clean
CLEAN_RING = 1.0            # seconds a clean note keeps ringing past its written duration (unless re-struck)
CLEAN_RELEASE = 0.15        # release time constant (s)


def render_clean(events, total_len=None):
    """(t, midi, dur, vel) -> stereo clean/ambient guitar: long-sustain strings, per-note pan spread, mild clean amp +
    open-back cab, compressor, chorus, reverb; notes ring up to CLEAN_RING s past dur or until the same note is
    re-struck. Normalised to a 0.4 peak."""
    r = np.random.default_rng(9)
    events = list(events)
    if not events:
        return np.zeros((int((total_len or 1.0) * SR), 2))
    evs = sorted([(float(t), int(m), float(d), float(v)) for (t, m, d, v) in events], key=lambda e: e[0])
    end = max(t + d for t, _, d, _ in evs) + CLEAN_RING + 2.0
    n_tot = int((total_len if total_len else end) * SR)
    out = np.zeros((n_tot, 2))
    for k, (t, midi, dur, vel) in enumerate(evs):
        vel = float(np.clip(vel, 0.05, 1.0))
        ring = dur + CLEAN_RING
        for (t2, m2, _, _) in evs[k + 1:]:                              # same string re-struck -> stop this one
            if m2 == midi and t2 > t:
                ring = min(ring, max(0.05, t2 - t - 0.004)); break
        n_note = int((ring + 5 * CLEAN_RELEASE) * SR)
        f0 = E.midi2hz(midi) * 2 ** (r.uniform(-3, 3) / 1200)
        s = A.string_note(f0, n_note, vel, False, seed=int(r.integers(1 << 30)), pick_pos=0.2 + 0.08 * r.random(),
                          t60=float(np.clip(5.0 * (130.8 / f0) ** 0.5, 1.5, 7.0)), hf_pole=0.3, bright=1500 + 3000 * vel)
        i0 = int(ring * SR)
        if i0 < n_note:
            s[i0:] *= np.exp(-np.arange(n_note - i0) / SR / CLEAN_RELEASE)
        p = 0.35 * np.sin(k * 1.7) + r.uniform(-0.15, 0.15)
        i = int(round(t * SR))
        A._place(out[:, 0], s * np.cos((p + 1) / 4 * np.pi), i)
        A._place(out[:, 1], s * np.sin((p + 1) / 4 * np.pi), i)
    out = np.stack([A._eq(A._antidenormal(out[:, c]), [('lp', 4200, 1.1)]) for c in range(2)], 1)   # pickup resonance
    y = np.stack([A.amp_sim(out[:, c], 'clean', seed=10 + c, gate=False) for c in range(2)], 1)
    y = E.compress(y, -22, 3.0, 0.006, 0.18, makeup_db=4.0)
    y = E.chorus(y, rate=0.7, depth=0.0028, mix=0.38, voices=2, base=0.011)
    y = E.reverb(y, decay=2.2, predelay=0.025, hf_damp=0.5, mix=0.22, lowcut=180, highcut=7000, seed=31)
    pk = np.abs(y).max()
    if pk > 1e-6:
        y = y * (0.4 / pk)
    return y


# ---------------------------------------------------------------------------------------------- extras
def feedback_swell(midi, dur=3.0, vel=0.9):
    """held note blooming into amp feedback (3rd partial + slow vibrato) over dur s -> mono (dur + 0.5 s)."""
    return A.feedback_swell(midi, dur, vel)


def pick_scrape(dur=1.2, seed=0):
    """pick edge dragged along the wound low string: falling resonance through the rhythm amp -> mono (dur s)."""
    return A.pick_scrape(dur, seed)


def pinch_harmonic(midi, dur=1.2, vel=1.0):
    """squealing pinch harmonic (3rd partial, wide 5.8 Hz vibrato) through the lead amp -> mono (dur + 0.3 s)."""
    return A.pinch_harmonic(midi, dur, vel)


def dive_bomb(midi, dur=2.0, vel=1.0, depth=-26.0):
    """whammy dive of `depth` semitones over dur s with an end wobble, lead amp -> mono (dur + 0.4 s)."""
    return A.dive_bomb(midi, dur, vel, depth)


__all__ = ['render_rhythm', 'render_rhythm_pair', 'render_lead', 'render_clean', 'feedback_swell', 'pick_scrape',
           'pinch_harmonic', 'dive_bomb', 'render_rhythm_di', 'render_lead_di', 'amp_sim', 'cab_ir', 'string_note', 'PRESETS']
